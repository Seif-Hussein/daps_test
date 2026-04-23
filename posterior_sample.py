import json
import yaml
import torch
from torchvision.utils import save_image
from forward_operator import get_operator
from data import get_dataset
from sampler import get_sampler, Trajectory
from model import get_model
from eval import get_eval_fn, get_eval_fn_cmp, Evaluator
from torch.nn.functional import interpolate
from pathlib import Path
from omegaconf import OmegaConf
from evaluate_fid import calculate_fid
from torch.utils.data import DataLoader
import hydra
from PIL import Image
import numpy as np
import imageio

import os
from datetime import datetime, timezone
from time import perf_counter

try:
    import wandb
except ImportError:
    wandb = None

try:
    import setproctitle
except ImportError:
    setproctitle = None


def resize(y, x, task_name):
    """
        Visualization Only: resize measurement y according to original signal image x
    """
    if y.shape != x.shape:
        ry = interpolate(y, size=x.shape[-2:], mode='bilinear', align_corners=False)
    else:
        ry = y
    if task_name == 'phase_retrieval':
        def norm_01(y):
            tmp = (y - y.mean()) / y.std()
            tmp = tmp.clip(-0.5, 0.5) * 3
            return tmp

        ry = norm_01(ry) * 2 - 1
    return ry


def safe_dir(dir):
    """
        get (or create) a directory
    """
    if not Path(dir).exists():
        Path(dir).mkdir()
    return Path(dir)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def to_python_scalar(x):
    if torch.is_tensor(x):
        x = x.detach().cpu()
        if x.numel() == 1:
            return x.item()
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def write_json(path, data):
    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    with open(tmp_path, 'w') as file:
        json.dump(data, file, indent=4)
    tmp_path.replace(path)


def build_progress_state(
    *,
    args,
    total_images,
    run_id,
    num_runs,
    step,
    num_steps,
    batch_start,
    batch_end,
    sigma=None,
    x0hat_results=None,
    x0y_results=None,
    status='running',
    message=None,
    started_at=None,
    elapsed_seconds=None,
    elapsed_seconds_per_image=None,
):
    progress = {
        'status': status,
        'message': message,
        'updated_at': now_iso(),
        'run_id': run_id,
        'num_runs': num_runs,
        'step': step,
        'num_steps': num_steps,
        'batch_start': batch_start,
        'batch_end': batch_end,
        'batch_size': batch_end - batch_start,
        'total_images': total_images,
        'task_name': args.task[args.task_group].operator.name,
        'run_name': args.name,
    }
    if started_at is not None:
        progress['started_at'] = started_at
    if elapsed_seconds is not None:
        progress['elapsed_seconds'] = elapsed_seconds
    if elapsed_seconds_per_image is not None:
        progress['elapsed_seconds_per_image'] = elapsed_seconds_per_image

    if sigma is not None:
        progress['sigma'] = to_python_scalar(sigma)

    for state_name, results in [('x0hat', x0hat_results), ('x0y', x0y_results)]:
        if results:
            progress[state_name] = {}
            for metric_name, metric_value in results.items():
                metric_value = metric_value.detach().cpu()
                progress[state_name][metric_name] = {
                    'mean': metric_value.mean().item(),
                    'std': metric_value.std().item() if metric_value.numel() > 1 else 0.0,
                    'min': metric_value.min().item(),
                    'max': metric_value.max().item(),
                }
    if x0y_results:
        progress['latest_metrics'] = {}
        for metric_name, metric_value in x0y_results.items():
            metric_mean = metric_value.detach().cpu().mean().item()
            progress['latest_metrics'][metric_name] = metric_mean
            progress[metric_name] = metric_mean
    return progress


def build_metric_history_row(
    *,
    run_id,
    step,
    num_steps,
    batch_start,
    batch_end,
    elapsed_seconds,
    sigma=None,
    x0hat_results=None,
    x0y_results=None,
):
    batch_size = batch_end - batch_start
    row = {
        'run_id': run_id,
        'step': step,
        'max_iter': num_steps,
        'batch_start': batch_start,
        'batch_end': batch_end,
        'batch_size': batch_size,
        'elapsed_seconds': elapsed_seconds,
        'elapsed_seconds_per_image': elapsed_seconds / max(batch_size, 1),
    }
    if sigma is not None:
        row['sigma'] = to_python_scalar(sigma)

    # Match the tuning notebooks: top-level metrics track the current corrected
    # estimate after the inverse-problem update.
    for metric_name, metric_value in (x0y_results or {}).items():
        metric_value = metric_value.detach().cpu()
        row[metric_name] = metric_value.mean().item()

    for metric_name, metric_value in (x0hat_results or {}).items():
        metric_value = metric_value.detach().cpu()
        row[f'x0hat_{metric_name}'] = metric_value.mean().item()

    return row


def norm(x):
    """
        normalize data to [0, 1] range
    """
    return (x * 0.5 + 0.5).clip(0, 1)


def tensor_to_pils(x):
    """
        [B, C, H, W] tensor -> list of pil images
    """
    pils = []
    for x_ in x:
        np_x = norm(x_).permute(1, 2, 0).cpu().numpy() * 255
        np_x = np_x.astype(np.uint8)
        pil_x = Image.fromarray(np_x)
        pils.append(pil_x)
    return pils


def tensor_to_numpy(x):
    """
        [B, C, H, W] tensor -> [B, C, H, W] numpy
    """
    np_images = norm(x).permute(0, 2, 3, 1).cpu().numpy() * 255
    return np_images.astype(np.uint8)


def save_mp4_video(gt, y, x0hat_traj, x0y_traj, xt_traj, output_path, fps=24, sec=5, space=4):
    """
        stack and save trajectory as mp4 video
    """
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8)
    ix, iy = x0hat_traj.shape[-2:]
    reindex = np.linspace(0, len(xt_traj) - 1, sec * fps).astype(int)
    np_x0hat_traj = tensor_to_numpy(x0hat_traj[reindex])
    np_x0y_traj = tensor_to_numpy(x0y_traj[reindex])
    np_xt_traj = tensor_to_numpy(xt_traj[reindex])
    np_y = tensor_to_numpy(y[None])[0]
    np_gt = tensor_to_numpy(gt[None])[0]
    for x0hat, x0y, xt in zip(np_x0hat_traj, np_x0y_traj, np_xt_traj):
        canvas = np.ones((ix, 5 * iy + 4 * space, 3), dtype=np.uint8) * 255
        cx = cy = 0
        canvas[cx:cx + ix, cy:cy + iy] = np_y

        cy += iy + space
        canvas[cx:cx + ix, cy:cy + iy] = np_gt

        cy += iy + space
        canvas[cx:cx + ix, cy:cy + iy] = x0y

        cy += iy + space
        canvas[cx:cx + ix, cy:cy + iy] = x0hat

        cy += iy + space
        canvas[cx:cx + ix, cy:cy + iy] = xt
        writer.append_data(canvas)
    writer.close()


def sample_in_batch(
    sampler,
    model,
    x_start,
    operator,
    y,
    evaluator,
    verbose,
    record,
    record_metrics,
    batch_size,
    gt,
    args,
    root,
    run_id,
    progress_path=None,
    started_at=None,
    run_start_perf_counter=None,
    metric_history_rows=None,
):
    """
        posterior sampling in batch
    """
    samples = []
    trajs = []
    for s in range(0, len(x_start), batch_size):
        # update evaluator to correct batch index
        cur_x_start = x_start[s:s + batch_size]
        cur_y = y[s:s + batch_size]
        cur_gt = gt[s: s + batch_size]
        sampler.progress_callback = None
        if progress_path is not None:
            def progress_callback(payload, batch_start=s, batch_end=min(s + batch_size, len(x_start))):
                elapsed_seconds = perf_counter() - run_start_perf_counter if run_start_perf_counter is not None else None
                elapsed_seconds_per_image = None
                if elapsed_seconds is not None:
                    elapsed_seconds_per_image = elapsed_seconds / max(batch_end - batch_start, 1)

                if metric_history_rows is not None and elapsed_seconds is not None:
                    metric_history_rows.append(build_metric_history_row(
                        run_id=run_id,
                        step=payload['step'],
                        num_steps=payload['num_steps'],
                        batch_start=batch_start,
                        batch_end=batch_end,
                        elapsed_seconds=elapsed_seconds,
                        sigma=payload['sigma'],
                        x0hat_results=payload.get('x0hat_results'),
                        x0y_results=payload.get('x0y_results'),
                    ))

                progress_update_every = max(1, int(args.progress_update_every))
                is_progress_update = (
                    payload['step'] % progress_update_every == 0
                    or payload['step'] == payload['num_steps'] - 1
                )
                if not is_progress_update:
                    return

                progress_state = build_progress_state(
                    args=args,
                    total_images=len(x_start),
                    run_id=run_id,
                    num_runs=args.num_runs,
                    step=payload['step'],
                    num_steps=payload['num_steps'],
                    batch_start=batch_start,
                    batch_end=batch_end,
                    sigma=payload['sigma'],
                    x0hat_results=payload.get('x0hat_results'),
                    x0y_results=payload.get('x0y_results'),
                    status='running',
                    started_at=started_at,
                    elapsed_seconds=elapsed_seconds,
                    elapsed_seconds_per_image=elapsed_seconds_per_image,
                )
                progress_state['max_iter'] = payload['num_steps']
                write_json(progress_path, progress_state)
            sampler.progress_callback = progress_callback
        cur_samples = sampler.sample(
            model,
            cur_x_start,
            operator,
            cur_y,
            evaluator,
            verbose=verbose,
            record=record,
            record_metrics=record_metrics,
            gt=cur_gt,
        )

        samples.append(cur_samples)
        if record or record_metrics:
            cur_trajs = sampler.trajectory.compile()
            trajs.append(cur_trajs)

        # log individual sample instances
        if args.save_samples:
            pil_image_list = tensor_to_pils(cur_samples)
            image_dir = safe_dir(root / 'samples')
            for idx in range(batch_size):
                image_path = image_dir / '{:05d}_run{:04d}.png'.format(idx+s, run_id)
                pil_image_list[idx].save(str(image_path))

        # log sampling trajectory and mp4 video
        if args.save_traj:
            traj_dir = safe_dir(root / 'trajectory')
            # save mp4 video for trajectories
            x0hat_traj = cur_trajs.tensor_data['x0hat']
            x0y_traj = cur_trajs.tensor_data['x0y']
            xt_traj = cur_trajs.tensor_data['xt']
            cur_resized_y = resize(cur_y, cur_samples, args.task[args.task_group].operator.name)
            slices = np.linspace(0, len(x0hat_traj)-1, 10).astype(int)
            slices = np.unique(slices)
            for idx in range(batch_size):
                if args.save_traj_video:
                    video_path = str(traj_dir / '{:05d}_run{:04d}.mp4'.format(idx+s, run_id))
                    save_mp4_video(cur_samples[idx], cur_resized_y[idx], x0hat_traj[:, idx], x0y_traj[:, idx], xt_traj[:, idx], video_path)
                # save long grid images
                selected_traj_grid = torch.cat([x0y_traj[slices, idx], x0hat_traj[slices, idx], xt_traj[slices, idx]], dim=0)
                traj_grid_path = str(traj_dir / '{:05d}_run{:04d}.png'.format(idx+s, run_id))
                save_image(selected_traj_grid * 0.5 + 0.5, fp=traj_grid_path, nrow=len(slices))
        
    if record or record_metrics:
        trajs = Trajectory.merge(trajs)
    else:
        trajs = None
    return torch.cat(samples, dim=0), trajs


def summarize_metric_trajectory(traj):
    sigma = traj.value_data['sigma']
    if sigma.ndim > 1:
        sigma = sigma[:, 0]

    summary = {
        'num_iterations': int(len(sigma)),
        'iteration': list(range(int(len(sigma)))),
        'sigma': sigma.tolist(),
        'x0hat': {},
        'x0y': {},
    }

    for state_name in ['x0hat', 'x0y']:
        prefix = f'{state_name}_'
        metric_names = sorted(name[len(prefix):] for name in traj.value_data.keys() if name.startswith(prefix))
        for metric_name in metric_names:
            values = traj.value_data[f'{prefix}{metric_name}']
            if values.ndim == 1:
                values = values[:, None]
            summary[state_name][metric_name] = {
                'mean': values.mean(dim=1).tolist(),
                'std': (values.std(dim=1) if values.shape[1] != 1 else torch.zeros(values.shape[0])).tolist(),
                'min': values.min(dim=1).values.tolist(),
                'max': values.max(dim=1).values.tolist(),
            }
    return summary


def summarize_metric_trajectories(trajs):
    run_summaries = [summarize_metric_trajectory(traj) for traj in trajs]
    summary = {'runs': run_summaries}

    if len(run_summaries) == 1:
        summary['aggregate_across_runs'] = run_summaries[0]
        return summary

    aggregate = {
        'num_iterations': run_summaries[0]['num_iterations'],
        'iteration': run_summaries[0]['iteration'],
        'sigma': run_summaries[0]['sigma'],
        'x0hat': {},
        'x0y': {},
    }

    for state_name in ['x0hat', 'x0y']:
        metric_names = run_summaries[0][state_name].keys()
        for metric_name in metric_names:
            aggregate[state_name][metric_name] = {}
            stat_names = run_summaries[0][state_name][metric_name].keys()
            for stat_name in stat_names:
                stat_values = np.array([run[state_name][metric_name][stat_name] for run in run_summaries])
                aggregate[state_name][metric_name][stat_name] = stat_values.mean(axis=0).tolist()
                if stat_name == 'mean':
                    aggregate[state_name][metric_name]['std_across_runs'] = stat_values.std(axis=0).tolist()

    summary['aggregate_across_runs'] = aggregate
    return summary


@hydra.main(version_base='1.3', config_path='configs', config_name='default.yaml')
def main(args):
    # fix random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device('cuda:{}'.format(args.gpu))

    if setproctitle is not None:
        setproctitle.setproctitle(args.name)
    print(yaml.dump(OmegaConf.to_container(args, resolve=True), indent=4))

    # get data
    dataset = get_dataset(**args.data)
    total_number = len(dataset)
    images = dataset.get_data(total_number, 0)

    # get operator & measurement
    task_group = args.task[args.task_group]
    operator = get_operator(**task_group.operator)
    y = operator.measure(images)

    # get sampler
    sampler = get_sampler(**args.sampler, mcmc_sampler_config=task_group.mcmc_sampler_config)

    # get model
    model = get_model(**args.model)

    # get evaluator
    eval_fn_list = []
    for eval_fn_name in args.eval_fn_list:
        eval_fn_list.append(get_eval_fn(eval_fn_name))
    evaluator = Evaluator(eval_fn_list)

    # log hyperparameters and configurations
    os.makedirs(args.save_dir, exist_ok=True)
    save_dir = safe_dir(Path(args.save_dir))
    root = safe_dir(save_dir / args.name)
    progress_path = root / 'progress.json'
    metric_history_path = root / 'metric_history.json'
    with open(str(root / 'config.yaml'), 'w') as file:
        yaml.safe_dump(OmegaConf.to_container(args, resolve=True), file, default_flow_style=False, allow_unicode=True)
    write_json(progress_path, {
        'status': 'initialized',
        'message': 'Starting run',
        'updated_at': now_iso(),
        'run_name': args.name,
        'num_runs': args.num_runs,
        'total_images': total_number,
        'task_name': args.task[args.task_group].operator.name,
    })

    # logging to wandb
    if args.wandb:
        if wandb is None:
            raise ImportError("wandb is not installed but args.wandb=True")
        wandb.init(
            project=args.project_name,
            name=args.name,
            config=OmegaConf.to_container(args, resolve=True)
        )

    # +++++++++++++++++++++++++++++++++++
    # main sampling process
    full_samples = []
    full_trajs = []
    metric_history_rows = []
    record_metrics = args.save_evolution_metrics
    for r in range(args.num_runs):
        print(f'Run: {r}')
        run_started_at = now_iso()
        run_start_perf_counter = perf_counter()
        write_json(progress_path, {
            'status': 'running',
            'message': 'Sampling in progress',
            'updated_at': run_started_at,
            'started_at': run_started_at,
            'elapsed_seconds': 0.0,
            'run_id': r,
            'num_runs': args.num_runs,
            'step': 0,
            'num_steps': sampler.annealing_scheduler.num_steps - 1,
            'batch_start': 0,
            'batch_end': 0,
            'batch_size': 0,
            'total_images': total_number,
            'task_name': args.task[args.task_group].operator.name,
            'run_name': args.name,
        })
        x_start = sampler.get_start(images.shape[0], model)
        samples, trajs = sample_in_batch(
            sampler,
            model,
            x_start,
            operator,
            y,
            evaluator,
            verbose=True,
            record=args.save_traj,
            record_metrics=record_metrics,
            batch_size=args.batch_size,
            gt=images,
            args=args,
            root=root,
            run_id=r,
            progress_path=progress_path,
            started_at=run_started_at,
            run_start_perf_counter=run_start_perf_counter,
            metric_history_rows=metric_history_rows,
        )
        full_samples.append(samples)
        if trajs is not None:
            full_trajs.append(trajs)
    full_samples = torch.stack(full_samples, dim=0)

    # evaluate and log metrics
    results = evaluator.report(images, y, full_samples)    
    if args.wandb:
        evaluator.log_wandb(results, args.batch_size)
    markdown_text = evaluator.display(results)
    with open(str(root / 'eval.md'), 'w') as file:
        file.write(markdown_text)
    json.dump(results, open(str(root / 'metrics.json'), 'w'), indent=4)
    print(markdown_text)

    if args.save_evolution_metrics:
        evolution_metrics = summarize_metric_trajectories(full_trajs)
        json.dump(evolution_metrics, open(str(root / 'metrics_evolution.json'), 'w'), indent=4)

    metric_history = {
        'metric_source': 'x0y',
        'time_unit': 'seconds',
        'elapsed_seconds_per_image_definition': 'elapsed_seconds / batch_size',
        'run_name': args.name,
        'task_name': args.task[args.task_group].operator.name,
        'num_runs': args.num_runs,
        'total_images': total_number,
        'rows': metric_history_rows,
    }
    write_json(metric_history_path, metric_history)

    # log grid results
    resized_y = resize(y, images, args.task[args.task_group].operator.name)
    stack = torch.cat([images, resized_y, full_samples.flatten(0, 1)])
    save_image(stack * 0.5 + 0.5, fp=str(root / 'grid_results.png'), nrow=total_number)
    
    # save raw trajectory data
    if args.save_traj_raw_data:
        traj_dir = safe_dir(root / 'trajectory')
        for run, sde_traj in enumerate(full_trajs):
            # might be SUPER LARGE
            print(f'saving trajectory run {run}...')
            traj_raw_data = safe_dir(traj_dir / 'raw')
            torch.save(sde_traj, str(traj_raw_data / 'trajectory_run{:04d}.pth'.format(run)))
    
    # evaluate FID score
    if args.eval_fid:
        print('Calculating FID...')
        fid_dir = safe_dir(root / 'fid')
        # select the best samples based on the best of the all runs
        full_samples # [num_runs, B, C, H, W]
        eval_fn_cmp = get_eval_fn_cmp(evaluator.main_eval_fn_name)
        eval_values = np.array(results[evaluator.main_eval_fn_name]['sample']) # [B, num_runs]
        if eval_fn_cmp == 'min':
            best_idx = np.argmin(eval_values, axis=1)
        elif eval_fn_cmp == 'max':
            best_idx = np.argmax(eval_values, axis=1)
        best_samples = full_samples[best_idx, np.arange(full_samples.shape[1])]
        # save the best samples
        best_sample_dir = safe_dir(fid_dir / 'best_sample')
        pil_image_list = tensor_to_pils(best_samples)
        for idx in range(len(pil_image_list)):
            image_path = best_sample_dir / '{:05d}.png'.format(idx)
            pil_image_list[idx].save(str(image_path))

        fake_dataset = get_dataset(args.data.name, resolution=args.data.resolution, root=str(best_sample_dir))
        real_loader = DataLoader(dataset, batch_size=100, shuffle=False)
        fake_loader = DataLoader(fake_dataset, batch_size=100, shuffle=False)

        fid_score = calculate_fid(real_loader, fake_loader)
        print(f'FID Score: {fid_score.item():.4f}')
        with open(str(fid_dir / 'fid.txt'), 'w') as file:
            file.write(f'FID Score: {fid_score.item():.4f}')
        if args.wandb:
            wandb.log({'FID': fid_score.item()})

    write_json(progress_path, {
        'status': 'completed',
        'message': f'finish {args.name}!',
        'updated_at': now_iso(),
        'started_at': run_started_at,
        'elapsed_seconds': perf_counter() - run_start_perf_counter,
        'run_name': args.name,
        'num_runs': args.num_runs,
        'total_images': total_number,
        'task_name': args.task[args.task_group].operator.name,
        'metrics_path': str(root / 'metrics.json'),
        'metrics_evolution_path': str(root / 'metrics_evolution.json'),
        'metric_history_path': str(metric_history_path),
    })
    print(f'finish {args.name}!')


if __name__ == '__main__':
    main()
