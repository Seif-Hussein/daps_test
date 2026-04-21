# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved

import datetime
import logging
import os
import shutil
import time
import traceback
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard as tb
import torchvision.utils as tvu
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from algos import build_algo
from datasets import build_loader
from models import build_model
from models.classifier_guidance_model import ClassifierGuidanceModel
from models.diffusion import Diffusion
from utils.distributed import get_logger, init_processes, common_init
from utils.functions import get_timesteps, postprocess, preprocess, strfdt
from utils.degredations import get_degreadation_image
from utils.progress import write_json_file, write_progress_json
from utils.save import save_result

torch.set_printoptions(sci_mode=False)


def main(cfg):
    print('cfg.exp.seed', cfg.exp.seed)
    print(cfg)
    common_init(dist.get_rank(), seed=cfg.exp.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(dist.get_rank())
    
    # import pdb; pdb.set_trace()
    
    logger = get_logger(name="main", cfg=cfg)
    logger.info(f'Experiment name is {cfg.exp.name}')
    exp_root = cfg.exp.root
    samples_root = cfg.exp.samples_root
    exp_name = cfg.exp.name
    samples_root = os.path.join(exp_root, samples_root, exp_name)
    progress_filename = getattr(cfg.exp, "progress_json", "progress.json")
    progress_path = os.path.join(samples_root, progress_filename)
    write_progress = bool(getattr(cfg.exp, "write_progress_json", True))
    metric_history_filename = getattr(cfg.exp, "metrics_history_json", "metrics_history.json")
    metric_history_path = os.path.join(samples_root, metric_history_filename)
    write_metric_history = bool(getattr(cfg.exp, "write_metrics_history_json", True))
    run_started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    dataset_name = cfg.dataset.name
    if dist.get_rank() == 0:
        if cfg.exp.overwrite:
            if os.path.exists(samples_root):
                shutil.rmtree(samples_root)
            os.makedirs(samples_root)
        else:
            if not os.path.exists(samples_root):
                os.makedirs(samples_root)
        if write_progress:
            write_progress_json(
                progress_path,
                {
                    "status": "initializing",
                    "experiment_name": exp_name,
                    "dataset_name": dataset_name,
                    "algorithm": cfg.algo.name,
                    "inverse_problem": cfg.algo.deg,
                    "seed": int(cfg.exp.seed),
                    "samples_root": samples_root,
                    "metric_history_path": metric_history_path,
                    "started_at": run_started_at,
                },
            )

            
    model, classifier = build_model(cfg)
    model.eval()
    if classifier is not None:
        classifier.eval()
    loader = build_loader(cfg)
    logger.info(f'Dataset size is {len(loader.dataset)}')
    diffusion = Diffusion(**cfg.diffusion)
    cg_model = ClassifierGuidanceModel(model, classifier, diffusion, cfg)   #?? what is the easiest way to call stable diffusion?

    algo = build_algo(cg_model, cfg)
    if "ddrm" in cfg.algo.name or "mcg" in cfg.algo.name or "dps" in cfg.algo.name or "pgdm" in cfg.algo.name or "reddiff" in cfg.algo.name:
        H = algo.H

    psnrs = []
    ssims = []
    lpips_scores = []
    metric_history_entries = []
    start_time = time.time()
    dataset_size = len(loader.dataset)
    planned_batches = len(loader)
    if cfg.exp.smoke_test > 0:
        planned_batches = min(planned_batches, cfg.exp.smoke_test)
    planned_images = min(dataset_size, planned_batches * cfg.loader.batch_size)
    completed_batches = 0
    completed_images = 0
    lpips_metric = None
    if write_metric_history:
        from torchmetrics.functional import structural_similarity_index_measure
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        lpips_metric = LearnedPerceptualImagePatchSimilarity(normalize=True, reduction="none")
        if torch.cuda.is_available():
            lpips_metric = lpips_metric.cuda()
        lpips_metric.eval()

    if dist.get_rank() == 0 and write_progress:
        write_progress_json(
            progress_path,
            {
                "status": "running",
                "experiment_name": exp_name,
                "dataset_name": dataset_name,
                "algorithm": cfg.algo.name,
                "inverse_problem": cfg.algo.deg,
                "seed": int(cfg.exp.seed),
                "samples_root": samples_root,
                "metric_history_path": metric_history_path,
                "dataset_size": int(dataset_size),
                "planned_batches": int(planned_batches),
                "planned_images": int(planned_images),
                "completed_batches": 0,
                "completed_images": 0,
                "current_batch_index": None,
                "current_batch_size": 0,
                "elapsed_sec": 0.0,
                "eta_sec": None,
                "mean_psnr": None,
                "mean_ssim": None,
                "mean_lpips": None,
                "started_at": run_started_at,
            },
        )
    if dist.get_rank() == 0 and write_metric_history:
        write_json_file(
            metric_history_path,
            {
                "status": "running",
                "experiment_name": exp_name,
                "dataset_name": dataset_name,
                "algorithm": cfg.algo.name,
                "inverse_problem": cfg.algo.deg,
                "seed": int(cfg.exp.seed),
                "samples_root": samples_root,
                "started_at": run_started_at,
                "entries": [],
            },
        )

    try:
        for it, (x, y, info) in enumerate(loader):
            if cfg.exp.smoke_test > 0 and it >= cfg.exp.smoke_test:
                break

            n, c, h, w = x.size()
            x, y = x.cuda(), y.cuda()

            x = preprocess(x)
            ts = get_timesteps(cfg)

            kwargs = info
            if "ddrm" in cfg.algo.name or "mcg" in cfg.algo.name or "dps" in cfg.algo.name or "pgdm" in cfg.algo.name or "reddiff" in cfg.algo.name:
                idx = info['index']
                if 'inp' in cfg.algo.deg or 'in2' in cfg.algo.deg:   #what is in2?
                    H.set_indices(idx)
                y_0 = H.measure(x)
                kwargs["y_0"] = y_0

            #pgdm
            if cfg.exp.save_evolution:
                xt_s, _, xt_vis, _, mu_fft_abs_s, mu_fft_ang_s = algo.sample(x, y, ts, **kwargs)
            else:
                xt_s, _ = algo.sample(x, y, ts, **kwargs)

            #visualiztion of steps
            if cfg.exp.save_evolution:
                xt_vis = postprocess(xt_vis).cpu()
                print('torch.max(mu_fft_abs_s)', torch.max(mu_fft_abs_s))
                print('torch.min(mu_fft_abs_s)', torch.min(mu_fft_abs_s))
                print('torch.max(mu_fft_ang_s)', torch.max(mu_fft_ang_s))
                print('torch.min(mu_fft_ang_s)', torch.min(mu_fft_ang_s))
                mu_fft_abs = torch.log(mu_fft_abs_s+1)
                mu_fft_ang = mu_fft_ang_s  #torch.log10(mu_fft_abs_s+1)
                mu_fft_abs = (mu_fft_abs - torch.min(mu_fft_abs))/(torch.max(mu_fft_abs) - torch.min(mu_fft_abs))
                mu_fft_ang = (mu_fft_ang - torch.min(mu_fft_ang))/(torch.max(mu_fft_ang) - torch.min(mu_fft_ang))
                xx = torch.cat((xt_vis, mu_fft_abs, mu_fft_ang), dim=2)
                save_result(dataset_name, xx, y, info, samples_root, "evol")

            target = postprocess(x).detach().clamp(0.0, 1.0)
            if isinstance(xt_s, list):
                xo_metric = postprocess(xt_s[0]).detach().to(x.device).clamp(0.0, 1.0)
            else:
                xo_metric = postprocess(xt_s).detach().to(x.device).clamp(0.0, 1.0)
            xo = xo_metric.cpu()

            save_result(dataset_name, xo, y, info, samples_root, "")

            mse = torch.mean((xo_metric - target) ** 2, dim=(1, 2, 3))
            psnr = 10 * torch.log10(1 / (mse + 1e-10))
            psnrs.append(psnr.detach().cpu())
            if write_metric_history:
                ssim = structural_similarity_index_measure(
                    xo_metric, target, reduction=None, data_range=1.0
                ).detach().cpu()
                with torch.no_grad():
                    lpips_batch = lpips_metric(xo_metric, target).reshape(-1).detach().cpu()
                ssims.append(ssim)
                lpips_scores.append(lpips_batch)

            if cfg.exp.save_deg:
                xo = postprocess(get_degreadation_image(y_0, H, cfg))
                save_result(dataset_name, xo, y, info, samples_root, "deg")

            if cfg.exp.save_ori:
                save_result(dataset_name, target.cpu(), y, info, samples_root, "ori")

            completed_batches = it + 1
            completed_images += n

            if dist.get_rank() == 0 and write_progress:
                elapsed_sec = time.time() - start_time
                eta_sec = None
                if completed_batches > 0 and planned_batches > completed_batches:
                    eta_sec = ((planned_batches - completed_batches) / completed_batches) * elapsed_sec
                mean_psnr = torch.cat(psnrs, dim=0).mean().item() if psnrs else None
                mean_ssim = torch.cat(ssims, dim=0).mean().item() if ssims else None
                mean_lpips = torch.cat(lpips_scores, dim=0).mean().item() if lpips_scores else None
                current_indices = None
                if isinstance(info, dict) and "index" in info:
                    current_indices = [int(v) for v in info["index"].view(-1).tolist()]
                write_progress_json(
                    progress_path,
                    {
                        "status": "running",
                        "experiment_name": exp_name,
                        "dataset_name": dataset_name,
                        "algorithm": cfg.algo.name,
                        "inverse_problem": cfg.algo.deg,
                        "seed": int(cfg.exp.seed),
                        "samples_root": samples_root,
                        "metric_history_path": metric_history_path,
                        "dataset_size": int(dataset_size),
                        "planned_batches": int(planned_batches),
                        "planned_images": int(planned_images),
                        "completed_batches": int(completed_batches),
                        "completed_images": int(completed_images),
                        "current_batch_index": int(it),
                        "current_batch_size": int(n),
                        "current_indices": current_indices,
                        "elapsed_sec": float(elapsed_sec),
                        "eta_sec": None if eta_sec is None else float(eta_sec),
                        "mean_psnr": None if mean_psnr is None else float(mean_psnr),
                        "mean_ssim": None if mean_ssim is None else float(mean_ssim),
                        "mean_lpips": None if mean_lpips is None else float(mean_lpips),
                        "started_at": run_started_at,
                    },
                )
                if write_metric_history:
                    batch_psnr = psnrs[-1].mean().item()
                    batch_ssim = ssims[-1].mean().item()
                    batch_lpips = lpips_scores[-1].mean().item()
                    metric_history_entries.append(
                        {
                            "batch_index": int(it),
                            "batch_size": int(n),
                            "completed_batches": int(completed_batches),
                            "completed_images": int(completed_images),
                            "elapsed_sec": float(elapsed_sec),
                            "eta_sec": None if eta_sec is None else float(eta_sec),
                            "batch_psnr": float(batch_psnr),
                            "batch_ssim": float(batch_ssim),
                            "batch_lpips": float(batch_lpips),
                            "mean_psnr": None if mean_psnr is None else float(mean_psnr),
                            "mean_ssim": None if mean_ssim is None else float(mean_ssim),
                            "mean_lpips": None if mean_lpips is None else float(mean_lpips),
                            "current_indices": current_indices,
                        }
                    )
                    write_json_file(
                        metric_history_path,
                        {
                            "status": "running",
                            "experiment_name": exp_name,
                            "dataset_name": dataset_name,
                            "algorithm": cfg.algo.name,
                            "inverse_problem": cfg.algo.deg,
                            "seed": int(cfg.exp.seed),
                            "samples_root": samples_root,
                            "started_at": run_started_at,
                            "entries": metric_history_entries,
                        },
                    )

            if it % cfg.exp.logfreq == 0 or cfg.exp.smoke_test > 0 or it < 10:
                now = time.time() - start_time
                now_in_hours = strfdt(datetime.timedelta(seconds=now))
                future = (planned_batches - it - 1) / (it + 1) * now if planned_batches > 0 else 0
                future_in_hours = strfdt(datetime.timedelta(seconds=future))
                logger.info(f"Iter {it}: {now_in_hours} has passed, expect to finish in {future_in_hours}")

        mean_psnr = None
        mean_ssim = None
        mean_lpips = None
        if len(psnrs) > 0:
            psnrs = torch.cat(psnrs, dim=0)
            mean_psnr = psnrs.mean().item()
            logger.info(f'PSNR: {mean_psnr}')
        if len(ssims) > 0:
            ssims = torch.cat(ssims, dim=0)
            mean_ssim = ssims.mean().item()
            logger.info(f'SSIM: {mean_ssim}')
        if len(lpips_scores) > 0:
            lpips_scores = torch.cat(lpips_scores, dim=0)
            mean_lpips = lpips_scores.mean().item()
            logger.info(f'LPIPS: {mean_lpips}')

        logger.info("Done.")
        now = time.time() - start_time
        now_in_hours = strfdt(datetime.timedelta(seconds=now))
        logger.info(f"Total time: {now_in_hours}")

        if dist.get_rank() == 0 and write_progress:
            write_progress_json(
                progress_path,
                {
                    "status": "completed",
                    "experiment_name": exp_name,
                    "dataset_name": dataset_name,
                    "algorithm": cfg.algo.name,
                    "inverse_problem": cfg.algo.deg,
                    "seed": int(cfg.exp.seed),
                    "samples_root": samples_root,
                    "metric_history_path": metric_history_path,
                    "dataset_size": int(dataset_size),
                    "planned_batches": int(planned_batches),
                    "planned_images": int(planned_images),
                    "completed_batches": int(completed_batches),
                    "completed_images": int(completed_images),
                    "current_batch_index": int(completed_batches - 1) if completed_batches > 0 else None,
                    "current_batch_size": 0,
                    "elapsed_sec": float(now),
                    "eta_sec": 0.0,
                    "mean_psnr": None if mean_psnr is None else float(mean_psnr),
                    "mean_ssim": None if mean_ssim is None else float(mean_ssim),
                    "mean_lpips": None if mean_lpips is None else float(mean_lpips),
                    "started_at": run_started_at,
                    "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                },
            )
        if dist.get_rank() == 0 and write_metric_history:
            write_json_file(
                metric_history_path,
                {
                    "status": "completed",
                    "experiment_name": exp_name,
                    "dataset_name": dataset_name,
                    "algorithm": cfg.algo.name,
                    "inverse_problem": cfg.algo.deg,
                    "seed": int(cfg.exp.seed),
                    "samples_root": samples_root,
                    "started_at": run_started_at,
                    "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "summary": {
                        "mean_psnr": None if mean_psnr is None else float(mean_psnr),
                        "mean_ssim": None if mean_ssim is None else float(mean_ssim),
                        "mean_lpips": None if mean_lpips is None else float(mean_lpips),
                        "completed_batches": int(completed_batches),
                        "completed_images": int(completed_images),
                        "elapsed_sec": float(now),
                    },
                    "entries": metric_history_entries,
                },
            )
    except Exception as exc:
        if dist.get_rank() == 0 and write_progress:
            write_progress_json(
                progress_path,
                {
                    "status": "failed",
                    "experiment_name": exp_name,
                    "dataset_name": dataset_name,
                    "algorithm": cfg.algo.name,
                    "inverse_problem": cfg.algo.deg,
                    "seed": int(cfg.exp.seed),
                    "samples_root": samples_root,
                    "metric_history_path": metric_history_path,
                    "dataset_size": int(dataset_size),
                    "planned_batches": int(planned_batches),
                    "planned_images": int(planned_images),
                    "completed_batches": int(completed_batches),
                    "completed_images": int(completed_images),
                    "elapsed_sec": float(time.time() - start_time),
                    "mean_psnr": torch.cat(psnrs, dim=0).mean().item() if psnrs else None,
                    "mean_ssim": torch.cat(ssims, dim=0).mean().item() if ssims else None,
                    "mean_lpips": torch.cat(lpips_scores, dim=0).mean().item() if lpips_scores else None,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "started_at": run_started_at,
                    "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                },
            )
        if dist.get_rank() == 0 and write_metric_history:
            write_json_file(
                metric_history_path,
                {
                    "status": "failed",
                    "experiment_name": exp_name,
                    "dataset_name": dataset_name,
                    "algorithm": cfg.algo.name,
                    "inverse_problem": cfg.algo.deg,
                    "seed": int(cfg.exp.seed),
                    "samples_root": samples_root,
                    "started_at": run_started_at,
                    "finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "summary": {
                        "mean_psnr": torch.cat(psnrs, dim=0).mean().item() if psnrs else None,
                        "mean_ssim": torch.cat(ssims, dim=0).mean().item() if ssims else None,
                        "mean_lpips": torch.cat(lpips_scores, dim=0).mean().item() if lpips_scores else None,
                        "completed_batches": int(completed_batches),
                        "completed_images": int(completed_images),
                        "elapsed_sec": float(time.time() - start_time),
                    },
                    "entries": metric_history_entries,
                },
            )
        raise


@hydra.main(version_base="1.2", config_path="_configs", config_name="ddrmpp")
def main_dist(cfg: DictConfig):
    cwd = HydraConfig.get().runtime.output_dir
    print(cwd)

    if cfg.dist.num_processes_per_node < 0:
        size = torch.cuda.device_count()
        cfg.dist.num_processes_per_node = size
    else:
        size = cfg.dist.num_processes_per_node
    if size > 1:
        num_proc_node = cfg.dist.num_proc_node
        num_process_per_node = cfg.dist.num_processes_per_node
        world_size = num_proc_node * num_process_per_node
        mp.spawn(
            init_processes, args=(world_size, main, cfg, cwd), nprocs=world_size, join=True,
        )
    else:
        init_processes(0, size, main, cfg, cwd)


if __name__ == "__main__":
    main_dist()
