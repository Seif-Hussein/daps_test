import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "PDHG_CT_Matched_Counts_Colab.ipynb"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def markdown_cell(source: str, cell_id: str | None = None) -> dict:
    metadata = {}
    if cell_id is not None:
        metadata["id"] = cell_id
    return {
        "cell_type": "markdown",
        "metadata": metadata,
        "source": source.splitlines(keepends=True),
    }


cells = [
    markdown_cell(
        """# PDHG CT Matched-Counts Benchmark In Colab

This notebook runs the `mycode2` / `dyscode` **PDHG** CT pipeline on the original three `L067` medical CT source slices:

- `0009`
- `0080`
- `0528`

It is designed for the comparison protocol we settled on:

- same original source slices
- fixed CT-value normalization to `[-1, 1]`
- same native-count acquisition (`num_angles`, `I0`, seed)
- optional **shared count cache** compatible with the DM4CT benchmark notebook

The main use case is:

1. run the official DM4CT notebook in `custom + shared_counts` mode
2. run this PDHG notebook with the same acquisition settings
3. compare methods under the same sampled photon counts
""",
        cell_id="title",
    ),
    markdown_cell(
        """## Runtime

In Colab, go to `Runtime > Change runtime type` and choose:

- Hardware accelerator: `GPU`
- Python version: default Colab runtime

Then run the cells from top to bottom.
""",
        cell_id="runtime",
    ),
    code_cell(
        """#@title Project Settings

SETUP_MODE = "git"  #@param ["git", "drive_zip"]
REPO_URL = "https://github.com/Seif-Hussein/dyscode.git"  #@param {type:"string"}
REPO_BRANCH = "codex-pdhg-colab-light-100"  #@param {type:"string"}
DRIVE_ZIP_PATH = "/content/drive/MyDrive/mycode2.zip"  #@param {type:"string"}

REPO_DIR = "/content/mycode2"  #@param {type:"string"}
PYTHON_BIN = "/usr/bin/python3"  #@param {type:"string"}
DRIVE_EXPORT_DIR = "/content/drive/MyDrive/pdhg_ct_matched_benchmark_exports"  #@param {type:"string"}
DRIVE_CT_DATA_DIR = ""  #@param {type:"string"}
DRIVE_CHECKPOINT_DIR = ""  #@param {type:"string"}
DRIVE_MEASUREMENT_CACHE_DIR = ""  #@param {type:"string"}
HF_MODEL_ID = "jiayangshi/lodochallenge_pixel_diffusion"  #@param {type:"string"}

RUN_NAME = "PDHG_CT_Matched_Counts_Benchmark"  #@param {type:"string"}
SESSION_TAG = ""  #@param {type:"string"}
CONFIG_NAME = "default_ct.yaml"  #@param ["default_ct.yaml", "default_ct_admm.yaml"]

SEED = 99  #@param {type:"integer"}
TOTAL_IMAGES = 3  #@param {type:"integer"}
DATA_START_IDX = 0  #@param {type:"integer"}
LOG_TAIL_LINES = 120  #@param {type:"integer"}

# Use the same TIFF-preparation path as the DM4CT benchmark notebook so both
# notebooks can reuse the same count cache when the acquisition matches.
DATA_PREP_MODE = "global_minmax_to_tiff"  #@param ["global_minmax_to_tiff", "reuse_preprocessed_tiff"]
PREP_OUTPUT_DIR = "/content/dm4ct_preprocessed"  #@param {type:"string"}
VALID_EXTENSIONS = ".tif,.tiff,.png,.jpg,.jpeg,.dcm,.ima"  #@param {type:"string"}

# Leave CT_VALUE_MIN / CT_VALUE_MAX blank to:
# - use the reference L067 range [-1024, 3071] for the bundled 3-slice subset
# - or scan DRIVE_CT_DATA_DIR when you point to your own CT data.
CT_VALUE_MIN = ""  #@param {type:"string"}
CT_VALUE_MAX = ""  #@param {type:"string"}

# Measurement cache mode:
# - independent: sample new native-count measurements for this run
# - shared_counts: load or create a shared native-count cache compatible with
#   the DM4CT official benchmark notebook when the acquisition matches
MEASUREMENT_MATCH_MODE = "shared_counts"  #@param ["independent", "shared_counts"]

# This should match MEDICAL_CT_PRESET from the DM4CT notebook if you want the
# same cache signature. For the PDHG-paper-style native-count benchmark, use
# the default "custom" together with 80 angles and I0=10000.
DM4CT_CACHE_PRESET_TAG = "custom"  #@param {type:"string"}

NUM_STEPS = 400  #@param {type:"integer"}
MAX_ITER = 400  #@param {type:"integer"}
SIGMA_MAX = 10.0  #@param {type:"number"}
SIGMA_MIN = 0.075  #@param {type:"number"}
TAU = 0.01  #@param {type:"number"}
SIGMA_DUAL = 1200.0  #@param {type:"number"}
I0 = 10000.0  #@param {type:"number"}
NUM_ANGLES = 80  #@param {type:"integer"}

EVAL_METRICS = "psnr;ssim"  #@param {type:"string"}
SAVE_SAMPLES = True  #@param {type:"boolean"}
SAVE_TRAJ = False  #@param {type:"boolean"}
SAVE_TRAJ_RAW_DATA = False  #@param {type:"boolean"}
SHOW_CONFIG = False  #@param {type:"boolean"}

# Optional extra Hydra overrides, separated by semicolons.
EXTRA_HYDRA_OVERRIDES = ""  #@param {type:"string"}
"""
    ),
    code_cell(
        """#@title Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
"""
    ),
    code_cell(
        """#@title Fetch The Repo
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

repo_dir = Path(REPO_DIR)
repo_dir.parent.mkdir(parents=True, exist_ok=True)
os.chdir(repo_dir.parent)

if repo_dir.exists():
    shutil.rmtree(repo_dir)

if SETUP_MODE == "git":
    subprocess.run([
        "git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, REPO_DIR
    ], check=True)
elif SETUP_MODE == "drive_zip":
    zip_path = Path(DRIVE_ZIP_PATH)
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as handle:
        handle.extractall(repo_dir.parent)
else:
    raise ValueError(f"Unsupported SETUP_MODE: {SETUP_MODE}")

os.chdir(REPO_DIR)
subprocess.run(["git", "status", "--short"], check=True)
"""
    ),
    code_cell(
        """#@title Install Dependencies
import os
import subprocess

os.chdir(REPO_DIR)
subprocess.run([PYTHON_BIN, "-m", "pip", "install", "-r", "requirements-colab-ct.txt"], check=True)
"""
    ),
    code_cell(
        """#@title Optional: Copy Local Checkpoint From Drive
import os
import shutil
from pathlib import Path

os.chdir(REPO_DIR)
local_checkpoint_path = None
if DRIVE_CHECKPOINT_DIR.strip():
    src = Path(DRIVE_CHECKPOINT_DIR)
    if not src.exists():
        raise FileNotFoundError(f"Checkpoint folder not found: {src}")
    dst = Path(REPO_DIR) / "pretrained-models" / "dm4ct" / "lodochallenge_pixel_diffusion"
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)
    local_checkpoint_path = dst.as_posix()
    print(f"Copied checkpoint to: {dst}")
else:
    print("Using Hugging Face model download.")
"""
    ),
    code_cell(
        """#@title Prepare CT Data And Shared-Count Context
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

os.chdir(REPO_DIR)


def _parse_extensions(raw: str):
    items = [item.strip().lower() for item in raw.split(",") if item.strip()]
    return tuple(items)


def _load_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".dcm", ".ima"}:
        import pydicom

        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        return arr * slope + intercept

    if suffix in {".tif", ".tiff"}:
        import tifffile

        arr = tifffile.imread(path).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        return arr

    arr = np.asarray(Image.open(path), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def _collect_files(root: Path, extensions):
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            files.append(path)
    return files


if DRIVE_CT_DATA_DIR.strip():
    input_root = Path(DRIVE_CT_DATA_DIR)
    if not input_root.exists():
        raise FileNotFoundError(f"CT data root not found: {input_root}")
else:
    fallback_candidates = [
        Path(REPO_DIR) / "demo-samples" / "ct_l067_original_ima_subset",
        Path(REPO_DIR) / "demo_samples" / "ct_l067_original_ima_subset",
    ]
    input_root = next((path for path in fallback_candidates if path.exists()), None)
    if input_root is None:
        import urllib.request

        support_root = Path("/content/dm4ct_support_subset") / "ct_l067_original_ima_subset"
        support_root.mkdir(parents=True, exist_ok=True)
        base_url = (
            "https://raw.githubusercontent.com/Seif-Hussein/daps_test/"
            "codex-reddiff-colab-operators/demo-samples/ct_l067_original_ima_subset"
        )
        subset_files = [
            "L067_FD_1_SHARP_1.CT.0002.0009.2016.01.21.18.11.40.977560.404629207.IMA",
            "L067_FD_1_SHARP_1.CT.0002.0080.2016.01.21.18.11.40.977560.404630911.IMA",
            "L067_FD_1_SHARP_1.CT.0002.0528.2016.01.21.18.11.40.977560.404644046.IMA",
        ]
        for name in subset_files:
            dst = support_root / name
            if not dst.exists():
                urllib.request.urlretrieve(f"{base_url}/{name}", dst.as_posix())
        input_root = support_root

extensions = _parse_extensions(VALID_EXTENSIONS)
all_files = _collect_files(input_root, extensions)
if not all_files:
    raise FileNotFoundError(f"No CT files found under: {input_root}")

end_idx = min(len(all_files), int(DATA_START_IDX) + int(TOTAL_IMAGES))
selected_files = all_files[int(DATA_START_IDX):end_idx]
if not selected_files:
    raise ValueError("Selected CT slice range is empty.")

prep_signature_payload = {
    "input_root": input_root.as_posix(),
    "data_start_idx": int(DATA_START_IDX),
    "total_images": int(TOTAL_IMAGES),
    "prep_mode": DATA_PREP_MODE,
    "num_input_files": len(all_files),
}
prep_signature = hashlib.sha1(json.dumps(prep_signature_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
prepared_root = Path(PREP_OUTPUT_DIR) / prep_signature
prepared_root.mkdir(parents=True, exist_ok=True)

global_min = None
global_max = None
prepared_paths = []

if CT_VALUE_MIN.strip() and CT_VALUE_MAX.strip():
    global_min = float(CT_VALUE_MIN)
    global_max = float(CT_VALUE_MAX)
elif (not DRIVE_CT_DATA_DIR.strip()) and input_root.name == "ct_l067_original_ima_subset":
    global_min = -1024.0
    global_max = 3071.0
else:
    for path in all_files:
        arr = _load_array(path)
        cur_min = float(np.nanmin(arr))
        cur_max = float(np.nanmax(arr))
        global_min = cur_min if global_min is None else min(global_min, cur_min)
        global_max = cur_max if global_max is None else max(global_max, cur_max)

if global_max <= global_min:
    raise ValueError("Global CT value range is degenerate.")

if DATA_PREP_MODE == "global_minmax_to_tiff":
    import tifffile

    for idx, path in enumerate(selected_files):
        arr = _load_array(path)
        arr = np.clip(arr, global_min, global_max)
        arr = (arr - global_min) / (global_max - global_min)
        arr = arr * 2.0 - 1.0
        dst = prepared_root / f"{idx:05d}.tif"
        tifffile.imwrite(dst, arr.astype(np.float32))
        prepared_paths.append(dst)
elif DATA_PREP_MODE == "reuse_preprocessed_tiff":
    prepared_root = input_root
    prepared_paths = selected_files
else:
    raise ValueError(f"Unsupported DATA_PREP_MODE: {DATA_PREP_MODE}")

if local_checkpoint_path:
    model_ref = local_checkpoint_path
else:
    model_ref = HF_MODEL_ID

measurement_cache_root = None
if MEASUREMENT_MATCH_MODE == "shared_counts":
    cache_root = Path(DRIVE_MEASUREMENT_CACHE_DIR) if DRIVE_MEASUREMENT_CACHE_DIR.strip() else Path(DRIVE_EXPORT_DIR) / "dm4ct_shared_counts"
    signature_payload = {
        "prepared_root": prepared_root.as_posix(),
        "selected_files": [p.as_posix() for p in prepared_paths],
        "preset": DM4CT_CACHE_PRESET_TAG,
        "seed": int(SEED),
        "num_angles": int(NUM_ANGLES),
        "photon_count": float(I0),
    }
    signature = hashlib.sha1(json.dumps(signature_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    measurement_cache_root = cache_root / signature
    measurement_cache_root.mkdir(parents=True, exist_ok=True)

prep_context = {
    "input_root": input_root.as_posix(),
    "prepared_root": prepared_root.as_posix(),
    "selected_files": [p.as_posix() for p in selected_files],
    "prepared_paths": [p.as_posix() for p in prepared_paths],
    "global_min": float(global_min),
    "global_max": float(global_max),
    "model_ref": model_ref,
    "measurement_cache_root": measurement_cache_root.as_posix() if measurement_cache_root is not None else "",
}

print(json.dumps(prep_context, indent=2))
"""
    ),
    code_cell(
        """#@title Write PDHG Matched-Counts Runner
import os
from pathlib import Path

os.chdir(REPO_DIR)

runner_code = r'''
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from datasets import get_dataset
from measurements import get_operator
from model import get_model
from sampler import get_sampler
from utils import set_seed
from utils.eval import Evaluator, get_eval_fn, get_eval_fn_cmp
from utils.inverse_sampler import sample_in_batch
from utils.logging import log_results


def _atomic_json_dump(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _compose_args(config_name: str, overrides: list[str]):
    with initialize(version_base="1.3", config_path="configs"):
        return compose(config_name=config_name, overrides=overrides)


def _load_or_sample_counts(operator, images: torch.Tensor, prepared_paths: list[str], cache_root: str, measurement_mode: str):
    if measurement_mode != "shared_counts" or not cache_root:
        return operator.measure(images)

    cache_dir = Path(cache_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = []
    cache_paths = []
    missing = False

    for idx, image_path_str in enumerate(prepared_paths):
        image_path = Path(image_path_str)
        cache_path = cache_dir / f"{idx:05d}_{image_path.stem}.pt"
        cache_paths.append(cache_path)
        if cache_path.exists():
            tensor = torch.load(cache_path, map_location="cpu")
            cached.append(tensor)
        else:
            missing = True
            break

    if not missing and cached:
        return torch.cat(cached, dim=0).to(device=images.device, dtype=images.dtype)

    y = operator.measure(images)
    for idx, cache_path in enumerate(cache_paths):
        torch.save(y[idx:idx + 1].detach().cpu(), cache_path)
    return y


def _extract_metric_lists(results: dict):
    metrics = {}
    for key, stats in results.items():
        cmp_key = get_eval_fn_cmp(key)
        per_image = list(stats[cmp_key])
        metrics[f"image_{key}"] = per_image
        metrics[f"mean_{key}"] = float(np.mean(per_image)) if per_image else None
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    hydra_overrides = cfg["hydra_overrides"]
    run_args = _compose_args(cfg["config_name"], hydra_overrides)

    if run_args.show_config:
        print(OmegaConf.to_yaml(run_args))

    set_seed(run_args.seed)
    device = f"cuda:{run_args.gpu}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    save_root = Path(cfg["save_root"])
    save_root.mkdir(parents=True, exist_ok=True)
    progress_path = save_root / "progress.json"
    context_path = save_root / "run_aux" / "context.json"
    context_path.parent.mkdir(parents=True, exist_ok=True)
    context_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    dataset = get_dataset(**run_args.data)
    total_number = min(run_args.total_images, len(dataset))
    dataloader = DataLoader(dataset, batch_size=run_args.total_images, shuffle=False)
    images = next(iter(dataloader)).to(device)

    operator = get_operator(**run_args.inverse_task.operator)
    progress = {
        "status": "measuring",
        "planned_images": int(total_number),
        "completed_images": 0,
        "save_root": save_root.as_posix(),
        "prepared_root": cfg["prepared_root"],
        "measurement_cache_root": cfg.get("measurement_cache_root", ""),
        "started_at": time.time(),
    }
    _atomic_json_dump(progress_path, progress)

    y = _load_or_sample_counts(
        operator=operator,
        images=images,
        prepared_paths=cfg["prepared_paths"],
        cache_root=cfg.get("measurement_cache_root", ""),
        measurement_mode=cfg["measurement_match_mode"],
    )

    model = get_model(**run_args.model)
    sampler = get_sampler(**run_args.sampler, **run_args.inverse_task)

    eval_fn_list = [get_eval_fn(name) for name in run_args.eval_fn_list]
    evaluator = Evaluator(eval_fn_list)
    record_trajectory = bool(run_args.save_traj or run_args.save_traj_raw_data)

    full_samples = []
    full_trajs = []
    full_metric_histories = []

    progress["status"] = "running"
    _atomic_json_dump(progress_path, progress)

    try:
        for run_idx in range(run_args.num_runs):
            samples, trajs, metric_history = sample_in_batch(
                sampler,
                model,
                images,
                operator,
                y,
                evaluator,
                verbose=True,
                record=record_trajectory,
                batch_size=run_args.batch_size,
                gt=images,
                wandb=False,
            )
            full_samples.append(samples)
            if record_trajectory:
                full_trajs.append(trajs)
            if metric_history:
                full_metric_histories.append(metric_history)

        full_samples = torch.stack(full_samples, dim=0)
        results = evaluator.report(images, y, full_samples)
        markdown_text = evaluator.display(results)
        print(markdown_text)

        metric_history_artifact = None
        if full_metric_histories:
            metric_history_artifact = (
                full_metric_histories[0]
                if len(full_metric_histories) == 1
                else {"runs": full_metric_histories}
            )

        log_results(
            run_args,
            full_trajs,
            results,
            images,
            y,
            full_samples,
            markdown_text,
            total_number,
            metric_history=metric_history_artifact,
        )

        artifact_root = save_root / f"{run_args.name}_{run_args.data.name}_{run_args.inverse_task.operator.name}"
        metrics_path = artifact_root / "metrics.json"
        metric_history_path = artifact_root / "metric_history.json"

        compact_metrics = {
            "method": "pdhg",
            "num_images": int(total_number),
        }
        compact_metrics.update(_extract_metric_lists(results))
        _atomic_json_dump(save_root / "metrics.json", compact_metrics)
        if metric_history_path.exists():
            payload = json.loads(metric_history_path.read_text(encoding="utf-8"))
            _atomic_json_dump(save_root / "metric_history.json", payload)

        progress.update(
            {
                "status": "completed",
                "completed_images": int(total_number),
                "artifact_root": artifact_root.as_posix(),
                "metrics_path": (save_root / "metrics.json").as_posix(),
                "metric_history_path": (save_root / "metric_history.json").as_posix(),
            }
        )
        _atomic_json_dump(progress_path, progress)
    except Exception as exc:
        progress["status"] = "failed"
        progress["error"] = repr(exc)
        _atomic_json_dump(progress_path, progress)
        raise


if __name__ == "__main__":
    main()
'''

runner_path = Path(REPO_DIR) / "run_pdhg_ct_matched_benchmark.py"
runner_path.write_text(runner_code, encoding="utf-8")
print(f"Wrote {runner_path}")
"""
    ),
    code_cell(
        """#@title Build Benchmark Run
import json
import os
import shlex
from datetime import datetime
from pathlib import Path

os.chdir(REPO_DIR)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
session_bits = [RUN_NAME.strip(), SESSION_TAG.strip(), timestamp]
run_tag = "_".join(bit for bit in session_bits if bit)
run_tag = run_tag.replace(" ", "_")
save_root = Path(REPO_DIR) / "results" / "pdhg_ct_matched_benchmark" / run_tag
run_aux_root = save_root / "run_aux"
run_aux_root.mkdir(parents=True, exist_ok=True)
latest_log_path = run_aux_root / "run.log"
latest_pid_path = run_aux_root / "run.pid"

overrides = [
    f"seed={SEED}",
    f"name={RUN_NAME}",
    f"total_images={TOTAL_IMAGES}",
    f"batch_size=1",
    f"save_dir={save_root.as_posix()}",
    f"save_samples={str(SAVE_SAMPLES).lower()}",
    f"save_traj={str(SAVE_TRAJ).lower()}",
    f"save_traj_raw_data={str(SAVE_TRAJ_RAW_DATA).lower()}",
    f"show_config={str(SHOW_CONFIG).lower()}",
    f"eval_fn_list=[{','.join(EVAL_METRICS.split(';'))}]",
    f"sampler=edm_pdhg",
    f"sampler.annealing_scheduler_config.num_steps={NUM_STEPS}",
    f"sampler.annealing_scheduler_config.sigma_max={SIGMA_MAX}",
    f"sampler.annealing_scheduler_config.sigma_min={SIGMA_MIN}",
    f"inverse_task.admm_config.max_iter={MAX_ITER}",
    f"inverse_task.admm_config.pdhg.tau={TAU}",
    f"inverse_task.admm_config.pdhg.sigma_dual={SIGMA_DUAL}",
    f"inverse_task.operator.I0={I0}",
    f"inverse_task.operator.num_angles={NUM_ANGLES}",
    f"data.image_root_path={prep_context['prepared_root']}",
    "data.start_idx=0",
    f"data.end_idx={len(prep_context['prepared_paths'])}",
    "+data.value_min=-1.0",
    "+data.value_max=1.0",
    "model.model_config.local_files_only=" + ("true" if local_checkpoint_path else "false"),
]

if local_checkpoint_path:
    overrides.append(f"model.model_config.model_id={local_checkpoint_path}")
else:
    overrides.append(f"model.model_config.model_id={HF_MODEL_ID}")

for raw in EXTRA_HYDRA_OVERRIDES.split(";"):
    item = raw.strip().strip('"').strip("'")
    if item:
        overrides.append(item)

runner_config = {
    "config_name": CONFIG_NAME,
    "hydra_overrides": overrides,
    "save_root": save_root.as_posix(),
    "prepared_root": prep_context["prepared_root"],
    "prepared_paths": prep_context["prepared_paths"],
    "measurement_match_mode": MEASUREMENT_MATCH_MODE,
    "measurement_cache_root": prep_context["measurement_cache_root"],
    "global_min": prep_context["global_min"],
    "global_max": prep_context["global_max"],
}

config_path = run_aux_root / "runner_config.json"
config_path.write_text(json.dumps(runner_config, indent=2), encoding="utf-8")

run_cmd = [
    PYTHON_BIN,
    "run_pdhg_ct_matched_benchmark.py",
    "--config",
    config_path.as_posix(),
]

summary = {
    "run_tag": run_tag,
    "config_name": CONFIG_NAME,
    "prepared_root": prep_context["prepared_root"],
    "prepared_paths": prep_context["prepared_paths"],
    "measurement_match_mode": MEASUREMENT_MATCH_MODE,
    "measurement_cache_root": prep_context["measurement_cache_root"],
    "num_angles": NUM_ANGLES,
    "I0": I0,
    "ct_value_min": prep_context["global_min"],
    "ct_value_max": prep_context["global_max"],
    "run_cmd": run_cmd,
}
print(json.dumps(summary, indent=2))
"""
    ),
    code_cell(
        """#@title Launch Benchmark Run
import json
import os
import subprocess
from pathlib import Path

os.chdir(REPO_DIR)

latest_log_path.parent.mkdir(parents=True, exist_ok=True)
with open(latest_log_path, "w", encoding="utf-8") as log_file:
    process = subprocess.Popen(
        run_cmd,
        cwd=REPO_DIR,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )

latest_pid_path.write_text(str(process.pid), encoding="utf-8")
print(f"PID: {process.pid}")
print(f"Log path: {latest_log_path}")
print(f"Save root: {save_root}")
"""
    ),
    code_cell(
        """#@title Status
import json
import os
from pathlib import Path

pid = None
if latest_pid_path.exists():
    try:
        pid = int(latest_pid_path.read_text(encoding="utf-8").strip())
    except Exception:
        pid = None

running = False
if pid is not None:
    try:
        os.kill(pid, 0)
        running = True
    except OSError:
        running = False

progress_path = save_root / "progress.json"
metrics_path = save_root / "metrics.json"
history_path = save_root / "metric_history.json"

print(f"PID: {pid}")
print(f"Running: {running}")
print(f"Log path: {latest_log_path}")
print(f"Save root: {save_root}")
print(f"progress.json: {progress_path.exists()}")
print(f"metric_history.json: {history_path.exists()}")
print(f"metrics.json: {metrics_path.exists()}")

if progress_path.exists():
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    print("\\nprogress.json:\\n")
    print(json.dumps(progress, indent=2))
else:
    print("\\nprogress.json does not exist yet.")

if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    print("\\nmetrics.json:\\n")
    print(json.dumps(metrics, indent=2))
else:
    print("\\nmetrics.json does not exist yet.")

if history_path.exists():
    history = json.loads(history_path.read_text(encoding="utf-8"))
    print("\\nmetric_history.json keys:", list(history.keys())[:10])
else:
    print("\\nmetric_history.json does not exist yet.")

print("\\nRecent log lines:\\n")
if latest_log_path.exists():
    lines = latest_log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines[-LOG_TAIL_LINES:]:
        print(line)
else:
    print("Log file does not exist yet.")
"""
    ),
    code_cell(
        """#@title Export Artifacts To Drive
import os
import shutil
from pathlib import Path

export_root = Path(DRIVE_EXPORT_DIR)
export_root.mkdir(parents=True, exist_ok=True)
dst = export_root / save_root.name
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(save_root, dst)
print(f"Exported to: {dst}")
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH}")
