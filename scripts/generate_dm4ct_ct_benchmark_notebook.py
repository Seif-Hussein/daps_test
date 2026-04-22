import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "DM4CT_CT_Official_Benchmark_Colab.ipynb"


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
        """# Official DM4CT CT Benchmark In Colab (REDDIFF / DPS)

This notebook runs the **official DM4CT CT implementations** for:

- `reddiff`
- `dps`

It is intentionally separate from our patched `dyscode` notebook so the benchmark path stays clean:

- official DM4CT operator / noiser / conditioning code
- official DM4CT REDDIFF and DPS pipelines
- fixed paper-style hyperparameters instead of retuning inside our framework

Default method settings are taken from the DM4CT medical-CT benchmark:

- `DPS`: step size `eta = 10`
- `REDDIFF`: learning rate `0.01`
- `REDDIFF`: measurement-consistency factor `0.5/1/1/1`
- `REDDIFF`: noise-fit factor `1e4`

The medical CT acquisition presets follow the DM4CT benchmark setup:

- `medical_40_clean`
- `medical_20_mild_noise`
- `medical_80_more_noise`
- `medical_80_noise_ring`

You can optionally enable a shared-count cache so repeated official DM4CT runs reuse the exact same sampled photon counts.
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
REPO_URL = "https://github.com/DM4CT/DM4CT.git"  #@param {type:"string"}
REPO_BRANCH = "main"  #@param {type:"string"}
DRIVE_ZIP_PATH = "/content/drive/MyDrive/DM4CT.zip"  #@param {type:"string"}

REPO_DIR = "/content/DM4CT"  #@param {type:"string"}
PYTHON_BIN = "/usr/bin/python3"  #@param {type:"string"}
DRIVE_EXPORT_DIR = "/content/drive/MyDrive/dm4ct_ct_benchmark_exports"  #@param {type:"string"}
DRIVE_CT_DATA_DIR = ""  #@param {type:"string"}
DRIVE_MODEL_PATH = ""  #@param {type:"string"}
DRIVE_MEASUREMENT_CACHE_DIR = ""  #@param {type:"string"}
HF_MODEL_ID = "jiayangshi/lodochallenge_pixel_diffusion"  #@param {type:"string"}

RUN_NAME = "DM4CT_CT_Official_Benchmark"  #@param {type:"string"}
SESSION_TAG = ""  #@param {type:"string"}
METHOD = "reddiff"  #@param ["reddiff", "dps"]
MEDICAL_CT_PRESET = "medical_80_more_noise"  #@param ["medical_40_clean", "medical_20_mild_noise", "medical_80_more_noise", "medical_80_noise_ring", "custom"]

SEED = 99  #@param {type:"integer"}
TOTAL_IMAGES = 1  #@param {type:"integer"}
DATA_START_IDX = 0  #@param {type:"integer"}
LOG_TAIL_LINES = 120  #@param {type:"integer"}
NUM_WORKERS = 0  #@param {type:"integer"}

# Dataset preparation:
# - global_minmax_to_tiff approximates the official DM4CT preprocessing by rescaling
#   the selected data to [-1, 1] TIFF slices using one global min/max over DRIVE_CT_DATA_DIR.
# - reuse_preprocessed_tiff assumes DRIVE_CT_DATA_DIR already points at preprocessed slices.
DATA_PREP_MODE = "global_minmax_to_tiff"  #@param ["global_minmax_to_tiff", "reuse_preprocessed_tiff"]
PREP_OUTPUT_DIR = "/content/dm4ct_preprocessed"  #@param {type:"string"}
VALID_EXTENSIONS = ".tif,.tiff,.png,.jpg,.jpeg,.dcm,.ima"  #@param {type:"string"}

# Optional shared-count mode for repeated official DM4CT runs. This does not change the
# reconstruction algorithm; it only reuses the same photon-count realization.
MEASUREMENT_MATCH_MODE = "independent"  #@param ["independent", "shared_counts"]

# Custom acquisition settings used only when MEDICAL_CT_PRESET = "custom".
CUSTOM_NUM_ANGLES = 80  #@param {type:"integer"}
CUSTOM_NOISER = "poisson"  #@param ["none", "poisson", "ring"]
CUSTOM_TRANSMITTANCE_RATE = 0.5  #@param {type:"number"}
CUSTOM_PHOTON_COUNT = 5000.0  #@param {type:"number"}
CUSTOM_BAD_PIXEL_RATIO = 0.05  #@param {type:"number"}
CUSTOM_RING_SCALE = 0.25  #@param {type:"number"}
CUSTOM_RING_SEED = 123  #@param {type:"integer"}

SAVE_SAMPLES = True  #@param {type:"boolean"}

# Advanced method overrides. Leave these unchanged if you want the official benchmark defaults.
ALLOW_METHOD_OVERRIDES = False  #@param {type:"boolean"}
DPS_NUM_STEPS_OVERRIDE = 1000  #@param {type:"integer"}
REDDIFF_NUM_STEPS_OVERRIDE = 200  #@param {type:"integer"}
REDDIFF_SIGMA_OVERRIDE = 0.0001  #@param {type:"number"}
REDDIFF_LR_OVERRIDE = 0.01  #@param {type:"number"}
REDDIFF_MEAS_WEIGHT_OVERRIDE = 1.0  #@param {type:"number"}
REDDIFF_NOISE_WEIGHT_OVERRIDE = 10000.0  #@param {type:"number"}
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
        """#@title Install DM4CT Dependencies
import os
import subprocess

os.chdir(REPO_DIR)

deps = [
    PYTHON_BIN,
    "-m",
    "pip",
    "install",
    "-q",
    "diffusers",
    "accelerate",
    "huggingface_hub",
    "tifffile",
    "scikit-image",
    "pydicom",
    "astra-toolbox",
]
subprocess.run(deps, check=True)
"""
    ),
    code_cell(
        """#@title Prepare CT Data And Model Paths
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


def _is_dicom(path: Path) -> bool:
    return path.suffix.lower() in {".dcm", ".ima"}


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


if not DRIVE_CT_DATA_DIR.strip():
    raise ValueError("DRIVE_CT_DATA_DIR must point to your CT dataset.")

input_root = Path(DRIVE_CT_DATA_DIR)
if not input_root.exists():
    raise FileNotFoundError(f"CT data root not found: {input_root}")

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

if DATA_PREP_MODE == "global_minmax_to_tiff":
    for path in all_files:
        arr = _load_array(path)
        cur_min = float(np.nanmin(arr))
        cur_max = float(np.nanmax(arr))
        global_min = cur_min if global_min is None else min(global_min, cur_min)
        global_max = cur_max if global_max is None else max(global_max, cur_max)

    if global_max <= global_min:
        raise ValueError("Global CT value range is degenerate.")

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

if DRIVE_MODEL_PATH.strip():
    effective_model_path = Path(DRIVE_MODEL_PATH)
    if not effective_model_path.exists():
        raise FileNotFoundError(f"Model path not found: {effective_model_path}")
    model_ref = effective_model_path.as_posix()
else:
    model_ref = HF_MODEL_ID

prep_context = {
    "input_root": input_root.as_posix(),
    "prepared_root": prepared_root.as_posix(),
    "selected_files": [p.as_posix() for p in selected_files],
    "prepared_paths": [p.as_posix() for p in prepared_paths],
    "global_min": global_min,
    "global_max": global_max,
    "model_ref": model_ref,
}

print(json.dumps(prep_context, indent=2))
"""
    ),
    code_cell(
        """#@title Write Official DM4CT Runner
import os
from pathlib import Path

os.chdir(REPO_DIR)

runner_code = r'''
import argparse
import json
import math
import random
import time
from pathlib import Path

import astra
import numpy as np
import torch
from diffusers import DDPMScheduler, DDIMScheduler, UNet2DModel
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tifffile import imwrite

from condition_methods import PosteriorSampling, RedDiff
from forward_operators_ct import NoNoise, Operator, PoissonNoise, PoissonNoiseRing
from pipelines import DDPMPipelineDPS, DDPMPipelineRedDiff


def _atomic_json_dump(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_uint8_image(arr: np.ndarray) -> np.ndarray:
    arr = np.clip((arr + 1.0) / 2.0, 0.0, 1.0)
    arr = np.round(arr * 255.0).astype(np.uint8)
    return arr


def _load_preprocessed_slice(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
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


class SharedCountPoissonNoise(PoissonNoise):
    def __init__(self, transmittance_rate: float, phonton_count: float, cache_root: str = ""):
        super().__init__(transmittance_rate, phonton_count)
        self.cache_root = Path(cache_root) if cache_root else None
        self.cache_key = None

    def _cache_path(self):
        if self.cache_root is None or self.cache_key is None:
            return None
        self.cache_root.mkdir(parents=True, exist_ok=True)
        return self.cache_root / f"{self.cache_key}.pt"

    def set_cache_key(self, key: str):
        self.cache_key = key

    def _load_cached_counts(self, template: torch.Tensor):
        cache_path = self._cache_path()
        if cache_path is None or not cache_path.exists():
            return None
        cached = torch.load(cache_path, map_location="cpu")
        if tuple(cached.shape) != tuple(template.shape):
            raise RuntimeError(
                f"Cached counts shape {tuple(cached.shape)} does not match expected {tuple(template.shape)}"
            )
        return cached.to(device=template.device, dtype=template.dtype)

    def _save_cached_counts(self, counts: torch.Tensor):
        cache_path = self._cache_path()
        if cache_path is None:
            return
        torch.save(counts.detach().cpu(), cache_path)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        self.scale_factor = self.cal_attenuation_factor(data, self.transmittance_rate)
        scaled = data * self.scale_factor
        transmission = torch.exp(-scaled)

        counts = self._load_cached_counts(transmission)
        if counts is None:
            counts = torch.poisson(transmission * self.phonton_count)
            self._save_cached_counts(counts)

        counts = counts.clone()
        counts[counts == 0] = 1
        data = torch.divide(counts, self.phonton_count)
        data = -torch.log(data)
        data /= self.scale_factor
        return data


class SharedCountPoissonNoiseRing(PoissonNoiseRing):
    def __init__(self, transmittance_rate: float, phonton_count: float, bad_pixel_ratio: float, scale=1, random_seed=123, cache_root: str = ""):
        super().__init__(transmittance_rate, phonton_count, bad_pixel_ratio, scale=scale, random_seed=random_seed)
        self.cache_root = Path(cache_root) if cache_root else None
        self.cache_key = None

    def _cache_path(self):
        if self.cache_root is None or self.cache_key is None:
            return None
        self.cache_root.mkdir(parents=True, exist_ok=True)
        return self.cache_root / f"{self.cache_key}.pt"

    def set_cache_key(self, key: str):
        self.cache_key = key

    def _load_cached_counts(self, template: torch.Tensor):
        cache_path = self._cache_path()
        if cache_path is None or not cache_path.exists():
            return None
        cached = torch.load(cache_path, map_location="cpu")
        if tuple(cached.shape) != tuple(template.shape):
            raise RuntimeError(
                f"Cached counts shape {tuple(cached.shape)} does not match expected {tuple(template.shape)}"
            )
        return cached.to(device=template.device, dtype=template.dtype)

    def _save_cached_counts(self, counts: torch.Tensor):
        cache_path = self._cache_path()
        if cache_path is None:
            return
        torch.save(counts.detach().cpu(), cache_path)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        if data.is_cuda:
            torch.cuda.manual_seed_all(self.random_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        self.scale_factor = self.cal_attenuation_factor(data, self.transmittance_rate)
        scaled = data * self.scale_factor
        transmission = torch.exp(-scaled)

        counts = self._load_cached_counts(transmission)
        if counts is None:
            counts = torch.poisson(transmission * self.phonton_count)
            self._save_cached_counts(counts)

        counts = counts.clone()
        counts[counts == 0] = 1
        data = torch.divide(counts, self.phonton_count)
        data = -torch.log(data)
        data /= self.scale_factor

        data_std = data.std()
        B, A, W = data.shape
        num_bad_pixels = int(B * W * self.bad_pixel_ratio)
        mask = torch.zeros(B * W, device=data.device)
        mask[:num_bad_pixels] = 1
        mask = mask[torch.randperm(B * W)].reshape(B, W)
        noise = torch.randn(B, W, device=data.device) * self.scale * data_std
        data = data.clone()
        data += noise[:, None, :] * mask[:, None, :]
        return data


def _make_noiser(cfg):
    mode = cfg["noiser"]
    match_mode = cfg["measurement_match_mode"]
    cache_root = cfg.get("measurement_cache_root", "")
    if mode == "none":
        return NoNoise()
    if mode == "poisson":
        if match_mode == "shared_counts":
            return SharedCountPoissonNoise(cfg["transmittance_rate"], cfg["photon_count"], cache_root=cache_root)
        return PoissonNoise(cfg["transmittance_rate"], cfg["photon_count"])
    if mode == "ring":
        if match_mode == "shared_counts":
            return SharedCountPoissonNoiseRing(
                cfg["transmittance_rate"],
                cfg["photon_count"],
                cfg["bad_pixel_ratio"],
                scale=cfg["ring_scale"],
                random_seed=cfg["ring_seed"],
                cache_root=cache_root,
            )
        return PoissonNoiseRing(
            cfg["transmittance_rate"],
            cfg["photon_count"],
            cfg["bad_pixel_ratio"],
            cfg["ring_scale"],
            cfg["ring_seed"],
        )
    raise ValueError(f"Unsupported noiser mode: {mode}")


def _set_cache_key(noiser, key: str):
    if hasattr(noiser, "set_cache_key"):
        noiser.set_cache_key(key)


def _compute_metrics(gt: np.ndarray, recon: np.ndarray):
    gt01 = np.clip((gt + 1.0) / 2.0, 0.0, 1.0)
    rec01 = np.clip((recon + 1.0) / 2.0, 0.0, 1.0)
    psnr = peak_signal_noise_ratio(gt01, rec01, data_range=1.0)
    ssim = structural_similarity(gt01, rec01, data_range=1.0)
    return float(psnr), float(ssim), gt01, rec01


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

    save_root = Path(cfg["save_root"])
    save_root.mkdir(parents=True, exist_ok=True)
    progress_path = save_root / "progress.json"
    history_path = save_root / "metric_history.json"
    samples_root = save_root / "samples"
    samples_root.mkdir(parents=True, exist_ok=True)

    _set_seed(int(cfg["seed"]))

    planned_images = len(cfg["image_paths"])
    progress = {
        "method": cfg["method"],
        "status": "initializing",
        "planned_images": planned_images,
        "completed_images": 0,
        "mean_psnr": None,
        "mean_ssim": None,
        "started_at": time.time(),
        "samples_root": samples_root.as_posix(),
    }
    history = {"images": []}
    _atomic_json_dump(progress_path, progress)
    _atomic_json_dump(history_path, history)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_ref = cfg["model_ref"]
    unet = UNet2DModel.from_pretrained(model_ref).to(device)

    if cfg["method"] == "dps":
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        pipeline = DDPMPipelineDPS(unet=unet, scheduler=scheduler)
    elif cfg["method"] == "reddiff":
        scheduler = DDIMScheduler(num_train_timesteps=1000)
        pipeline = DDPMPipelineRedDiff(unet=unet, scheduler=scheduler)
    else:
        raise ValueError(f"Unsupported method: {cfg['method']}")

    angles = np.linspace(0, np.pi, int(cfg["num_angles"]))
    vol_geom = astra.create_vol_geom(512, 512, 1)
    proj_geom = astra.create_proj_geom("parallel3d", 1, 1, 1, 512, angles)
    operator = Operator(volume_geometry=vol_geom, projection_geometry=proj_geom)
    noiser = _make_noiser(cfg)

    if cfg["method"] == "dps":
        conditioning_method = PosteriorSampling(operator=operator, noiser=noiser, scale=cfg["dps_step_size"])
    else:
        conditioning_method = RedDiff(operator=operator, noiser=noiser)
    pipeline.measurement_condition = conditioning_method

    psnrs = []
    ssims = []
    start_time = time.time()
    progress["status"] = "running"
    _atomic_json_dump(progress_path, progress)

    try:
        for image_offset, image_path_str in enumerate(cfg["image_paths"]):
            image_path = Path(image_path_str)
            cache_key = f"{image_offset:05d}_{image_path.stem}"
            _set_cache_key(noiser, cache_key)

            gt_arr = _load_preprocessed_slice(image_path)
            gt_tensor = torch.from_numpy(gt_arr).to(device=device, dtype=torch.float32).unsqueeze(0)

            measurement = operator(gt_tensor)
            measurement_noisy = noiser(measurement)

            if cfg["method"] == "dps":
                output = pipeline(
                    num_inference_steps=int(cfg["num_inference_steps"]),
                    measurement=measurement_noisy,
                    output_type=np.array,
                )
            else:
                output = pipeline(
                    num_inference_steps=int(cfg["num_inference_steps"]),
                    measurement=measurement_noisy,
                    sigma=float(cfg["reddiff_sigma"]),
                    loss_measurement_weight=float(cfg["reddiff_meas_weight"]),
                    loss_noise_weight=float(cfg["reddiff_noise_weight"]),
                    lr=float(cfg["reddiff_lr"]),
                    output_type=np.array,
                )

            recon_arr = np.asarray(output.images[0], dtype=np.float32).squeeze()
            psnr, ssim, gt01, rec01 = _compute_metrics(gt_arr, recon_arr)
            psnrs.append(psnr)
            ssims.append(ssim)

            if cfg["save_samples"]:
                gt_png = Image.fromarray(_to_uint8_image(gt_arr), mode="L")
                rec_png = Image.fromarray(_to_uint8_image(recon_arr), mode="L")
                gt_png.save(samples_root / f"{image_offset:05d}_gt.png")
                rec_png.save(samples_root / f"{image_offset:05d}_{cfg['method']}.png")
                imwrite(samples_root / f"{image_offset:05d}_{cfg['method']}.tif", recon_arr.astype(np.float32))

            elapsed = time.time() - start_time
            history["images"].append(
                {
                    "position": image_offset,
                    "source_path": image_path.as_posix(),
                    "psnr": psnr,
                    "ssim": ssim,
                    "elapsed_sec": elapsed,
                }
            )
            progress.update(
                {
                    "completed_images": image_offset + 1,
                    "elapsed_sec": elapsed,
                    "mean_psnr": float(np.mean(psnrs)),
                    "mean_ssim": float(np.mean(ssims)),
                    "status": "running",
                }
            )
            _atomic_json_dump(history_path, history)
            _atomic_json_dump(progress_path, progress)

        metrics = {
            "method": cfg["method"],
            "num_images": planned_images,
            "mean_psnr": float(np.mean(psnrs)) if psnrs else None,
            "mean_ssim": float(np.mean(ssims)) if ssims else None,
            "image_psnr": psnrs,
            "image_ssim": ssims,
        }
        _atomic_json_dump(save_root / "metrics.json", metrics)
        progress["status"] = "completed"
        _atomic_json_dump(progress_path, progress)
    except Exception as exc:
        progress["status"] = "failed"
        progress["error"] = repr(exc)
        _atomic_json_dump(progress_path, progress)
        raise


if __name__ == "__main__":
    main()
'''

runner_path = Path(REPO_DIR) / "run_dm4ct_ct_benchmark.py"
runner_path.write_text(runner_code, encoding="utf-8")
print(f"Wrote {runner_path}")
"""
    ),
    code_cell(
        """#@title Build Benchmark Run
import hashlib
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
save_root = Path(REPO_DIR) / "results" / "dm4ct_ct_benchmark" / run_tag
run_aux_root = save_root / "run_aux"
run_aux_root.mkdir(parents=True, exist_ok=True)
latest_log_path = run_aux_root / "run.log"
latest_pid_path = run_aux_root / "run.pid"

benchmark_presets = {
    "medical_40_clean": {
        "num_angles": 40,
        "noiser": "none",
        "transmittance_rate": 0.5,
        "photon_count": 10000.0,
        "bad_pixel_ratio": 0.0,
        "ring_scale": 0.25,
        "ring_seed": 123,
        "reddiff_meas_weight": 0.5,
    },
    "medical_20_mild_noise": {
        "num_angles": 20,
        "noiser": "poisson",
        "transmittance_rate": 0.5,
        "photon_count": 10000.0,
        "bad_pixel_ratio": 0.0,
        "ring_scale": 0.25,
        "ring_seed": 123,
        "reddiff_meas_weight": 1.0,
    },
    "medical_80_more_noise": {
        "num_angles": 80,
        "noiser": "poisson",
        "transmittance_rate": 0.5,
        "photon_count": 5000.0,
        "bad_pixel_ratio": 0.0,
        "ring_scale": 0.25,
        "ring_seed": 123,
        "reddiff_meas_weight": 1.0,
    },
    "medical_80_noise_ring": {
        "num_angles": 80,
        "noiser": "ring",
        "transmittance_rate": 0.5,
        "photon_count": 10000.0,
        "bad_pixel_ratio": 0.05,
        "ring_scale": 0.25,
        "ring_seed": 123,
        "reddiff_meas_weight": 1.0,
    },
}

if MEDICAL_CT_PRESET == "custom":
    effective = {
        "num_angles": int(CUSTOM_NUM_ANGLES),
        "noiser": CUSTOM_NOISER,
        "transmittance_rate": float(CUSTOM_TRANSMITTANCE_RATE),
        "photon_count": float(CUSTOM_PHOTON_COUNT),
        "bad_pixel_ratio": float(CUSTOM_BAD_PIXEL_RATIO),
        "ring_scale": float(CUSTOM_RING_SCALE),
        "ring_seed": int(CUSTOM_RING_SEED),
        "reddiff_meas_weight": float(REDDIFF_MEAS_WEIGHT_OVERRIDE),
    }
else:
    effective = dict(benchmark_presets[MEDICAL_CT_PRESET])

effective_num_steps = 1000 if METHOD == "dps" else 200
effective_dps_step = 10.0
effective_reddiff_sigma = 1.0e-4
effective_reddiff_lr = 0.01
effective_reddiff_meas_weight = effective["reddiff_meas_weight"]
effective_reddiff_noise_weight = 1.0e4

if ALLOW_METHOD_OVERRIDES:
    if METHOD == "dps":
        effective_num_steps = int(DPS_NUM_STEPS_OVERRIDE)
    else:
        effective_num_steps = int(REDDIFF_NUM_STEPS_OVERRIDE)
        effective_reddiff_sigma = float(REDDIFF_SIGMA_OVERRIDE)
        effective_reddiff_lr = float(REDDIFF_LR_OVERRIDE)
        effective_reddiff_meas_weight = float(REDDIFF_MEAS_WEIGHT_OVERRIDE)
        effective_reddiff_noise_weight = float(REDDIFF_NOISE_WEIGHT_OVERRIDE)

measurement_cache_root = None
if MEASUREMENT_MATCH_MODE == "shared_counts":
    cache_root = Path(DRIVE_MEASUREMENT_CACHE_DIR) if DRIVE_MEASUREMENT_CACHE_DIR.strip() else Path(DRIVE_EXPORT_DIR) / "dm4ct_shared_counts"
    signature_payload = {
        "prepared_root": prep_context["prepared_root"],
        "selected_files": prep_context["prepared_paths"],
        "preset": MEDICAL_CT_PRESET,
        "method": METHOD,
        "seed": int(SEED),
        "num_angles": int(effective["num_angles"]),
        "noiser": effective["noiser"],
        "transmittance_rate": float(effective["transmittance_rate"]),
        "photon_count": float(effective["photon_count"]),
        "bad_pixel_ratio": float(effective["bad_pixel_ratio"]),
        "ring_scale": float(effective["ring_scale"]),
        "ring_seed": int(effective["ring_seed"]),
    }
    signature = hashlib.sha1(json.dumps(signature_payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    measurement_cache_root = cache_root / signature
    measurement_cache_root.mkdir(parents=True, exist_ok=True)

runner_config = {
    "method": METHOD,
    "seed": int(SEED),
    "model_ref": prep_context["model_ref"],
    "image_paths": prep_context["prepared_paths"],
    "save_root": save_root.as_posix(),
    "num_angles": int(effective["num_angles"]),
    "noiser": effective["noiser"],
    "transmittance_rate": float(effective["transmittance_rate"]),
    "photon_count": float(effective["photon_count"]),
    "bad_pixel_ratio": float(effective["bad_pixel_ratio"]),
    "ring_scale": float(effective["ring_scale"]),
    "ring_seed": int(effective["ring_seed"]),
    "measurement_match_mode": MEASUREMENT_MATCH_MODE,
    "measurement_cache_root": measurement_cache_root.as_posix() if measurement_cache_root is not None else "",
    "num_inference_steps": int(effective_num_steps),
    "dps_step_size": float(effective_dps_step),
    "reddiff_sigma": float(effective_reddiff_sigma),
    "reddiff_lr": float(effective_reddiff_lr),
    "reddiff_meas_weight": float(effective_reddiff_meas_weight),
    "reddiff_noise_weight": float(effective_reddiff_noise_weight),
    "save_samples": bool(SAVE_SAMPLES),
}

config_path = run_aux_root / "runner_config.json"
config_path.write_text(json.dumps(runner_config, indent=2), encoding="utf-8")

run_cmd = [
    PYTHON_BIN,
    "run_dm4ct_ct_benchmark.py",
    "--config",
    config_path.as_posix(),
]

summary = {
    "method": METHOD,
    "medical_ct_preset": MEDICAL_CT_PRESET,
    "data_prep_mode": DATA_PREP_MODE,
    "prepared_root": prep_context["prepared_root"],
    "num_images": len(prep_context["prepared_paths"]),
    "measurement_match_mode": MEASUREMENT_MATCH_MODE,
    "measurement_cache_root": measurement_cache_root.as_posix() if measurement_cache_root is not None else None,
    "num_angles": effective["num_angles"],
    "noiser": effective["noiser"],
    "transmittance_rate": effective["transmittance_rate"],
    "photon_count": effective["photon_count"],
    "bad_pixel_ratio": effective["bad_pixel_ratio"],
    "ring_scale": effective["ring_scale"],
    "ring_seed": effective["ring_seed"],
    "num_inference_steps": effective_num_steps,
    "dps_step_size": effective_dps_step if METHOD == "dps" else None,
    "reddiff_sigma": effective_reddiff_sigma if METHOD == "reddiff" else None,
    "reddiff_lr": effective_reddiff_lr if METHOD == "reddiff" else None,
    "reddiff_meas_weight": effective_reddiff_meas_weight if METHOD == "reddiff" else None,
    "reddiff_noise_weight": effective_reddiff_noise_weight if METHOD == "reddiff" else None,
}

print("Effective DM4CT benchmark settings:\\n")
print(json.dumps(summary, indent=2))
print("\\nCommand:\\n")
print(" ".join(shlex.quote(part) for part in run_cmd))

last_context = {
    "run_tag": run_tag,
    "save_root": save_root.as_posix(),
    "run_aux_root": run_aux_root.as_posix(),
    "latest_log_path": latest_log_path.as_posix(),
    "latest_pid_path": latest_pid_path.as_posix(),
    "config_path": config_path.as_posix(),
    "run_cmd": run_cmd,
    "summary": summary,
}
context_path = run_aux_root / "context.json"
context_path.write_text(json.dumps(last_context, indent=2), encoding="utf-8")
print(f"Context saved to: {context_path}")
"""
    ),
    code_cell(
        """#@title Launch The Run In Background
import os
import subprocess
from pathlib import Path

os.chdir(REPO_DIR)
latest_log_path = Path(last_context["latest_log_path"])
latest_pid_path = Path(last_context["latest_pid_path"])
latest_log_path.parent.mkdir(parents=True, exist_ok=True)

with latest_log_path.open("w", encoding="utf-8") as log_handle:
    process = subprocess.Popen(
        last_context["run_cmd"],
        cwd=REPO_DIR,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )

latest_pid_path.write_text(str(process.pid), encoding="utf-8")
print(f"PID: {process.pid}")
print(f"Log: {latest_log_path}")
print(f"Save root: {last_context['save_root']}")
"""
    ),
    code_cell(
        """#@title Show Run Status
import json
from pathlib import Path

save_root = Path(last_context["save_root"])
log_path = Path(last_context["latest_log_path"])
pid_path = Path(last_context["latest_pid_path"])
progress_path = save_root / "progress.json"
history_path = save_root / "metric_history.json"
metrics_path = save_root / "metrics.json"

pid = int(pid_path.read_text(encoding="utf-8").strip()) if pid_path.exists() else None
running = False
if pid is not None:
    running = Path(f"/proc/{pid}").exists()

print(f"PID: {pid}")
print(f"Running: {running}")
print(f"Log path: {log_path}")
print(f"Progress JSON: {progress_path}")
print(f"Metric history JSON: {history_path}")
print(f"Metrics JSON: {metrics_path}")
print(f"Save root: {save_root}")

if progress_path.exists():
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    print("\\nProgress summary:\\n")
    for key in ["status", "completed_images", "planned_images", "elapsed_sec", "mean_psnr", "mean_ssim"]:
        print(f"{key}: {progress.get(key)}")
else:
    print("\\nProgress JSON does not exist yet.")

if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    print("\\nmetrics.json:\\n")
    print(json.dumps(metrics, indent=2))
else:
    print("\\nmetrics.json does not exist yet.")

if history_path.exists():
    history = json.loads(history_path.read_text(encoding="utf-8"))
    print("\\nmetric_history.json entries:", len(history.get("images", [])))
    if history.get("images"):
        print("Last metric entry:")
        print(json.dumps(history["images"][-1], indent=2))
else:
    print("\\nmetric_history.json does not exist yet.")

saved_pngs = sorted((save_root / "samples").glob("*.png")) if (save_root / "samples").exists() else []
print(f"\\nSaved PNG count: {len(saved_pngs)}")

if log_path.exists():
    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    tail = lines[-int(LOG_TAIL_LINES):]
    print("\\nRecent log lines:\\n")
    print("\\n".join(tail) if tail else "<log is empty>")
else:
    print("\\nLog file not found yet.")
"""
    ),
    code_cell(
        """#@title Show Results And Artifacts
import json
from pathlib import Path

save_root = Path(last_context["save_root"])
samples_root = save_root / "samples"
progress_path = save_root / "progress.json"
history_path = save_root / "metric_history.json"
metrics_path = save_root / "metrics.json"

print("Artifacts:\\n")
print(f"save_root: {save_root}")
print(f"samples_root: {samples_root}")
print(f"progress.json: {progress_path.exists()}")
print(f"metric_history.json: {history_path.exists()}")
print(f"metrics.json: {metrics_path.exists()}")
print(f"png_count: {len(list(samples_root.glob('*.png'))) if samples_root.exists() else 0}")

if metrics_path.exists():
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    print("\\nmetrics.json:\\n")
    print(json.dumps(metrics, indent=2))
"""
    ),
    code_cell(
        """#@title Copy Run Artifacts To Drive
import shutil
from pathlib import Path

export_root = Path(DRIVE_EXPORT_DIR)
export_root.mkdir(parents=True, exist_ok=True)

targets = [
    Path(last_context["save_root"]),
    Path(last_context["run_aux_root"]),
]

for src in targets:
    if not src.exists():
        print(f"Skipping missing path: {src}")
        continue

    dst = export_root / src.name
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    print(f"Copied {src} -> {dst}")
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

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH}")
