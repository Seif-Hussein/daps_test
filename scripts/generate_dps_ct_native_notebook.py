import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "DPS_CT_Native_Counts_Colab.ipynb"


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
        """# DPS CT Native-Counts Benchmark In Colab

This notebook runs **DPS ported into your native-count CT formulation** on the original three `L067` medical CT source slices:

- `0009`
- `0080`
- `0528`

It keeps the comparison target clear:

- same original CT-valued source slices
- fixed CT-value normalization to `[-1, 1]`
- same native transmission-count measurement model
- same nonnegative attenuation mapping before Poisson count sampling
- same `mycode2` / `dyscode` codebase and diffusion checkpoint family

In other words, this is a **same-formulation DPS baseline** rather than the official DM4CT log-sinogram pipeline.
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
DRIVE_EXPORT_DIR = "/content/drive/MyDrive/dps_ct_native_benchmark_exports"  #@param {type:"string"}
DRIVE_CT_DATA_DIR = ""  #@param {type:"string"}
DRIVE_CHECKPOINT_DIR = ""  #@param {type:"string"}
HF_MODEL_ID = "jiayangshi/lodochallenge_pixel_diffusion"  #@param {type:"string"}

RUN_NAME = "DPS_CT_Native_Counts_Benchmark"  #@param {type:"string"}
SESSION_TAG = ""  #@param {type:"string"}
CONFIG_NAME = "default_ct.yaml"  #@param ["default_ct.yaml", "default_ct_admm.yaml"]

SEED = 99  #@param {type:"integer"}
TOTAL_IMAGES = 3  #@param {type:"integer"}
DATA_START_IDX = 0  #@param {type:"integer"}
BATCH_SIZE = 1  #@param {type:"integer"}
LOG_TAIL_LINES = 120  #@param {type:"integer"}

# Convert the original source slices once into float TIFFs with a fixed CT range.
DATA_PREP_MODE = "global_minmax_to_tiff"  #@param ["global_minmax_to_tiff", "reuse_preprocessed_tiff"]
PREP_OUTPUT_DIR = "/content/dps_ct_native_preprocessed"  #@param {type:"string"}
VALID_EXTENSIONS = ".tif,.tiff,.png,.jpg,.jpeg,.dcm,.ima"  #@param {type:"string"}

# Leave these blank to use the reference L067 CT range [-1024, 3071] for the
# bundled 3-slice subset, or to scan DRIVE_CT_DATA_DIR when you point to your
# own CT data.
CT_VALUE_MIN = ""  #@param {type:"string"}
CT_VALUE_MAX = ""  #@param {type:"string"}

# Shared DPS diffusion schedule in this codebase.
# These are not PDHG settings; they are the sigma/noise schedule used by `edm_dps`.
NUM_STEPS = 100  #@param {type:"integer"}
SIGMA_MAX = 10.0  #@param {type:"number"}
SIGMA_MIN = 0.1  #@param {type:"number"}

# DPS-specific settings in the native-count formulation.
DPS_STEP_SIZE = 10.0  #@param {type:"number"}
DPS_ZETA_MODE = "constant"  #@param ["constant", "residual_norm"]

# CT acquisition settings for your formulation.
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
        """#@title Prepare CT Data
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

prep_context = {
    "input_root": input_root.as_posix(),
    "prepared_root": prepared_root.as_posix(),
    "selected_files": [p.as_posix() for p in selected_files],
    "prepared_paths": [p.as_posix() for p in prepared_paths],
    "global_min": float(global_min),
    "global_max": float(global_max),
    "model_ref": model_ref,
}

print(json.dumps(prep_context, indent=2))
"""
    ),
    code_cell(
        """#@title Apply DPS And Native CT Operator Patches
import os
from pathlib import Path

os.chdir(REPO_DIR)


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        print(f"Pattern not found in {path.name}, leaving it unchanged.")
        return
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"Patched {path}")


dps_path = Path(REPO_DIR) / "sampler" / "dps.py"
replace_once(
    dps_path,
    "        self.zeta_min = 0.6\\n        self.zeta_max = 0.6\\n",
    "",
)
replace_once(
    dps_path,
    "                # Forward prediction and squared residual (no 1/(2*sigma_y^2) scaling;\\n                # matches Algorithm 1 / Eq. (21) in DPS paper).\\n                pred = operator(x0_hat)\\n                per_sample_sq, per_sample_norm = self._residual_sq_and_norm(operator, pred, measurement)\\n\\n                loss = per_sample_sq.sum()  # scalar\\n",
    "                # Native count-domain CT needs a likelihood-consistent mismatch, since\\n                # operator(x0_hat) returns line integrals while the measurement stores counts.\\n                measurement_model = getattr(operator, 'measurement_model', None)\\n                if measurement_model == 'transmission_ct' or getattr(operator, 'name', None) in {'transmission_ct', 'transmission_ct_native'}:\\n                    per_sample_sq = operator.error(x0_hat, measurement)\\n                    per_sample_norm = (per_sample_sq + 1e-12).sqrt()\\n                else:\\n                    pred = operator(x0_hat)\\n                    per_sample_sq, per_sample_norm = self._residual_sq_and_norm(operator, pred, measurement)\\n\\n                loss = per_sample_sq.sum()  # scalar\\n",
)

native_operator_code = r'''
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .base import Operator
from .registry import register_operator


@register_operator(name="transmission_ct_native")
class TransmissionCTNative(Operator):
    measurement_model = "transmission_ct"

    def __init__(
        self,
        resolution=512,
        num_angles=60,
        num_detectors=None,
        angle_offset_deg=0.0,
        detector_scale=1.0,
        attenuation_min=0.0,
        attenuation_max=1.0,
        eta=1.0,
        I0=1.0e4,
        channels=1,
        clamp_input=True,
        sigma=1.0,
        measurement_cache_path="",
        device="cuda",
    ):
        super().__init__(sigma)
        self.resolution = int(resolution)
        self.num_angles = int(num_angles)
        self.num_detectors = int(num_detectors) if num_detectors is not None else int(resolution)
        self.angle_offset_deg = float(angle_offset_deg)
        self.detector_scale = float(detector_scale)
        self.attenuation_min = float(attenuation_min)
        self.attenuation_max = float(attenuation_max)
        self.eta = float(eta)
        self.I0 = float(I0)
        self.channels = int(channels)
        self.clamp_input = bool(clamp_input)
        self.device = device
        self.measurement_cache_path = str(measurement_cache_path) if measurement_cache_path is not None else ""

        angles = torch.arange(self.num_angles, dtype=torch.float32)
        angles = angles * (math.pi / max(1, self.num_angles))
        angles = angles + math.radians(self.angle_offset_deg)
        self.angles = angles

    def _cache_path(self):
        if not self.measurement_cache_path:
            return None
        return Path(self.measurement_cache_path)

    def _load_cached_counts(self, reference: torch.Tensor):
        cache_path = self._cache_path()
        if cache_path is None or not cache_path.exists():
            return None
        cached = torch.load(cache_path, map_location="cpu")
        if tuple(cached.shape) != tuple(reference.shape):
            raise RuntimeError(
                f"Cached measurement shape {tuple(cached.shape)} does not match expected {tuple(reference.shape)}"
            )
        return cached.to(device=reference.device, dtype=reference.dtype)

    def _save_cached_counts(self, counts: torch.Tensor):
        cache_path = self._cache_path()
        if cache_path is None:
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(counts.detach().cpu(), cache_path)

    def _angles_on(self, device, dtype):
        return self.angles.to(device=device, dtype=dtype)

    def _attenuation_image(self, x: torch.Tensor) -> torch.Tensor:
        x01 = (x + 1.0) * 0.5
        if self.clamp_input:
            x01 = x01.clamp(0.0, 1.0)
        return self.attenuation_min + x01 * (self.attenuation_max - self.attenuation_min)

    def _image_from_mu(self, mu: torch.Tensor) -> torch.Tensor:
        denom = max(self.attenuation_max - self.attenuation_min, 1.0e-12)
        x01 = (mu - self.attenuation_min) / denom
        x = x01 * 2.0 - 1.0
        return x.clamp(-1.0, 1.0)

    @staticmethod
    def _resize_detector_axis(sino: torch.Tensor, num_detectors: int) -> torch.Tensor:
        if sino.shape[-1] == num_detectors:
            return sino
        b, c, a, d = sino.shape
        flat = sino.reshape(b * c * a, 1, d)
        flat = F.interpolate(flat, size=num_detectors, mode="linear", align_corners=False)
        return flat.reshape(b, c, a, num_detectors)

    def _forward_mu(self, mu: torch.Tensor) -> torch.Tensor:
        if mu.dim() != 4:
            raise ValueError(f"Expected [B,C,H,W] tensor, got {tuple(mu.shape)}")

        b, c, h, w = mu.shape
        if c != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {c}")

        angles = self._angles_on(mu.device, mu.dtype)
        projections = []
        dx = 2.0 / max(h, 1)

        for angle in angles:
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)
            theta = torch.tensor(
                [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0]],
                device=mu.device,
                dtype=mu.dtype,
            ).unsqueeze(0).expand(b, -1, -1)
            grid = F.affine_grid(theta, mu.size(), align_corners=False)
            rotated = F.grid_sample(mu, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            proj = rotated.sum(dim=-2) * dx * self.detector_scale
            projections.append(proj)

        sino = torch.stack(projections, dim=-2)
        sino = self._resize_detector_axis(sino, self.num_detectors)
        return sino

    def _backproject_mu(self, sino: torch.Tensor, out_hw=None) -> torch.Tensor:
        if sino.dim() != 4:
            raise ValueError(f"Expected [B,C,A,W] tensor, got {tuple(sino.shape)}")

        b, c, _, d = sino.shape
        h = self.resolution if out_hw is None else int(out_hw[0])
        w = self.resolution if out_hw is None else int(out_hw[1])
        dx = 2.0 / max(h, 1)
        sino = self._resize_detector_axis(sino, w)
        angles = self._angles_on(sino.device, sino.dtype)

        back = torch.zeros((b, c, h, w), device=sino.device, dtype=sino.dtype)
        for idx, angle in enumerate(angles):
            proj = sino[:, :, idx, :]
            slab = proj.unsqueeze(-2).expand(b, c, h, w) * dx * self.detector_scale
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)
            theta = torch.tensor(
                [[cos_a, sin_a, 0.0], [-sin_a, cos_a, 0.0]],
                device=sino.device,
                dtype=sino.dtype,
            ).unsqueeze(0).expand(b, -1, -1)
            grid = F.affine_grid(theta, slab.size(), align_corners=False)
            back = back + F.grid_sample(slab, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        return back

    def adjoint(self, p, x_like=None):
        mu_grad = self._backproject_mu(p)
        scale = 0.5 * (self.attenuation_max - self.attenuation_min)
        return mu_grad * scale

    def __call__(self, x):
        mu = self._attenuation_image(x)
        return self._forward_mu(mu)

    def incident_counts(self, reference: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(self.I0):
            i0 = self.I0.to(device=reference.device, dtype=reference.dtype)
        else:
            i0 = torch.as_tensor(self.I0, device=reference.device, dtype=reference.dtype)
        while i0.dim() < reference.dim():
            i0 = i0.unsqueeze(0)
        return torch.broadcast_to(i0, reference.shape)

    def mean_measurement(self, x: torch.Tensor) -> torch.Tensor:
        z = self(x)
        return self.incident_counts(z) * torch.exp(-z).clamp_min(0.0)

    def measure(self, x):
        rate = self.mean_measurement(x).clamp_min(0.0).clamp_max(1.0e12)
        cached = self._load_cached_counts(rate)
        if cached is not None:
            return cached
        counts = torch.poisson(rate)
        self._save_cached_counts(counts)
        return counts

    def error(self, x, y):
        z = self(x)
        i0 = self.incident_counts(z)
        return self.eta * (i0 * torch.exp(-z) + y * z).flatten(1).sum(-1)

    def loss(self, x, y):
        return self.error(x, y).sum()

    def post_ml_op(self, x, y):
        return x

'''

native_operator_path = Path(REPO_DIR) / "measurements" / "transmission_ct_native.py"
native_operator_path.write_text(native_operator_code, encoding="utf-8")
print(f"Wrote {native_operator_path}")

measurements_init_path = Path(REPO_DIR) / "measurements" / "__init__.py"
measurements_init = measurements_init_path.read_text(encoding="utf-8")
if "from .transmission_ct_native import TransmissionCTNative" not in measurements_init:
    measurements_init = measurements_init.replace(
        "from .transmission_ct import TransmissionCT\\n",
        "from .transmission_ct import TransmissionCT\\nfrom .transmission_ct_native import TransmissionCTNative\\n",
    )
if "TransmissionCT, TransmissionCTNative, DownSamplingExplicit" not in measurements_init:
    measurements_init = measurements_init.replace(
        "TransmissionCT, DownSamplingExplicit",
        "TransmissionCT, TransmissionCTNative, DownSamplingExplicit",
    )
measurements_init_path.write_text(measurements_init, encoding="utf-8")
print(f"Patched {measurements_init_path}")
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
save_root = Path(REPO_DIR) / "results" / "dps_ct_native_benchmark" / run_tag
run_aux_root = save_root / "run_aux"
hydra_root = run_aux_root / "hydra"
run_aux_root.mkdir(parents=True, exist_ok=True)
latest_log_path = run_aux_root / "run.log"
latest_pid_path = run_aux_root / "run.pid"

overrides = [
    f"seed={SEED}",
    f"name={RUN_NAME}",
    f"total_images={TOTAL_IMAGES}",
    f"batch_size={BATCH_SIZE}",
    f"save_dir={save_root.as_posix()}",
    f"save_samples={str(SAVE_SAMPLES).lower()}",
    f"save_traj={str(SAVE_TRAJ).lower()}",
    f"save_traj_raw_data={str(SAVE_TRAJ_RAW_DATA).lower()}",
    f"show_config={str(SHOW_CONFIG).lower()}",
    f"eval_fn_list=[{','.join(EVAL_METRICS.split(';'))}]",
    "sampler=edm_dps",
    f"sampler.annealing_scheduler_config.num_steps={NUM_STEPS}",
    f"sampler.annealing_scheduler_config.sigma_max={SIGMA_MAX}",
    f"sampler.annealing_scheduler_config.sigma_min={SIGMA_MIN}",
    "inverse_task.operator.name=transmission_ct_native",
    f"inverse_task.operator.I0={I0}",
    f"inverse_task.operator.num_angles={NUM_ANGLES}",
    f"+inverse_task.admm_config.dps.zeta_base={DPS_STEP_SIZE}",
    f"+inverse_task.admm_config.dps.zeta_mode={DPS_ZETA_MODE}",
    "+inverse_task.admm_config.dps.zeta_min=0.0",
    "+inverse_task.admm_config.dps.zeta_max=1000000.0",
    f"data.image_root_path={prep_context['prepared_root']}",
    "data.start_idx=0",
    f"data.end_idx={len(prep_context['prepared_paths'])}",
    "+data.value_min=-1.0",
    "+data.value_max=1.0",
    "data.percentile_min=null",
    "data.percentile_max=null",
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

run_cmd = [
    PYTHON_BIN,
    "recover_inverse2.py",
    "--config-name",
    CONFIG_NAME,
    f"hydra.run.dir={hydra_root.as_posix()}",
] + overrides

summary = {
    "run_tag": run_tag,
    "config_name": CONFIG_NAME,
    "prepared_root": prep_context["prepared_root"],
    "prepared_paths": prep_context["prepared_paths"],
    "global_min": prep_context["global_min"],
    "global_max": prep_context["global_max"],
    "num_angles": NUM_ANGLES,
    "I0": I0,
    "num_steps": NUM_STEPS,
    "dps_step_size": DPS_STEP_SIZE,
    "dps_zeta_mode": DPS_ZETA_MODE,
    "run_cmd": run_cmd,
}

print("Effective native-count DPS settings:\\n")
print(json.dumps(summary, indent=2))
print("\\nCommand:\\n")
print(" ".join(shlex.quote(part) for part in run_cmd))

last_context = {
    "run_tag": run_tag,
    "save_root": save_root.as_posix(),
    "run_aux_root": run_aux_root.as_posix(),
    "latest_log_path": latest_log_path.as_posix(),
    "latest_pid_path": latest_pid_path.as_posix(),
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
    print("\\nmetric_history.json entries:", len(history.get("entries", [])))
    if history.get("entries"):
        print("Last metric entry:")
        print(json.dumps(history["entries"][-1], indent=2))
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
