import json
from pathlib import Path
import hashlib


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "CT_Diffusion_Single_Run_Colab.ipynb"


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
        """# CT Single Run In Colab (PDHG / RED-diff / DPS)

This notebook mirrors `mycode2/notebooks/pdhg_ct_single_run_colab.ipynb`, but adds sampler selection for:

- `edm_pdhg`
- `edm_reddiff`
- `edm_dps`

It keeps Colab as a thin launcher around the CT-capable `dyscode` pipeline and runs `recover_inverse2.py` with Hydra overrides.

By default it now keeps a fair native-count CT comparison across methods:

- `PDHG`, `RED-diff`, and `DPS` all use the same count-domain transmission CT model
- RED-diff gets count-aware pseudo-inverse initialization
- DPS gets a count-domain likelihood-gradient step instead of comparing line integrals to counts
- a shared-count cache can force native-count and DM4CT-style runs to reuse the exact same photon-count realization

An opt-in DM4CT-style mode is still available for exploratory runs:

- global min/max CT normalization
- log-sinogram Poisson noiser with optional ring artifacts
- pseudo-inverse initialization for CT

Medical CT defaults taken from DM4CT Table 11:

- DPS: step size `eta = 10`
- RED-diff: learning rate `0.01`
- RED-diff: factor on measurement consistency error `0.5/1/1/1`
- RED-diff: factor on noise fit error `1e4`

This launcher defaults to the same 80-angle noisy CT setup used by the original single-run CT notebook, so `REDDIFF_MEAS_WEIGHT` starts at `1.0`. If you want the clean 40-angle medical setting from the benchmark, set `REDDIFF_MEAS_WEIGHT = 0.5`.

The notebook also applies a post-clone patch so:

- DPS uses its configured CT step size instead of a hardcoded clamp
- RED-diff exposes separate CT measurement-consistency and noise-fit weights
- a native-count CT operator with pseudo-inverse support is added for fair comparisons
- a DM4CT-style CT operator is also available as an explicit alternate mode
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
DRIVE_EXPORT_DIR = "/content/drive/MyDrive/ct_diffusion_single_run_exports"  #@param {type:"string"}
DRIVE_CT_DATA_DIR = ""  #@param {type:"string"}
DRIVE_CHECKPOINT_DIR = ""  #@param {type:"string"}
DRIVE_MEASUREMENT_CACHE_DIR = ""  #@param {type:"string"}
HF_MODEL_ID = "jiayangshi/lodochallenge_pixel_diffusion"  #@param {type:"string"}

RUN_NAME = "CT_Diffusion_Single_Run"  #@param {type:"string"}
SESSION_TAG = ""  #@param {type:"string"}
CONFIG_NAME = "default_ct.yaml"  #@param ["default_ct.yaml", "default_ct_admm.yaml"]
SAMPLER_CONFIG = "edm_pdhg"  #@param ["edm_pdhg", "edm_reddiff", "edm_dps"]
USE_CT_BENCHMARK_PRESET = True  #@param {type:"boolean"}

SEED = 99  #@param {type:"integer"}
TOTAL_IMAGES = 1  #@param {type:"integer"}
BATCH_SIZE = 1  #@param {type:"integer"}
DATA_START_IDX = 1  #@param {type:"integer"}
DATA_END_IDX = 2  #@param {type:"integer"}

NUM_STEPS = 400  #@param {type:"integer"}
MAX_ITER = 400  #@param {type:"integer"}
SIGMA_MAX = 10.0  #@param {type:"number"}
SIGMA_MIN = 0.075  #@param {type:"number"}
TAU = 0.01  #@param {type:"number"}
SIGMA_DUAL = 1200.0  #@param {type:"number"}
I0 = 10000.0  #@param {type:"number"}
NUM_ANGLES = 80  #@param {type:"integer"}
ADMM_LGVD_NUM_STEPS = 10  #@param {type:"integer"}

# CT consistency mode:
# - native_counts keeps all methods on the same count-domain transmission CT model
# - dm4ct_log ports the DM4CT-style log-sinogram measurement path
# - auto is kept for experimentation, but native_counts is the fair default
CT_OPERATOR_MODE = "native_counts"  #@param ["native_counts", "dm4ct_log", "auto"]
CT_PREPROCESS_MODE = "current_percentile"  #@param ["current_percentile", "dm4ct_global", "explicit_minmax", "auto"]
CT_VALUE_MIN = ""  #@param {type:"string"}
CT_VALUE_MAX = ""  #@param {type:"string"}
MEASUREMENT_MATCH_MODE = "shared_counts"  #@param ["shared_counts", "independent"]
DM4CT_TRANSMITTANCE_RATE = 0.5  #@param {type:"number"}
DM4CT_BAD_PIXEL_RATIO = 0.0  #@param {type:"number"}
DM4CT_RING_SCALE = 0.25  #@param {type:"number"}
DM4CT_RING_SEED = 123  #@param {type:"integer"}
DM4CT_PINV_METHOD = "sirt"  #@param ["sirt", "fbp", "backproject"]
DM4CT_PINV_ITERS = 60  #@param {type:"integer"}

# DM4CT Medical CT Table 11 gives DPS step size eta = 10.
DPS_STEP_SIZE = 10.0  #@param {type:"number"}
DPS_ZETA_MODE = "constant"  #@param ["constant", "residual_norm"]

# DM4CT Medical CT Table 11 gives RED-diff:
# - learning rate = 0.01
# - measurement-consistency factor = 0.5/1/1/1 across configs i-iv
# - noise-fit factor = 1e4
# The default CT launcher settings here are closest to the noisy 80-angle setups,
# so REDDIFF_MEAS_WEIGHT defaults to 1.0.
REDDIFF_LR = 0.01  #@param {type:"number"}
REDDIFF_MEAS_WEIGHT = 1.0  #@param {type:"number"}
REDDIFF_NOISE_WEIGHT = 10000.0  #@param {type:"number"}
REDDIFF_TIME_SAMPLING = "descending"  #@param ["descending", "random"]
REDDIFF_TIME_SPACING = "linear"  #@param ["linear", "log", "exp"]
REDDIFF_WEIGHT_TYPE = "constant"  #@param ["constant", "sigma", "sigma2", "inv_snr", "sqrt_inv_snr"]
REDDIFF_DATA_TERM = "nll"  #@param ["nll", "mse"]
REDDIFF_INIT_MODE = "pinv"  #@param ["noise", "random", "pinv", "measurement", "y"]

EVAL_METRICS = "psnr;ssim"  #@param {type:"string"}
SAVE_SAMPLES = True  #@param {type:"boolean"}
SAVE_TRAJ = False  #@param {type:"boolean"}
SAVE_TRAJ_RAW_DATA = False  #@param {type:"boolean"}
SHOW_CONFIG = False  #@param {type:"boolean"}
LOG_TAIL_LINES = 120  #@param {type:"integer"}

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
        """#@title Apply CT Sampler And Operator Patches
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

reddiff_path = Path(REPO_DIR) / "sampler" / "reddiff.py"
replace_once(
    reddiff_path,
    "        self.lr = float(_maybe_get(self.cfg, \\"lr\\", 0.5))\\n        self.lam = float(_maybe_get(self.cfg, \\"lambda\\", 0.25))\\n",
    "        self.lr = float(_maybe_get(self.cfg, \\"lr\\", 0.5))\\n        self.obs_weight = float(_maybe_get(self.cfg, \\"obs_weight\\", _maybe_get(self.cfg, \\"measurement_consistency_weight\\", 1.0)))\\n        self.lam = float(_maybe_get(self.cfg, \\"noise_fit_weight\\", _maybe_get(self.cfg, \\"lambda\\", 0.25)))\\n",
)
replace_once(
    reddiff_path,
    "            loss = data_loss + lam_t * reg_loss\\n",
    "            loss = self.obs_weight * data_loss + lam_t * reg_loss\\n",
)
replace_once(
    reddiff_path,
    "                    \\"RED-diff/data_loss\\": float(data_loss.item()),\\n",
    "                    \\"RED-diff/data_loss\\": float(data_loss.item()),\\n                    \\"RED-diff/obs_weight\\": float(self.obs_weight),\\n",
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
        pinv_method="sirt",
        pinv_iterations=60,
        pinv_relax=1.0,
        pinv_min=0.0,
        pinv_max=1.0,
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
        self.pinv_method = str(pinv_method).lower()
        self.pinv_iterations = int(pinv_iterations)
        self.pinv_relax = float(pinv_relax)
        self.pinv_min = float(pinv_min)
        self.pinv_max = float(pinv_max)
        self.measurement_cache_path = str(measurement_cache_path) if measurement_cache_path is not None else ""

        angles = torch.arange(self.num_angles, dtype=torch.float32)
        angles = angles * (math.pi / max(1, self.num_angles))
        angles = angles + math.radians(self.angle_offset_deg)
        self.angles = angles

        self._sirt_cache = {}

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

    def _ramp_filter(self, sino: torch.Tensor) -> torch.Tensor:
        n = sino.shape[-1]
        freq = torch.fft.rfftfreq(n, d=1.0).to(device=sino.device, dtype=sino.dtype)
        filt = freq.reshape((1, 1, 1, -1))
        spec = torch.fft.rfft(sino, dim=-1)
        return torch.fft.irfft(spec * filt, n=n, dim=-1)

    def _ensure_sirt_weights(self, device, dtype):
        key = (str(device), str(dtype))
        if key in self._sirt_cache:
            return self._sirt_cache[key]

        ones_mu = torch.ones((1, self.channels, self.resolution, self.resolution), device=device, dtype=dtype)
        R = self._forward_mu(ones_mu)
        R = torch.where(R > 1.0e-8, 1.0 / R, torch.zeros_like(R))

        ones_sino = torch.ones((1, self.channels, self.num_angles, self.num_detectors), device=device, dtype=dtype)
        C = self._backproject_mu(ones_sino)
        C = torch.where(C > 1.0e-8, 1.0 / C, torch.zeros_like(C))

        self._sirt_cache[key] = (R, C)
        return R, C

    def _counts_to_line_integrals(self, y_counts: torch.Tensor) -> torch.Tensor:
        i0 = self.incident_counts(y_counts)
        frac = (y_counts.clamp_min(1.0) / i0.clamp_min(1.0)).clamp(min=1.0e-12, max=1.0)
        return -torch.log(frac)

    def _pinv_backproject(self, measurement: torch.Tensor) -> torch.Tensor:
        z = self._counts_to_line_integrals(measurement)
        return self._backproject_mu(z) / max(self.num_angles, 1)

    def _pinv_fbp(self, measurement: torch.Tensor) -> torch.Tensor:
        z = self._counts_to_line_integrals(measurement)
        filtered = self._ramp_filter(z)
        recon = self._backproject_mu(filtered)
        recon = recon * (math.pi / max(2 * self.num_angles, 1))
        return recon

    def _pinv_sirt(self, measurement: torch.Tensor) -> torch.Tensor:
        z = self._counts_to_line_integrals(measurement)
        mu = torch.zeros(
            (measurement.shape[0], self.channels, self.resolution, self.resolution),
            device=measurement.device,
            dtype=measurement.dtype,
        )
        R, C = self._ensure_sirt_weights(measurement.device, measurement.dtype)
        for _ in range(max(self.pinv_iterations, 1)):
            residual = z - self._forward_mu(mu)
            mu = mu + self.pinv_relax * (C * self._backproject_mu(R * residual))
            mu = mu.clamp(min=self.pinv_min, max=self.pinv_max)
        return mu

    def pinv(self, measurement: torch.Tensor) -> torch.Tensor:
        method = self.pinv_method
        if method == "backproject":
            mu = self._pinv_backproject(measurement)
        elif method == "fbp":
            mu = self._pinv_fbp(measurement)
        else:
            mu = self._pinv_sirt(measurement)
        return self._image_from_mu(mu)
'''

native_operator_path = Path(REPO_DIR) / "measurements" / "transmission_ct_native.py"
native_operator_path.write_text(native_operator_code, encoding="utf-8")
print(f"Wrote {native_operator_path}")

dm4ct_operator_code = r'''
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .base import Operator
from .registry import register_operator


@register_operator(name="transmission_ct_dm4ct")
class TransmissionCTDM4CT(Operator):
    measurement_model = "linear_mse"

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
        transmittance_rate=0.5,
        bad_pixel_ratio=0.0,
        ring_scale=0.0,
        ring_seed=123,
        pinv_method="sirt",
        pinv_iterations=60,
        pinv_relax=1.0,
        pinv_min=0.0,
        pinv_max=1.0,
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
        self.transmittance_rate = float(transmittance_rate)
        self.bad_pixel_ratio = float(bad_pixel_ratio)
        self.ring_scale = float(ring_scale)
        self.ring_seed = int(ring_seed)
        self.pinv_method = str(pinv_method).lower()
        self.pinv_iterations = int(pinv_iterations)
        self.pinv_relax = float(pinv_relax)
        self.pinv_min = float(pinv_min)
        self.pinv_max = float(pinv_max)
        self.measurement_cache_path = str(measurement_cache_path) if measurement_cache_path is not None else ""

        angles = torch.arange(self.num_angles, dtype=torch.float32)
        angles = angles * (math.pi / max(1, self.num_angles))
        angles = angles + math.radians(self.angle_offset_deg)
        self.angles = angles

        self._sirt_cache = {}

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

    def _dm4ct_scale_factor(self, sinogram: torch.Tensor) -> torch.Tensor:
        target = max(self.transmittance_rate, 1.0e-6)
        mean_sino = sinogram.reshape(sinogram.shape[0], -1).mean(dim=1).clamp_min(1.0e-8)
        scale = -math.log(target) / mean_sino
        return scale.reshape((-1,) + (1,) * (sinogram.dim() - 1))

    def _native_count_rate(self, sinogram: torch.Tensor) -> torch.Tensor:
        return self.incident_counts(sinogram) * torch.exp(-sinogram).clamp_min(0.0)

    def _apply_ring_artifacts(self, data: torch.Tensor) -> torch.Tensor:
        if self.bad_pixel_ratio <= 0.0 or self.ring_scale <= 0.0:
            return data

        b = data.shape[0]
        w = data.shape[-1]
        cpu_gen = torch.Generator(device="cpu")
        cpu_gen.manual_seed(self.ring_seed)

        num_bad = int(b * w * self.bad_pixel_ratio)
        mask = torch.zeros(b * w, dtype=data.dtype)
        if num_bad > 0:
            mask[:num_bad] = 1.0
            perm = torch.randperm(b * w, generator=cpu_gen)
            mask = mask[perm]
        mask = mask.reshape(b, w).to(device=data.device, dtype=data.dtype)

        noise = torch.randn((b, w), generator=cpu_gen, dtype=data.dtype).to(device=data.device, dtype=data.dtype)
        data_std = data.reshape(b, -1).std(dim=1, unbiased=False).clamp_min(1.0e-8)
        noise = noise * (self.ring_scale * data_std[:, None])

        return data + noise[:, None, :] * mask[:, None, :]

    def measure(self, x):
        sinogram = self(x)
        cache_path = self._cache_path()
        cached_counts = self._load_cached_counts(sinogram)
        if cached_counts is not None or cache_path is not None:
            counts = cached_counts
            if counts is None:
                counts = torch.poisson(self._native_count_rate(sinogram).clamp(min=0.0, max=1.0e12))
                self._save_cached_counts(counts)
            counts = counts.clamp_min(1.0)
            y = -torch.log((counts / self.incident_counts(sinogram)).clamp_min(1.0e-12))
            y = self._apply_ring_artifacts(y)
            return y

        scale = self._dm4ct_scale_factor(sinogram)
        scaled = sinogram * scale
        transmission = torch.exp((-scaled).clamp(min=-80.0, max=80.0))
        counts = torch.poisson((transmission * self.incident_counts(sinogram)).clamp(min=0.0, max=1.0e12))
        counts = counts.clamp_min(1.0)
        y = -torch.log((counts / self.incident_counts(sinogram)).clamp_min(1.0e-12))
        y = y / scale
        y = self._apply_ring_artifacts(y)
        return y

    def error(self, x, y):
        return ((self(x) - y) ** 2).flatten(1).sum(-1)

    def loss(self, x, y):
        return 0.5 * self.error(x, y).sum()

    def post_ml_op(self, x, y):
        return x

    def _ramp_filter(self, sino: torch.Tensor) -> torch.Tensor:
        n = sino.shape[-1]
        freq = torch.fft.rfftfreq(n, d=1.0).to(device=sino.device, dtype=sino.dtype)
        filt = freq.reshape((1, 1, 1, -1))
        spec = torch.fft.rfft(sino, dim=-1)
        return torch.fft.irfft(spec * filt, n=n, dim=-1)

    def _ensure_sirt_weights(self, device, dtype):
        key = (str(device), str(dtype))
        if key in self._sirt_cache:
            return self._sirt_cache[key]

        ones_mu = torch.ones((1, self.channels, self.resolution, self.resolution), device=device, dtype=dtype)
        R = self._forward_mu(ones_mu)
        R = torch.where(R > 1.0e-8, 1.0 / R, torch.zeros_like(R))

        ones_sino = torch.ones((1, self.channels, self.num_angles, self.num_detectors), device=device, dtype=dtype)
        C = self._backproject_mu(ones_sino)
        C = torch.where(C > 1.0e-8, 1.0 / C, torch.zeros_like(C))

        self._sirt_cache[key] = (R, C)
        return R, C

    def _pinv_backproject(self, measurement: torch.Tensor) -> torch.Tensor:
        return self._backproject_mu(measurement) / max(self.num_angles, 1)

    def _pinv_fbp(self, measurement: torch.Tensor) -> torch.Tensor:
        filtered = self._ramp_filter(measurement)
        recon = self._backproject_mu(filtered)
        recon = recon * (math.pi / max(2 * self.num_angles, 1))
        return recon

    def _pinv_sirt(self, measurement: torch.Tensor) -> torch.Tensor:
        mu = torch.zeros(
            (measurement.shape[0], self.channels, self.resolution, self.resolution),
            device=measurement.device,
            dtype=measurement.dtype,
        )
        R, C = self._ensure_sirt_weights(measurement.device, measurement.dtype)
        for _ in range(max(self.pinv_iterations, 1)):
            residual = measurement - self._forward_mu(mu)
            mu = mu + self.pinv_relax * (C * self._backproject_mu(R * residual))
            mu = mu.clamp(min=self.pinv_min, max=self.pinv_max)
        return mu

    def pinv(self, measurement: torch.Tensor) -> torch.Tensor:
        method = self.pinv_method
        if method == "backproject":
            mu = self._pinv_backproject(measurement)
        elif method == "fbp":
            mu = self._pinv_fbp(measurement)
        else:
            mu = self._pinv_sirt(measurement)
        return self._image_from_mu(mu)
'''

dm4ct_operator_path = Path(REPO_DIR) / "measurements" / "transmission_ct_dm4ct.py"
dm4ct_operator_path.write_text(dm4ct_operator_code, encoding="utf-8")
print(f"Wrote {dm4ct_operator_path}")

measurements_init_path = Path(REPO_DIR) / "measurements" / "__init__.py"
measurements_init = measurements_init_path.read_text(encoding="utf-8")
if "from .transmission_ct_native import TransmissionCTNative" not in measurements_init:
    measurements_init = measurements_init.replace(
        "from .transmission_ct import TransmissionCT\\n",
        "from .transmission_ct import TransmissionCT\\nfrom .transmission_ct_native import TransmissionCTNative\\n",
    )
if "from .transmission_ct_dm4ct import TransmissionCTDM4CT" not in measurements_init:
    measurements_init = measurements_init.replace(
        "from .transmission_ct_native import TransmissionCTNative\\n",
        "from .transmission_ct_native import TransmissionCTNative\\nfrom .transmission_ct_dm4ct import TransmissionCTDM4CT\\n",
    )
if "TransmissionCT, TransmissionCTNative, TransmissionCTDM4CT" not in measurements_init:
    measurements_init = measurements_init.replace(
        "TransmissionCT, DownSamplingExplicit",
        "TransmissionCT, TransmissionCTNative, TransmissionCTDM4CT, DownSamplingExplicit",
    )
measurements_init_path.write_text(measurements_init, encoding="utf-8")
print(f"Patched {measurements_init_path}")
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
        """#@title Build Run Command
import json
import hashlib
import os
import shlex
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

os.chdir(REPO_DIR)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
session_bits = [RUN_NAME.strip(), SESSION_TAG.strip(), timestamp]
run_tag = "_".join(bit for bit in session_bits if bit)
run_tag = run_tag.replace(" ", "_")
save_root = Path(REPO_DIR) / "results" / "colab_ct_diffusion_single" / run_tag
hydra_root = save_root / "hydra"
run_aux_root = save_root / "run_aux"
latest_log_path = run_aux_root / "run.log"
latest_pid_path = run_aux_root / "run.pid"

def _resolve_effective_operator_mode():
    mode = CT_OPERATOR_MODE.strip().lower()
    if mode == "auto":
        return "native_counts" if SAMPLER_CONFIG == "edm_pdhg" else "dm4ct_log"
    if mode not in {"native_counts", "dm4ct_log"}:
        raise ValueError(f"Unsupported CT_OPERATOR_MODE: {CT_OPERATOR_MODE}")
    return mode


def _resolve_effective_preprocess_mode(operator_mode: str):
    mode = CT_PREPROCESS_MODE.strip().lower()
    if mode == "auto":
        return "dm4ct_global" if operator_mode == "dm4ct_log" else "current_percentile"
    if mode not in {"current_percentile", "dm4ct_global", "explicit_minmax"}:
        raise ValueError(f"Unsupported CT_PREPROCESS_MODE: {CT_PREPROCESS_MODE}")
    return mode


def _load_ct_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".dcm", ".ima"}:
        import pydicom

        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        return arr * slope + intercept

    arr = np.asarray(Image.open(path), dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def _compute_ct_value_range(root: Path):
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".dcm", ".ima"}
    files = sorted([p for p in root.rglob("*") if p.suffix.lower() in exts])
    if not files:
        raise FileNotFoundError(f"No CT image files found under: {root}")

    global_min = None
    global_max = None
    for path in files:
        arr = _load_ct_array(path)
        cur_min = float(np.nanmin(arr))
        cur_max = float(np.nanmax(arr))
        global_min = cur_min if global_min is None else min(global_min, cur_min)
        global_max = cur_max if global_max is None else max(global_max, cur_max)
    return float(global_min), float(global_max), len(files)


def _measurement_signature():
    payload = {
        "data_root": effective_data_root.as_posix(),
        "data_start_idx": int(DATA_START_IDX),
        "data_end_idx": int(DATA_END_IDX),
        "total_images": int(TOTAL_IMAGES),
        "seed": int(SEED),
        "resolution": 512,
        "num_angles": int(NUM_ANGLES),
        "I0": float(I0),
        "attenuation_min": 0.0,
        "attenuation_max": 1.0,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:16]


effective_operator_mode = _resolve_effective_operator_mode()
effective_preprocess_mode = _resolve_effective_preprocess_mode(effective_operator_mode)
effective_measurement_match_mode = MEASUREMENT_MATCH_MODE.strip().lower()
if effective_measurement_match_mode not in {"shared_counts", "independent"}:
    raise ValueError(f"Unsupported MEASUREMENT_MATCH_MODE: {MEASUREMENT_MATCH_MODE}")

effective_dps_step_size = float(DPS_STEP_SIZE)
effective_dps_zeta_mode = DPS_ZETA_MODE
effective_reddiff_lr = float(REDDIFF_LR)
effective_reddiff_meas_weight = float(REDDIFF_MEAS_WEIGHT)
effective_reddiff_noise_weight = float(REDDIFF_NOISE_WEIGHT)
effective_reddiff_data_term = REDDIFF_DATA_TERM
effective_reddiff_init_mode = REDDIFF_INIT_MODE
effective_reddiff_weight_type = REDDIFF_WEIGHT_TYPE

if USE_CT_BENCHMARK_PRESET:
    if SAMPLER_CONFIG == "edm_dps":
        effective_dps_step_size = 10.0
        effective_dps_zeta_mode = "constant"
    elif SAMPLER_CONFIG == "edm_reddiff":
        effective_reddiff_lr = 0.01
        effective_reddiff_noise_weight = 1.0e4
        if effective_operator_mode == "dm4ct_log":
            effective_reddiff_data_term = "mse"
            effective_reddiff_init_mode = "pinv"
            effective_reddiff_weight_type = "constant"

if effective_operator_mode == "dm4ct_log" and SAMPLER_CONFIG == "edm_reddiff" and effective_reddiff_data_term == "nll":
    print("Switching RED-diff data term from nll to mse for dm4ct_log mode.")
    effective_reddiff_data_term = "mse"

effective_data_root = Path(DRIVE_CT_DATA_DIR) if DRIVE_CT_DATA_DIR.strip() else Path(REPO_DIR) / "demo-samples" / "ct_l067_subset"
if DRIVE_CT_DATA_DIR.strip() and not effective_data_root.exists():
    raise FileNotFoundError(f"CT data folder not found: {effective_data_root}")

effective_value_min = None
effective_value_max = None
effective_value_scan_count = None
if effective_preprocess_mode == "dm4ct_global":
    effective_value_min, effective_value_max, effective_value_scan_count = _compute_ct_value_range(effective_data_root)
elif effective_preprocess_mode == "explicit_minmax":
    if not CT_VALUE_MIN.strip() or not CT_VALUE_MAX.strip():
        raise ValueError("CT_VALUE_MIN and CT_VALUE_MAX must be provided for explicit_minmax mode.")
    effective_value_min = float(CT_VALUE_MIN)
    effective_value_max = float(CT_VALUE_MAX)

measurement_cache_path = None
measurement_signature = None
if effective_measurement_match_mode == "shared_counts":
    measurement_signature = _measurement_signature()
    measurement_cache_root = Path(DRIVE_MEASUREMENT_CACHE_DIR) if DRIVE_MEASUREMENT_CACHE_DIR.strip() else Path(DRIVE_EXPORT_DIR) / "shared_measurements"
    measurement_cache_root.mkdir(parents=True, exist_ok=True)
    measurement_cache_path = measurement_cache_root / f"ct_counts_{measurement_signature}.pt"

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
    f"sampler={SAMPLER_CONFIG}",
    f"sampler.annealing_scheduler_config.num_steps={NUM_STEPS}",
    f"sampler.annealing_scheduler_config.sigma_max={SIGMA_MAX}",
    f"sampler.annealing_scheduler_config.sigma_min={SIGMA_MIN}",
    f"inverse_task.admm_config.max_iter={MAX_ITER}",
    f"model.model_config.local_files_only={'true' if local_checkpoint_path else 'false'}",
]

if effective_operator_mode == "dm4ct_log":
    overrides.extend([
        "inverse_task.operator.name=transmission_ct_dm4ct",
        f"inverse_task.operator.I0={I0}",
        f"inverse_task.operator.num_angles={NUM_ANGLES}",
        f"+inverse_task.operator.transmittance_rate={DM4CT_TRANSMITTANCE_RATE}",
        f"+inverse_task.operator.bad_pixel_ratio={DM4CT_BAD_PIXEL_RATIO}",
        f"+inverse_task.operator.ring_scale={DM4CT_RING_SCALE}",
        f"+inverse_task.operator.ring_seed={DM4CT_RING_SEED}",
        f"+inverse_task.operator.pinv_method={DM4CT_PINV_METHOD}",
        f"+inverse_task.operator.pinv_iterations={DM4CT_PINV_ITERS}",
    ])
    if measurement_cache_path is not None:
        overrides.append(f"+inverse_task.operator.measurement_cache_path={measurement_cache_path.as_posix()}")
else:
    overrides.extend([
        "inverse_task.operator.name=transmission_ct_native",
        f"inverse_task.operator.I0={I0}",
        f"inverse_task.operator.num_angles={NUM_ANGLES}",
        f"+inverse_task.operator.pinv_method={DM4CT_PINV_METHOD}",
        f"+inverse_task.operator.pinv_iterations={DM4CT_PINV_ITERS}",
    ])
    if measurement_cache_path is not None:
        overrides.append(f"+inverse_task.operator.measurement_cache_path={measurement_cache_path.as_posix()}")

if local_checkpoint_path:
    overrides.append(f"model.model_config.model_id={local_checkpoint_path}")
else:
    overrides.append(f"model.model_config.model_id={HF_MODEL_ID}")

if SAMPLER_CONFIG == "edm_pdhg":
    overrides.extend([
        f"inverse_task.admm_config.pdhg.tau={TAU}",
        f"inverse_task.admm_config.pdhg.sigma_dual={SIGMA_DUAL}",
    ])
elif SAMPLER_CONFIG == "edm_dps":
    overrides.extend([
        f"+inverse_task.admm_config.dps.zeta_base={effective_dps_step_size}",
        f"+inverse_task.admm_config.dps.zeta_mode={effective_dps_zeta_mode}",
        "+inverse_task.admm_config.dps.zeta_min=0.0",
        "+inverse_task.admm_config.dps.zeta_max=1000000.0",
    ])
elif SAMPLER_CONFIG == "edm_reddiff":
    overrides.extend([
        f"+inverse_task.admm_config.red_diff.lr={effective_reddiff_lr}",
        f"+inverse_task.admm_config.red_diff.obs_weight={effective_reddiff_meas_weight}",
        f"+inverse_task.admm_config.red_diff.noise_fit_weight={effective_reddiff_noise_weight}",
        f"+inverse_task.admm_config.red_diff.data_term={effective_reddiff_data_term}",
        f"+inverse_task.admm_config.red_diff.init={effective_reddiff_init_mode}",
        f"+inverse_task.admm_config.red_diff.weight_type={effective_reddiff_weight_type}",
        f"+inverse_task.admm_config.red_diff.time_sampling={REDDIFF_TIME_SAMPLING}",
        f"+inverse_task.admm_config.red_diff.time_spacing={REDDIFF_TIME_SPACING}",
    ])
else:
    raise ValueError(f"Unsupported sampler config: {SAMPLER_CONFIG}")

if CONFIG_NAME == "default_ct_admm.yaml":
    overrides.append(f"inverse_task.admm_config.denoise.lgvd.num_steps={ADMM_LGVD_NUM_STEPS}")

if DRIVE_CT_DATA_DIR.strip():
    overrides.extend([
        f"data.image_root_path={effective_data_root.as_posix()}",
        f"data.start_idx={DATA_START_IDX}",
        f"data.end_idx={DATA_END_IDX}",
    ])
else:
    overrides.extend([
        f"data.start_idx={DATA_START_IDX}",
        f"data.end_idx={DATA_END_IDX}",
    ])

if effective_preprocess_mode in {"dm4ct_global", "explicit_minmax"}:
    overrides.extend([
        f"+data.value_min={effective_value_min}",
        f"+data.value_max={effective_value_max}",
        "data.percentile_min=null",
        "data.percentile_max=null",
    ])

if EXTRA_HYDRA_OVERRIDES.strip():
    overrides.extend([item.strip() for item in EXTRA_HYDRA_OVERRIDES.split(';') if item.strip()])

run_cmd = [
    PYTHON_BIN,
    "recover_inverse2.py",
    "--config-name",
    CONFIG_NAME,
    f"hydra.run.dir={hydra_root.as_posix()}",
] + overrides

preset_summary = {
    "sampler_config": SAMPLER_CONFIG,
    "use_ct_benchmark_preset": bool(USE_CT_BENCHMARK_PRESET),
    "ct_operator_mode": effective_operator_mode,
    "ct_preprocess_mode": effective_preprocess_mode,
    "measurement_match_mode": effective_measurement_match_mode,
    "measurement_signature": measurement_signature,
    "measurement_cache_path": measurement_cache_path.as_posix() if measurement_cache_path is not None else None,
    "ct_data_root": effective_data_root.as_posix(),
    "ct_value_min": effective_value_min,
    "ct_value_max": effective_value_max,
    "ct_value_scan_count": effective_value_scan_count,
    "dm4ct_transmittance_rate": DM4CT_TRANSMITTANCE_RATE if effective_operator_mode == "dm4ct_log" else None,
    "dm4ct_bad_pixel_ratio": DM4CT_BAD_PIXEL_RATIO if effective_operator_mode == "dm4ct_log" else None,
    "dm4ct_ring_scale": DM4CT_RING_SCALE if effective_operator_mode == "dm4ct_log" else None,
    "dm4ct_pinv_method": DM4CT_PINV_METHOD if effective_operator_mode == "dm4ct_log" else None,
    "dm4ct_pinv_iters": DM4CT_PINV_ITERS if effective_operator_mode == "dm4ct_log" else None,
    "dps_step_size": effective_dps_step_size if SAMPLER_CONFIG == "edm_dps" else None,
    "dps_zeta_mode": effective_dps_zeta_mode if SAMPLER_CONFIG == "edm_dps" else None,
    "reddiff_lr": effective_reddiff_lr if SAMPLER_CONFIG == "edm_reddiff" else None,
    "reddiff_meas_weight": effective_reddiff_meas_weight if SAMPLER_CONFIG == "edm_reddiff" else None,
    "reddiff_noise_weight": effective_reddiff_noise_weight if SAMPLER_CONFIG == "edm_reddiff" else None,
    "reddiff_data_term": effective_reddiff_data_term if SAMPLER_CONFIG == "edm_reddiff" else None,
    "reddiff_init_mode": effective_reddiff_init_mode if SAMPLER_CONFIG == "edm_reddiff" else None,
    "reddiff_weight_type": effective_reddiff_weight_type if SAMPLER_CONFIG == "edm_reddiff" else None,
}
print("Effective sampler settings:\\n")
print(json.dumps(preset_summary, indent=2))

print("\\nCommand:\\n")
print(" ".join(shlex.quote(part) for part in run_cmd))

last_context = {
    "run_tag": run_tag,
    "save_root": save_root.as_posix(),
    "hydra_root": hydra_root.as_posix(),
    "latest_log_path": latest_log_path.as_posix(),
    "latest_pid_path": latest_pid_path.as_posix(),
    "run_cmd": run_cmd,
    "preset_summary": preset_summary,
}

run_aux_root.mkdir(parents=True, exist_ok=True)
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
context_path = log_path.parent / "context.json"

pid = int(pid_path.read_text(encoding="utf-8").strip()) if pid_path.exists() else None
running = False
if pid is not None:
    running = Path(f"/proc/{pid}").exists()

print(f"PID: {pid}")
print(f"Running: {running}")
print(f"Log path: {log_path}")
print(f"Save root: {save_root}")
print(f"Context path: {context_path}")

metrics_matches = sorted(save_root.rglob("metrics.json"))
metric_history_matches = sorted(save_root.rglob("metric_history.json"))
eval_matches = sorted(save_root.rglob("eval.md"))
sample_pngs = sorted(save_root.rglob("*.png"))

print("\\nArtifacts so far:\\n")
print(f"metrics.json matches: {[p.as_posix() for p in metrics_matches]}")
print(f"metric_history.json matches: {[p.as_posix() for p in metric_history_matches]}")
print(f"eval.md matches: {[p.as_posix() for p in eval_matches]}")
print(f"saved PNG count: {len(sample_pngs)}")

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
metrics_matches = sorted(save_root.rglob("metrics.json"))
metric_history_matches = sorted(save_root.rglob("metric_history.json"))
eval_matches = sorted(save_root.rglob("eval.md"))
grid_matches = sorted(save_root.rglob("grid_results.png"))

print("Artifacts:\\n")
print(f"save_root: {save_root}")
print(f"metrics.json matches: {[p.as_posix() for p in metrics_matches]}")
print(f"metric_history.json matches: {[p.as_posix() for p in metric_history_matches]}")
print(f"eval.md matches: {[p.as_posix() for p in eval_matches]}")
print(f"grid_results.png matches: {[p.as_posix() for p in grid_matches]}")

if eval_matches:
    eval_path = eval_matches[0]
    print("\\neval.md:\\n")
    print(eval_path.read_text(encoding="utf-8", errors="ignore"))
else:
    print("\\nNo eval.md found yet.")

if metrics_matches:
    metrics_path = metrics_matches[0]
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    print("\\nmetrics.json:\\n")
    print(json.dumps(metrics, indent=2))
else:
    print("\\nNo metrics.json found yet.")

if metric_history_matches:
    metric_history_path = metric_history_matches[0]
    metric_history = json.loads(metric_history_path.read_text(encoding="utf-8"))
    print("\\nmetric_history.json summary:\\n")
    print(f"path: {metric_history_path}")
    if isinstance(metric_history, dict) and "runs" in metric_history:
        print(f"runs captured: {len(metric_history['runs'])}")
        history_view = metric_history["runs"][0] if metric_history["runs"] else {}
    else:
        history_view = metric_history
    series_keys = [key for key, value in history_view.items() if isinstance(value, list)]
    print(f"series keys: {series_keys}")
    for key in series_keys:
        series = history_view[key]
        if not series:
            print(f"{key}: <empty>")
            continue
        tail = series[-5:] if len(series) > 5 else series
        print(f"{key}: len={len(series)} final={series[-1]} tail={tail}")
else:
    print("\\nNo metric_history.json found yet.")
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
    Path(last_context["hydra_root"]),
    Path(last_context["latest_log_path"]),
    Path(last_context["latest_log_path"]).parent / "context.json",
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


NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH}")
