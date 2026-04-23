import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "REDDiff_DPS_CT_Native_Counts_Colab.ipynb"


def code_cell(source: str, cell_id: str | None = None) -> dict:
    metadata = {}
    if cell_id is not None:
        metadata["id"] = cell_id
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
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


CT_DEGRADATION_CODE = r'''

class TransmissionCTNativeCountsH(H_functions):
    """Native transmission-count CT degradation for REDDIFF/DPS.

    Input images are expected in the RED-diff repo convention x in [-1, 1].
    The grayscale channel is mapped to nonnegative attenuation, projected with a
    differentiable rotate/sum projector, and measured as Poisson transmission
    counts: y ~ Poisson(I0 * exp(-A mu(x))).
    """

    measurement_model = "transmission_ct_native_counts"

    def __init__(
        self,
        channels,
        img_dim,
        num_angles=80,
        num_detectors=None,
        I0=10000.0,
        attenuation_min=0.0,
        attenuation_max=1.0,
        detector_scale=1.0,
        angle_offset_deg=0.0,
        loss_reduction="mean",
        device="cuda",
    ):
        self.channels = int(channels)
        self.img_dim = int(img_dim)
        self.num_angles = int(num_angles)
        self.num_detectors = int(num_detectors) if num_detectors is not None else int(img_dim)
        self.I0 = float(I0)
        self.attenuation_min = float(attenuation_min)
        self.attenuation_max = float(attenuation_max)
        self.detector_scale = float(detector_scale)
        self.angle_offset_deg = float(angle_offset_deg)
        self.loss_reduction = str(loss_reduction)
        self.device = torch.device(device)
        self.sigma = 0.0

        angles = torch.arange(self.num_angles, dtype=torch.float32, device=self.device)
        angles = angles * (np.pi / max(1, self.num_angles))
        angles = angles + np.deg2rad(self.angle_offset_deg)
        self.angles = angles

    def _as_gray(self, image):
        if image.dim() != 4:
            raise ValueError(f"Expected [B,C,H,W], got {tuple(image.shape)}")
        if image.shape[1] == 1:
            return image
        return image.mean(dim=1, keepdim=True)

    def _repeat_channels(self, image):
        if self.channels == 1:
            return image
        return image.repeat(1, self.channels, 1, 1)

    def _mu(self, image):
        gray = self._as_gray(image)
        x01 = ((gray + 1.0) * 0.5).clamp(0.0, 1.0)
        return self.attenuation_min + x01 * (self.attenuation_max - self.attenuation_min)

    def _resize_detector_axis(self, sino, num_detectors):
        if sino.shape[-1] == num_detectors:
            return sino
        b, c, a, d = sino.shape
        flat = sino.reshape(b * c * a, 1, d)
        flat = F.interpolate(flat, size=num_detectors, mode="linear", align_corners=False)
        return flat.reshape(b, c, a, num_detectors)

    def _forward_mu(self, mu):
        b, c, h, w = mu.shape
        projections = []
        dx = 2.0 / max(h, 1)
        angles = self.angles.to(device=mu.device, dtype=mu.dtype)
        for angle in angles:
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)
            theta = torch.stack(
                [
                    torch.stack([cos_a, -sin_a, torch.zeros_like(cos_a)]),
                    torch.stack([sin_a, cos_a, torch.zeros_like(cos_a)]),
                ]
            ).unsqueeze(0).expand(b, -1, -1)
            grid = F.affine_grid(theta, mu.size(), align_corners=False)
            rotated = F.grid_sample(mu, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            proj = rotated.sum(dim=-2) * dx * self.detector_scale
            projections.append(proj)
        sino = torch.stack(projections, dim=-2)
        return self._resize_detector_axis(sino, self.num_detectors)

    def _backproject_mu(self, sino, out_hw=None):
        b, c, _, _ = sino.shape
        h = self.img_dim if out_hw is None else int(out_hw[0])
        w = self.img_dim if out_hw is None else int(out_hw[1])
        dx = 2.0 / max(h, 1)
        sino = self._resize_detector_axis(sino, w)
        angles = self.angles.to(device=sino.device, dtype=sino.dtype)
        back = torch.zeros((b, c, h, w), device=sino.device, dtype=sino.dtype)
        for idx, angle in enumerate(angles):
            proj = sino[:, :, idx, :]
            slab = proj.unsqueeze(-2).expand(b, c, h, w) * dx * self.detector_scale
            cos_a = torch.cos(angle)
            sin_a = torch.sin(angle)
            theta = torch.stack(
                [
                    torch.stack([cos_a, sin_a, torch.zeros_like(cos_a)]),
                    torch.stack([-sin_a, cos_a, torch.zeros_like(cos_a)]),
                ]
            ).unsqueeze(0).expand(b, -1, -1)
            grid = F.affine_grid(theta, slab.size(), align_corners=False)
            back = back + F.grid_sample(slab, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        return back

    def line_integrals(self, image):
        return self._forward_mu(self._mu(image))

    def incident_counts(self, reference):
        return torch.full_like(reference, fill_value=float(self.I0))

    def H(self, image):
        z = self.line_integrals(image)
        return self.incident_counts(z) * torch.exp(-z).clamp_min(0.0)

    def measure(self, image):
        rate = self.H(image).clamp_min(0.0).clamp_max(1.0e12)
        return torch.poisson(rate)

    def _counts_to_line_integrals(self, counts):
        i0 = self.incident_counts(counts)
        frac = (counts.clamp_min(1.0) / i0.clamp_min(1.0)).clamp(min=1.0e-12, max=1.0)
        return -torch.log(frac)

    def measurement_loss_per_sample(self, image, counts):
        z = self.line_integrals(image)
        i0 = self.incident_counts(z)
        per_bin = i0 * torch.exp(-z).clamp_min(0.0) + counts * z
        flat = per_bin.flatten(1)
        if self.loss_reduction == "sum":
            return flat.sum(dim=1)
        return flat.mean(dim=1)

    def measurement_loss(self, image, counts):
        return self.measurement_loss_per_sample(image, counts).mean()

    def measurement_loss_and_norm(self, image, counts):
        per_sample = self.measurement_loss_per_sample(image, counts)
        return per_sample.sum(), per_sample.clamp_min(1.0e-12).sqrt()

    def H_pinv(self, counts):
        z = self._counts_to_line_integrals(counts)
        mu = self._backproject_mu(z) / max(self.num_angles, 1)
        denom = max(self.attenuation_max - self.attenuation_min, 1.0e-12)
        x01 = (mu - self.attenuation_min) / denom
        x = (x01 * 2.0 - 1.0).clamp(-1.0, 1.0)
        return self._repeat_channels(x)
'''


PATCH_CELL = r'''#@title Patch RED-diff Repo With Native-Count CT
import os
from pathlib import Path

os.chdir(REPO_DIR)

def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if old not in text:
        print(f"Pattern not found in {path}; leaving unchanged.")
        return
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"Patched {path}")

deg_path = Path(REPO_DIR) / "utils" / "degredations.py"
text = deg_path.read_text(encoding="utf-8")
if "class TransmissionCTNativeCountsH" not in text:
    marker = "\ndef build_degredation_model(cfg: DictConfig):\n"
    text = text.replace(marker, CT_DEGRADATION_CODE + marker, 1)
if 'elif deg in {"transmission_ct_native", "ct_native_counts"}:' not in text:
    branch = """    elif deg in {"transmission_ct_native", "ct_native_counts"}:
        H = TransmissionCTNativeCountsH(
            channels=c,
            img_dim=w,
            num_angles=int(_cfg_or_default(operator_cfg, "num_angles", 80)),
            num_detectors=int(_cfg_or_default(operator_cfg, "num_detectors", w)),
            I0=float(_cfg_or_default(operator_cfg, "I0", 10000.0)),
            attenuation_min=float(_cfg_or_default(operator_cfg, "attenuation_min", 0.0)),
            attenuation_max=float(_cfg_or_default(operator_cfg, "attenuation_max", 1.0)),
            detector_scale=float(_cfg_or_default(operator_cfg, "detector_scale", 1.0)),
            angle_offset_deg=float(_cfg_or_default(operator_cfg, "angle_offset_deg", 0.0)),
            loss_reduction=str(_cfg_or_default(operator_cfg, "loss_reduction", "mean")),
            device=device,
        )
"""
    text = text.replace('    if deg == "deno":\n        H = Denoising(c, w, device)\n', '    if deg == "deno":\n        H = Denoising(c, w, device)\n' + branch, 1)
deg_path.write_text(text, encoding="utf-8")
print(f"Native CT degradation ready in {deg_path}")

dps_path = Path(REPO_DIR) / "algos" / "dps.py"
replace_once(
    dps_path,
    "            mat_norm = ((y_0 - H.H(x0_pred)).reshape(n, -1) ** 2).sum(dim=1).sqrt().detach()\n            mat = ((y_0 - H.H(x0_pred)).reshape(n, -1) ** 2).sum()\n\n            grad_term = torch.autograd.grad(mat, xt, retain_graph=True)[0]\n",
    "            if hasattr(H, \"measurement_loss_and_norm\"):\n                mat, mat_norm = H.measurement_loss_and_norm(x0_pred, y_0)\n                mat_norm = mat_norm.detach()\n            else:\n                mat_norm = ((y_0 - H.H(x0_pred)).reshape(n, -1) ** 2).sum(dim=1).sqrt().detach()\n                mat = ((y_0 - H.H(x0_pred)).reshape(n, -1) ** 2).sum()\n\n            grad_term = torch.autograd.grad(mat, xt, retain_graph=True)[0]\n",
)

reddiff_path = Path(REPO_DIR) / "algos" / "reddiff.py"
replace_once(
    reddiff_path,
    "            e_obs = y_0 - H.H(x0_pred)\n            loss_obs = (e_obs**2).mean()/2\n",
    "            if hasattr(H, \"measurement_loss\"):\n                loss_obs = H.measurement_loss(x0_pred, y_0)\n            else:\n                e_obs = y_0 - H.H(x0_pred)\n                loss_obs = (e_obs**2).mean()/2\n",
)
'''


cells = [
    markdown_cell(
        """# RED-diff / DPS Native-Count CT Single Run In Colab

This notebook keeps the **RED-diff repo algorithm framework** (`algos/dps.py`, `algos/reddiff.py`, `main.py`) and adds a native transmission-count CT degradation to it.

The CT measurement model is the same class of model as the PDHG formulation:

`counts ~ Poisson(I0 * exp(-A mu(x)))`

Important scope:

- This is a same-framework REDDIFF/DPS CT integration notebook.
- DPS guidance is mapped through `algo.grad_term_weight`; `algo.eta` is kept as DDIM stochasticity.
- The default model config is still the RED-diff repo's `ffhq256_uncond` ADM checkpoint path. A compatible CT-trained checkpoint would be needed for a true medical-prior run in this exact framework.
""",
        cell_id="title",
    ),
    code_cell(
        """#@title Project Settings

SETUP_MODE = "git"  #@param ["git", "drive_zip"]
REPO_URL = "https://github.com/Seif-Hussein/daps_test.git"  #@param {type:"string"}
REPO_BRANCH = "codex-reddiff-colab-operators"  #@param {type:"string"}
DRIVE_ZIP_PATH = "/content/drive/MyDrive/RED-diff.zip"  #@param {type:"string"}

REPO_DIR = "/content/RED-diff"  #@param {type:"string"}
PYTHON_BIN = "/usr/bin/python3"  #@param {type:"string"}
DRIVE_EXPORT_DIR = "/content/drive/MyDrive/reddiff_ct_native_single_run_exports"  #@param {type:"string"}
DRIVE_CT_DATA_DIR = ""  #@param {type:"string"}
DRIVE_FFHQ_CKPT_PATH = ""  #@param {type:"string"}

RUN_NAME = "REDDiff_DPS_CT_Native_Counts"  #@param {type:"string"}
SESSION_TAG = ""  #@param {type:"string"}
BASE_CONFIG_NAME = "ffhq256_uncond"  #@param ["ffhq256_uncond"]
ALGO_NAME = "dps"  #@param ["dps", "reddiff"]

SEED = 99  #@param {type:"integer"}
TOTAL_IMAGES = 3  #@param {type:"integer"}
BATCH_SIZE = 1  #@param {type:"integer"}
DATA_START_IDX = 0  #@param {type:"integer"}
NUM_WORKERS = 0  #@param {type:"integer"}
LOG_TAIL_LINES = 120  #@param {type:"integer"}

PREP_OUTPUT_DIR = "/content/reddiff_ct_native_preprocessed"  #@param {type:"string"}
CT_VALUE_MIN = ""  #@param {type:"string"}
CT_VALUE_MAX = ""  #@param {type:"string"}
VALID_EXTENSIONS = ".tif,.tiff,.png,.jpg,.jpeg,.dcm,.ima"  #@param {type:"string"}

# DDPM/DDIM schedule used by this RED-diff repo.
NUM_STEPS = 1000  #@param {type:"integer"}

# Native-count CT acquisition.
I0 = 10000.0  #@param {type:"number"}
NUM_ANGLES = 80  #@param {type:"integer"}
NUM_DETECTORS = 256  #@param {type:"integer"}
ATTENUATION_MIN = 0.0  #@param {type:"number"}
ATTENUATION_MAX = 1.0  #@param {type:"number"}
CT_LOSS_REDUCTION = "mean"  #@param ["mean", "sum"]

# DPS settings in this RED-diff repo.
DPS_GRAD_TERM_WEIGHT = 1.0  #@param {type:"number"}
DPS_DDIM_ETA = 0.0  #@param {type:"number"}

# REDDIFF settings in this RED-diff repo.
REDDIFF_GRAD_TERM_WEIGHT = 0.25  #@param {type:"number"}
REDDIFF_OBS_WEIGHT = 1.0  #@param {type:"number"}
REDDIFF_LR = 0.01  #@param {type:"number"}
REDDIFF_DDIM_ETA = 0.0  #@param {type:"number"}
REDDIFF_DENOISE_TERM_WEIGHT = "linear"  #@param ["linear", "sqrt", "log", "square", "trunc_linear", "const", "power2over3"]
REDDIFF_SIGMA_X0 = 0.0001  #@param {type:"number"}

SAVE_ORI = True  #@param {type:"boolean"}
SAVE_DEG = True  #@param {type:"boolean"}
SAVE_EVOLUTION = False  #@param {type:"boolean"}
SMOKE_TEST = 0  #@param {type:"integer"}

# Optional extra Hydra overrides, separated by semicolons.
EXTRA_HYDRA_OVERRIDES = ""  #@param {type:"string"}
""",
        cell_id="settings",
    ),
    code_cell(
        """#@title Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
""",
        cell_id="mount-drive",
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
        "git", "clone", "--branch", REPO_BRANCH, "--single-branch", REPO_URL, repo_dir.as_posix()
    ], check=True)
elif SETUP_MODE == "drive_zip":
    zip_path = Path(DRIVE_ZIP_PATH)
    if not zip_path.exists():
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(repo_dir.parent)
    extracted_root = repo_dir.parent / zip_path.stem
    if extracted_root.exists() and extracted_root != repo_dir:
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        extracted_root.rename(repo_dir)
else:
    raise ValueError(f"Unsupported SETUP_MODE: {SETUP_MODE}")

os.chdir(repo_dir)
print(f"Repo ready: {repo_dir}")
""",
        cell_id="fetch-repo",
    ),
    code_cell(
        """#@title Install Colab Dependencies And Checkpoint
import os
import shutil
import subprocess
from pathlib import Path

os.chdir(REPO_DIR)
req_path = Path("requirements-colab.txt")
if req_path.exists():
    subprocess.run([PYTHON_BIN, "-m", "pip", "install", "-q", "-r", req_path.as_posix()], check=True)
    print(f"Installed {req_path}")
else:
    fallback = [
        "hydra-core==1.3.2",
        "omegaconf==2.3.0",
        "gdown==5.2.0",
        "lmdb>=1.4,<2",
        "tensorboard>=2.14",
        "scipy==1.14.1",
        "torchmetrics>=1.4,<2",
        "pydicom>=2.4,<4",
    ]
    subprocess.run([PYTHON_BIN, "-m", "pip", "install", "-q", *fallback], check=True)
    print("Installed fallback Colab packages")

ffhq_ckpt_id = "1BGwhRWUoguF-D8wlZ65tf227gp3cDUDh"
repo_dir = Path(REPO_DIR)
ffhq_ckpt_path = repo_dir / "colab_runs" / "ckpts" / "ffhq" / "ffhq_10m.pt"
ffhq_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
drive_ckpt_path = Path(DRIVE_FFHQ_CKPT_PATH) if str(DRIVE_FFHQ_CKPT_PATH).strip() else None
legacy_ckpt_path = repo_dir / "pretrained-models" / "ffhq_10m.pt"

if ffhq_ckpt_path.exists():
    print(f"Checkpoint already present: {ffhq_ckpt_path}")
elif drive_ckpt_path is not None and drive_ckpt_path.exists():
    shutil.copy2(drive_ckpt_path, ffhq_ckpt_path)
    print(f"Copied checkpoint from Drive to: {ffhq_ckpt_path}")
elif legacy_ckpt_path.exists():
    shutil.copy2(legacy_ckpt_path, ffhq_ckpt_path)
    print(f"Copied checkpoint from repo cache to: {ffhq_ckpt_path}")
else:
    subprocess.run(["gdown", "--id", ffhq_ckpt_id, "-O", ffhq_ckpt_path.as_posix()], check=True)
    print(f"Downloaded checkpoint to: {ffhq_ckpt_path}")
""",
        cell_id="install",
    ),
    code_cell(f"CT_DEGRADATION_CODE = {CT_DEGRADATION_CODE!r}\n\n{PATCH_CELL}", cell_id="patch-ct"),
    code_cell(
        """#@title Prepare CT Slices For The RED-diff Dataset Loader
import json
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

os.chdir(REPO_DIR)

valid_exts = {ext.strip().lower() for ext in VALID_EXTENSIONS.split(",") if ext.strip()}

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

if DRIVE_CT_DATA_DIR.strip():
    input_root = Path(DRIVE_CT_DATA_DIR)
else:
    input_root = Path(REPO_DIR) / "demo-samples" / "ct_l067_original_ima_subset"

if not input_root.exists():
    raise FileNotFoundError(f"CT data folder not found: {input_root}")

all_files = sorted([p for p in input_root.rglob("*") if p.is_file() and p.suffix.lower() in valid_exts])
if not all_files:
    raise FileNotFoundError(f"No CT files with extensions {sorted(valid_exts)} under {input_root}")

start = int(DATA_START_IDX)
end = min(start + int(TOTAL_IMAGES), len(all_files))
selected_files = all_files[start:end]
if not selected_files:
    raise ValueError("Selected CT slice range is empty.")

if CT_VALUE_MIN.strip() and CT_VALUE_MAX.strip():
    value_min = float(CT_VALUE_MIN)
    value_max = float(CT_VALUE_MAX)
elif (not DRIVE_CT_DATA_DIR.strip()) and input_root.name == "ct_l067_original_ima_subset":
    value_min = -1024.0
    value_max = 3071.0
else:
    value_min = min(float(np.nanmin(_load_ct_array(path))) for path in all_files)
    value_max = max(float(np.nanmax(_load_ct_array(path))) for path in all_files)

if value_max <= value_min:
    raise ValueError(f"Invalid CT value range: {value_min}, {value_max}")

prep_root = Path(PREP_OUTPUT_DIR)
if prep_root.exists():
    shutil.rmtree(prep_root)
prep_root.mkdir(parents=True, exist_ok=True)

prepared = []
for idx, src in enumerate(selected_files):
    arr = _load_ct_array(src)
    arr01 = ((arr - value_min) / (value_max - value_min)).clip(0.0, 1.0)
    uint8 = (arr01 * 255.0 + 0.5).astype(np.uint8)
    pil = Image.fromarray(uint8, mode="L").resize((256, 256), Image.BICUBIC).convert("RGB")
    out = prep_root / f"ct_{idx:05d}_{src.stem}.png"
    pil.save(out)
    prepared.append(out)

print("Prepared CT slices:")
for path in prepared:
    print(" -", path)
print(f"CT value range: [{value_min}, {value_max}]")
""",
        cell_id="prepare-data",
    ),
    code_cell(
        """#@title Build Single-Run Command
import json
import os
import shlex
import time
from pathlib import Path

os.chdir(REPO_DIR)

def strip_wrapped_quotes(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        return text[1:-1].strip()
    return text

def parse_list(text: str):
    items = []
    for raw_item in text.replace("\\n", ";").split(";"):
        raw_item = strip_wrapped_quotes(raw_item.strip())
        if raw_item:
            items.append(raw_item)
    return items

session_tag = SESSION_TAG.strip() or time.strftime("%Y%m%d-%H%M%S")
run_slug = f"{ALGO_NAME}_transmission_ct_native_{session_tag}"
run_name = f"{RUN_NAME}_{session_tag}"

repo_dir = Path(REPO_DIR)
run_aux_root = repo_dir / "colab_runs"
subset_root = run_aux_root / "subsets"
subset_root.mkdir(parents=True, exist_ok=True)
subset_path = subset_root / f"{run_slug}.txt"

prepared_root = Path(PREP_OUTPUT_DIR)
prepared_files = sorted(prepared_root.glob("*.png"))
subset_path.write_text("".join(f"{path.name} 0\\n" for path in prepared_files), encoding="utf-8")

exp_root = run_aux_root
samples_root = exp_root / "samples" / run_slug
progress_path = samples_root / "progress.json"
metrics_history_path = samples_root / "metrics_history.json"
latest_log_path = run_aux_root / f"{run_slug}.log"
latest_pid_path = run_aux_root / f"{run_slug}.pid"

effective_batch_size = min(int(BATCH_SIZE), max(1, len(prepared_files)))

run_cmd = [
    PYTHON_BIN,
    "main.py",
    "-cn",
    BASE_CONFIG_NAME,
    "dataset=drive_images_256",
    f"dataset.root={prepared_root.as_posix()}",
    f"dataset.subset_txt={subset_path.as_posix()}",
    f"algo={ALGO_NAME}",
    "algo.deg=transmission_ct_native",
    "classifier=none",
    "dist.num_processes_per_node=1",
    f"exp.root={exp_root.as_posix()}",
    f"exp.name={run_slug}",
    "exp.samples_root=samples",
    "exp.overwrite=true",
    f"exp.seed={SEED}",
    f"exp.num_steps={NUM_STEPS}",
    "exp.write_progress_json=true",
    "exp.progress_json=progress.json",
    "exp.write_metrics_history_json=true",
    "exp.metrics_history_json=metrics_history.json",
    f"exp.save_ori={'true' if SAVE_ORI else 'false'}",
    f"exp.save_deg={'true' if SAVE_DEG else 'false'}",
    f"exp.save_evolution={'true' if SAVE_EVOLUTION else 'false'}",
    f"exp.smoke_test={SMOKE_TEST}",
    f"loader.batch_size={effective_batch_size}",
    f"loader.num_workers={NUM_WORKERS}",
    "loader.shuffle=false",
    "loader.drop_last=false",
    "loader.pin_memory=true",
    "algo.sigma_y=0.0",
    "algo.operator.sigma=0.0",
    f"+algo.operator.I0={float(I0)}",
    f"+algo.operator.num_angles={int(NUM_ANGLES)}",
    f"+algo.operator.num_detectors={int(NUM_DETECTORS)}",
    f"+algo.operator.attenuation_min={float(ATTENUATION_MIN)}",
    f"+algo.operator.attenuation_max={float(ATTENUATION_MAX)}",
    f"+algo.operator.loss_reduction={CT_LOSS_REDUCTION}",
]

if ALGO_NAME == "dps":
    run_cmd.extend([
        "algo.awd=true",
        f"algo.grad_term_weight={float(DPS_GRAD_TERM_WEIGHT)}",
        f"algo.eta={float(DPS_DDIM_ETA)}",
    ])
elif ALGO_NAME == "reddiff":
    run_cmd.extend([
        "algo.awd=true",
        f"algo.grad_term_weight={float(REDDIFF_GRAD_TERM_WEIGHT)}",
        f"algo.obs_weight={float(REDDIFF_OBS_WEIGHT)}",
        f"algo.lr={float(REDDIFF_LR)}",
        f"algo.eta={float(REDDIFF_DDIM_ETA)}",
        f"algo.denoise_term_weight={REDDIFF_DENOISE_TERM_WEIGHT}",
        f"algo.sigma_x0={float(REDDIFF_SIGMA_X0)}",
    ])
else:
    raise ValueError(f"Unsupported ALGO_NAME: {ALGO_NAME}")

run_cmd.extend(parse_list(EXTRA_HYDRA_OVERRIDES))

print(f"Run slug: {run_slug}")
print(f"Algorithm: {ALGO_NAME}")
print(f"Prepared images: {len(prepared_files)}")
print(f"Effective batch size: {effective_batch_size}")
if ALGO_NAME == "dps":
    print(f"DPS guidance weight: {DPS_GRAD_TERM_WEIGHT}")
    print(f"DPS DDIM eta: {DPS_DDIM_ETA}")
else:
    print(f"REDDIFF grad weight: {REDDIFF_GRAD_TERM_WEIGHT}")
    print(f"REDDIFF obs weight: {REDDIFF_OBS_WEIGHT}")
print(f"Samples root: {samples_root}")
print(f"Progress JSON: {progress_path}")
print("\\nCommand:")
print(" ".join(shlex.quote(part) for part in run_cmd))

run_context = {
    "run_slug": run_slug,
    "samples_root": samples_root.as_posix(),
    "progress_path": progress_path.as_posix(),
    "metrics_history_path": metrics_history_path.as_posix(),
    "latest_log_path": latest_log_path.as_posix(),
    "latest_pid_path": latest_pid_path.as_posix(),
    "run_cmd": run_cmd,
}
""",
        cell_id="build-command",
    ),
    code_cell(
        """#@title Run
import subprocess
from pathlib import Path

latest_log_path = Path(run_context["latest_log_path"])
latest_pid_path = Path(run_context["latest_pid_path"])
latest_log_path.parent.mkdir(parents=True, exist_ok=True)

with latest_log_path.open("w", encoding="utf-8") as log_file:
    proc = subprocess.Popen(run_context["run_cmd"], stdout=log_file, stderr=subprocess.STDOUT)
    latest_pid_path.write_text(str(proc.pid), encoding="utf-8")
    print(f"Started PID {proc.pid}")
    print(f"Log: {latest_log_path}")
    ret = proc.wait()

print(f"Process exited with code {ret}")
if ret != 0:
    raise RuntimeError(f"Run failed with exit code {ret}. Check {latest_log_path}")
""",
        cell_id="run",
    ),
    code_cell(
        """#@title Show Artifacts
import json
from pathlib import Path

samples_root = Path(run_context["samples_root"])
progress_path = Path(run_context["progress_path"])
history_path = Path(run_context["metrics_history_path"])
log_path = Path(run_context["latest_log_path"])

print(f"samples_root: {samples_root}")
print(f"progress.json: {progress_path.exists()}")
print(f"metrics_history.json: {history_path.exists()}")

if history_path.exists():
    history = json.loads(history_path.read_text(encoding="utf-8"))
    print("\\nmetrics_history.json:")
    entries = history.get("entries", [])
    if entries:
        last = entries[-1]
        print(json.dumps({
            "status": history.get("status"),
            "mean_psnr": last.get("mean_psnr"),
            "mean_ssim": last.get("mean_ssim"),
            "mean_lpips": last.get("mean_lpips"),
            "completed_images": last.get("completed_images"),
        }, indent=2))
    else:
        print(json.dumps(history, indent=2)[:4000])

pngs = sorted(samples_root.rglob("*.png"))
print(f"png_count: {len(pngs)}")
for path in pngs[:20]:
    print(path)

if log_path.exists():
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    print("\\nLog tail:")
    print("\\n".join(lines[-int(LOG_TAIL_LINES):]))
""",
        cell_id="artifacts",
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(f"Wrote {NOTEBOOK_PATH}")
