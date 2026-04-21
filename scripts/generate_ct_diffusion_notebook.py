import json
from pathlib import Path


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

Medical CT defaults taken from DM4CT Table 11:

- DPS: step size `eta = 10`
- RED-diff: learning rate `0.01`
- RED-diff: factor on measurement consistency error `0.5/1/1/1`
- RED-diff: factor on noise fit error `1e4`

This launcher defaults to the same 80-angle noisy CT setup used by the original single-run CT notebook, so `REDDIFF_MEAS_WEIGHT` starts at `1.0`. If you want the clean 40-angle medical setting from the benchmark, set `REDDIFF_MEAS_WEIGHT = 0.5`.

The notebook also applies a small post-clone patch so:

- DPS uses its configured CT step size instead of a hardcoded clamp
- RED-diff exposes separate CT measurement-consistency and noise-fit weights
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
REDDIFF_WEIGHT_TYPE = "inv_snr"  #@param ["constant", "sigma", "sigma2", "inv_snr", "sqrt_inv_snr"]
REDDIFF_DATA_TERM = "nll"  #@param ["nll", "mse"]
REDDIFF_INIT_MODE = "noise"  #@param ["noise", "random", "pinv", "measurement", "y"]

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
        """#@title Apply CT Sampler Compatibility Patches
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
import os
import shlex
from datetime import datetime
from pathlib import Path

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

effective_dps_step_size = float(DPS_STEP_SIZE)
effective_dps_zeta_mode = DPS_ZETA_MODE
effective_reddiff_lr = float(REDDIFF_LR)
effective_reddiff_meas_weight = float(REDDIFF_MEAS_WEIGHT)
effective_reddiff_noise_weight = float(REDDIFF_NOISE_WEIGHT)

if USE_CT_BENCHMARK_PRESET:
    if SAMPLER_CONFIG == "edm_dps":
        effective_dps_step_size = 10.0
        effective_dps_zeta_mode = "constant"
    elif SAMPLER_CONFIG == "edm_reddiff":
        effective_reddiff_lr = 0.01
        effective_reddiff_noise_weight = 1.0e4

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
    f"inverse_task.operator.I0={I0}",
    f"inverse_task.operator.num_angles={NUM_ANGLES}",
    f"model.model_config.local_files_only={'true' if local_checkpoint_path else 'false'}",
]

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
        f"+inverse_task.admm_config.red_diff.data_term={REDDIFF_DATA_TERM}",
        f"+inverse_task.admm_config.red_diff.init={REDDIFF_INIT_MODE}",
        f"+inverse_task.admm_config.red_diff.weight_type={REDDIFF_WEIGHT_TYPE}",
        f"+inverse_task.admm_config.red_diff.time_sampling={REDDIFF_TIME_SAMPLING}",
        f"+inverse_task.admm_config.red_diff.time_spacing={REDDIFF_TIME_SPACING}",
    ])
else:
    raise ValueError(f"Unsupported sampler config: {SAMPLER_CONFIG}")

if CONFIG_NAME == "default_ct_admm.yaml":
    overrides.append(f"inverse_task.admm_config.denoise.lgvd.num_steps={ADMM_LGVD_NUM_STEPS}")

if DRIVE_CT_DATA_DIR.strip():
    drive_data_dir = Path(DRIVE_CT_DATA_DIR)
    if not drive_data_dir.exists():
        raise FileNotFoundError(f"CT data folder not found: {drive_data_dir}")
    overrides.extend([
        f"data.image_root_path={drive_data_dir.as_posix()}",
        f"data.start_idx={DATA_START_IDX}",
        f"data.end_idx={DATA_END_IDX}",
    ])
else:
    overrides.extend([
        f"data.start_idx={DATA_START_IDX}",
        f"data.end_idx={DATA_END_IDX}",
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
    "dps_step_size": effective_dps_step_size if SAMPLER_CONFIG == "edm_dps" else None,
    "dps_zeta_mode": effective_dps_zeta_mode if SAMPLER_CONFIG == "edm_dps" else None,
    "reddiff_lr": effective_reddiff_lr if SAMPLER_CONFIG == "edm_reddiff" else None,
    "reddiff_meas_weight": effective_reddiff_meas_weight if SAMPLER_CONFIG == "edm_reddiff" else None,
    "reddiff_noise_weight": effective_reddiff_noise_weight if SAMPLER_CONFIG == "edm_reddiff" else None,
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
