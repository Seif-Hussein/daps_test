import json
from pathlib import Path


BASE_NOTEBOOK = Path("scripts/FFHQ100_DAPS_Benchmark_Colab.ipynb")


TASKS = [
    {
        "label": "Super-Resolution",
        "filename": "FFHQ100_DAPS_SuperResolution_Colab.ipynb",
        "task_config": "down_sampling",
        "run_name": "superres_1k_5x200_ffhq100_benchmark",
        "diffusion_steps": 5,
        "annealing_steps": 200,
        "description": "super-resolution (x4)",
    },
    {
        "label": "Random Inpainting",
        "filename": "FFHQ100_DAPS_InpaintingRandom_Colab.ipynb",
        "task_config": "inpainting_rand",
        "run_name": "inpainting_random_1k_5x200_ffhq100_benchmark",
        "diffusion_steps": 5,
        "annealing_steps": 200,
        "description": "random inpainting (70% mask)",
    },
    {
        "label": "Motion Deblur",
        "filename": "FFHQ100_DAPS_MotionDeblur_Colab.ipynb",
        "task_config": "motion_blur",
        "run_name": "motion_deblur_1k_5x200_ffhq100_benchmark",
        "diffusion_steps": 5,
        "annealing_steps": 200,
        "description": "motion deblurring",
    },
    {
        "label": "Gaussian Blur",
        "filename": "FFHQ100_DAPS_GaussianBlur_Colab.ipynb",
        "task_config": "gaussian_blur",
        "run_name": "gaussian_blur_1k_5x200_ffhq100_benchmark",
        "diffusion_steps": 5,
        "annealing_steps": 200,
        "description": "Gaussian deblurring",
    },
    {
        "label": "Box Inpainting",
        "filename": "FFHQ100_DAPS_InpaintingBox_Colab.ipynb",
        "task_config": "inpainting",
        "run_name": "inpainting_box_1k_5x200_ffhq100_benchmark",
        "diffusion_steps": 5,
        "annealing_steps": 200,
        "description": "box inpainting",
    },
    {
        "label": "Phase Retrieval",
        "filename": "FFHQ100_DAPS_PhaseRetrieval_Colab.ipynb",
        "task_config": "phase_retrieval",
        "run_name": "phase_retrieval_4k_10x400_ffhq100_benchmark",
        "diffusion_steps": 10,
        "annealing_steps": 400,
        "description": "phase retrieval",
    },
]


def replace_setting(cell_source, old, new):
    updated = []
    for line in cell_source:
        updated.append(line.replace(old, new))
    return updated


def build_intro(task):
    return [
        f"# FFHQ100 DAPS {task['label']} on Colab\n",
        "\n",
        f"This notebook runs DAPS on the 100-image FFHQ benchmark set for **{task['description']}**.\n",
        "\n",
        "Default behavior:\n",
        "\n",
        "- clones the repo locally into `/content/DAPS-main`\n",
        "- still mounts Google Drive for dataset access\n",
        "- uses the dataset from Google Drive at `/content/drive/MyDrive/mycode/test-ffhq`\n",
        "- runs all PNGs in that folder by default\n",
        f"- uses the paper-style per-run step budget for this task: `{task['diffusion_steps']}` ODE / diffusion steps and `{task['annealing_steps']}` annealing steps\n",
        "- saves final metrics to `metrics.json`\n",
        "- saves per-iteration average PSNR / SSIM / LPIPS trajectories to `metrics_evolution.json`\n",
        "- disables full trajectory tensor saving to avoid Colab RAM growth\n",
        "\n",
        "Before running, switch Colab to a GPU runtime with `Runtime -> Change runtime type -> GPU`.\n",
    ]


def main():
    base = json.loads(BASE_NOTEBOOK.read_text(encoding="utf-8"))

    for task in TASKS:
        nb = json.loads(json.dumps(base))
        nb["cells"][0]["source"] = build_intro(task)

        cell5 = nb["cells"][5]["source"]
        cell5 = replace_setting(cell5, 'TASK_CONFIG = "phase_retrieval"', f'TASK_CONFIG = "{task["task_config"]}"')
        cell5 = replace_setting(cell5, "DIFFUSION_STEPS = 10", f"DIFFUSION_STEPS = {task['diffusion_steps']}")
        cell5 = replace_setting(cell5, "ANNEALING_STEPS = 400", f"ANNEALING_STEPS = {task['annealing_steps']}")
        cell5 = replace_setting(cell5, "BATCH_SIZE = 10", "BATCH_SIZE = 100")
        cell5 = replace_setting(cell5, 'DATA_ROOT = "dataset/demo-ffhq"', 'DRIVE_FFHQ_DATA_DIR = "/content/drive/MyDrive/mycode/test-ffhq"\nDATA_ROOT = DRIVE_FFHQ_DATA_DIR')
        cell5 = replace_setting(
            cell5,
            'RUN_NAME = f"phase_retrieval_4k_10x400_ffhq100_benchmark"',
            f'RUN_NAME = f"{task["run_name"]}"',
        )
        nb["cells"][5]["source"] = cell5

        out_path = Path("scripts") / task["filename"]
        out_path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
