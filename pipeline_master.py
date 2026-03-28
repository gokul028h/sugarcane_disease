import subprocess
import sys
import os
import yaml
from pathlib import Path

def run_command(command, description):
    print(f"\n>>> {description}...")
    try:
        # Using the venv python if possible, otherwise system python
        python_exe = os.path.join("myenv", "Scripts", "python.exe")
        if not os.path.exists(python_exe):
            python_exe = "python"
            
        full_cmd = [python_exe] + command
        subprocess.run(full_cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during {description}: {e}")
        sys.exit(1)

def main():
    # 1. Load Config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Ensure experiments directory exists
    Path("experiments").mkdir(exist_ok=True)

    print("====================================================")
    print("   SUGARCANE DISEASE DETECTION: FULL PIPELINE       ")
    print("====================================================")

    # Phase 1: Classification Training (Swin Transformer)
    # This acts as the backbone and the source for Grad-CAM masks.
    run_command(
        ["src/training/train_classification.py", "--model", "swin"], 
        "Phase 1: Training Swin Transformer Backbone"
    )

    # Phase 2: Pseudo-Mask Generation
    # Uses the trained Swin model to explain its predictions and save binary masks.
    run_command(
        ["src/data/generate_pseudo_masks.py"], 
        "Phase 2: Generating Weakly Supervised Pseudo-Masks"
    )

    # Phase 3: Segmentation Training (U-Net)
    # Trains the U-Net on the generated pseudo-masks for pixel-wise localization.
    run_command(
        ["src/training/train_segmentation.py"], 
        "Phase 3: Training U-Net Segmentation Model"
    )

    # Phase 4: Comparative Analysis & Ablation Studies
    # Train baseline models for the comparison tables in the paper.
    baselines = ["resnet_finetuned", "efficientnet"]
    for model in baselines:
        run_command(
            ["src/training/train_classification.py", "--model", model], 
            f"Phase 4: Training Baseline Model ({model})"
        )

    # Phase 5: Final Result Aggregation
    print("\n>>> Phase 5: Generating Final Comparison Tables and Plots...")
    # This would involve a call to generate_comparison_table from research_plots.py
    # We can create a small script for this or include it here if research_plots is importable.
    python_exe = os.path.join("myenv", "Scripts", "python.exe")
    if not os.path.exists(python_exe): python_exe = "python"
    
    summary_script = "from src.evaluation.research_plots import generate_comparison_table; generate_comparison_table('experiments', 'experiments/model_comparison_results.csv')"
    subprocess.run([python_exe, "-c", summary_script], check=True)

    print("\n====================================================")
    print("PIPELINE COMPLETE: Results saved in experiments/")
    print("You can now start the FastAPI server using: uvicorn api.app:app --reload")
    print("====================================================")

if __name__ == "__main__":
    main()
