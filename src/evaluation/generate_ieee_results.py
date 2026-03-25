"""
IEEE Results Generation Script.
Generates all publication-quality figures, tables, and visualizations
from existing experiment results.

Collects results from all trained models and experiments to produce:
- Model comparison table (Fig X)
- Ablation study table (Table X)
- Training curves comparison (Fig X)
- ROC/PR curves (Fig X)
- Confusion matrices (Fig X)
- Segmentation visualization grid (Fig X)
- Statistical CV results (Table X)
- Severity distribution (Fig X)
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import json
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from src.evaluation.visualize import (
    plot_model_comparison_table, plot_ablation_table,
    plot_multi_model_curves, plot_training_curves
)


def generate_all_ieee_figures():
    """Generate all IEEE publication figures from existing experiment data."""
    
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    OUTPUT_DIR = Path("experiments") / "ieee_results"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("  GENERATING IEEE PUBLICATION FIGURES")
    print("=" * 60)
    
    # 1. Collect model comparison data
    print("\n[1/6] Collecting model comparison data...")
    models = ["efficientnet", "convnext", "swin"]
    comparison_data = []
    all_histories = {}
    
    for model_name in models:
        exp_dir = Path(config["experiments"].get(model_name, f"experiments/{model_name}"))
        
        # Training metrics (for history, time, params)
        metrics_file = exp_dir / "metrics.json"
        eval_file = exp_dir / "evaluation_metrics.json"
        
        train_metrics = {}
        eval_metrics = {}
        
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                train_metrics = json.load(f)
            if "history" in train_metrics:
                all_histories[model_name] = train_metrics["history"]
        
        if eval_file.exists():
            with open(eval_file, "r") as f:
                eval_metrics = json.load(f)
        
        if eval_metrics:
            comparison_data.append({
                "model": model_name,
                "accuracy": eval_metrics.get("accuracy", 0),
                "precision": eval_metrics.get("macro_avg", {}).get("precision", 0),
                "recall": eval_metrics.get("macro_avg", {}).get("recall", 0),
                "f1": eval_metrics.get("macro_avg", {}).get("f1_score", 0),
                "params_M": train_metrics.get("total_parameters", 0) / 1e6,
                "training_time_min": train_metrics.get("total_training_time_seconds", 0) / 60.0
            })
    
    if comparison_data:
        plot_model_comparison_table(comparison_data, str(OUTPUT_DIR / "model_comparison_table.png"))
        print(f"  → Model comparison table saved")
    else:
        print(f"  → No evaluation data found. Run model training first.")
    
    # 2. Multi-model training curves
    print("\n[2/6] Generating multi-model training curves...")
    if all_histories:
        plot_multi_model_curves(all_histories, str(OUTPUT_DIR))
        print(f"  → Multi-model curves saved")
    
    # Per-model training curves
    for model_name, history in all_histories.items():
        model_fig_dir = OUTPUT_DIR / model_name
        model_fig_dir.mkdir(exist_ok=True)
        plot_training_curves(history, str(model_fig_dir), model_name)
        print(f"  → {model_name} training curves saved")
    
    # 3. Ablation results
    print("\n[3/6] Collecting ablation study results...")
    ablation_file = OUTPUT_DIR / "ablation_results.json"
    if ablation_file.exists():
        with open(ablation_file, "r") as f:
            ablation_data = json.load(f)
        plot_ablation_table(ablation_data, str(OUTPUT_DIR / "ablation_study_table.png"))
        print(f"  → Ablation table saved")
    else:
        print(f"  → No ablation data found. Run ablation_study.py first.")
    
    # 4. CV results summary
    print("\n[4/6] Collecting cross-validation results...")
    cv_summary = []
    for model_name in models:
        cv_file = OUTPUT_DIR / f"cv_results_{model_name}.json"
        if cv_file.exists():
            with open(cv_file, "r") as f:
                cv_data = json.load(f)
            cv_summary.append(cv_data)
            print(f"  → {model_name}: Acc = {cv_data['accuracy']['mean']:.4f} ± {cv_data['accuracy']['std']:.4f}")
    
    if cv_summary:
        # Generate CV comparison table
        fig, ax = plt.subplots(figsize=(14, 3 + len(cv_summary) * 0.8))
        ax.axis('off')
        
        columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'K-Folds']
        table_data = []
        for s in cv_summary:
            table_data.append([
                s['model'],
                f"{s['accuracy']['mean']:.4f} ± {s['accuracy']['std']:.4f}",
                f"{s['precision']['mean']:.4f} ± {s['precision']['std']:.4f}",
                f"{s['recall']['mean']:.4f} ± {s['recall']['std']:.4f}",
                f"{s['f1_score']['mean']:.4f} ± {s['f1_score']['std']:.4f}",
                str(s['k_folds'])
            ])
        
        table = ax.table(cellText=table_data, colLabels=columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.8)
        
        for j in range(len(columns)):
            table[0, j].set_facecolor('#4472C4')
            table[0, j].set_text_props(color='white', fontweight='bold')
        
        ax.set_title('Cross-Validation Results (Mean ± Std)', fontsize=14, fontweight='bold', pad=20)
        fig.savefig(OUTPUT_DIR / "cv_comparison_table.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  → CV comparison table saved")
    
    # 5. Segmentation metrics
    print("\n[5/6] Collecting segmentation results...")
    seg_metrics_file = Path(config["experiments"]["unet_segmentation"]) / "segmentation_metrics.json"
    if seg_metrics_file.exists():
        with open(seg_metrics_file, "r") as f:
            seg_metrics = json.load(f)
        print(f"  → IoU: {seg_metrics['mean_iou']:.4f} ± {seg_metrics['std_iou']:.4f}")
        print(f"  → Dice: {seg_metrics['mean_dice']:.4f} ± {seg_metrics['std_dice']:.4f}")
    else:
        print(f"  → No segmentation metrics found. Run segmentation_eval.py first.")
    
    # 6. Generate master summary
    print("\n[6/6] Generating master summary...")
    master_summary = {
        "model_comparison": comparison_data,
        "cv_summary": cv_summary if cv_summary else [],
        "segmentation": seg_metrics if seg_metrics_file.exists() else {},
    }
    
    with open(OUTPUT_DIR / "master_summary.json", "w") as f:
        json.dump(master_summary, f, indent=4, default=str)
    
    print(f"\n{'='*60}")
    print(f"  ALL IEEE FIGURES GENERATED")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'='*60}")
    
    # List generated files
    generated = list(OUTPUT_DIR.rglob("*.png")) + list(OUTPUT_DIR.rglob("*.json"))
    print(f"\nGenerated {len(generated)} files:")
    for f in sorted(generated):
        print(f"  📄 {f.relative_to(OUTPUT_DIR)}")


if __name__ == "__main__":
    generate_all_ieee_figures()
