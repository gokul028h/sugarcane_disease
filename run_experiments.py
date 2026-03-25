"""
Master Experiment Runner for IEEE Publication.
Orchestrates the complete experimental pipeline:
1. Train all classification models
2. Generate pseudo masks
3. Train segmentation model
4. Run evaluations
5. Run ablation study
6. Run K-Fold CV
7. Failure analysis
8. Generate IEEE figures

Usage:
    python run_experiments.py --all                  # Run everything
    python run_experiments.py --train                # Train all models
    python run_experiments.py --evaluate             # Evaluate all models
    python run_experiments.py --ablation             # Run ablation study
    python run_experiments.py --cv --model swin      # Run CV for a specific model
    python run_experiments.py --figures              # Generate IEEE figures only
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import argparse
import yaml
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Master Experiment Runner")
    parser.add_argument("--all", action="store_true", help="Run complete pipeline")
    parser.add_argument("--train", action="store_true", help="Train all models")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate all models")
    parser.add_argument("--segmentation", action="store_true", help="Train + evaluate segmentation")
    parser.add_argument("--pseudo-masks", action="store_true", help="Generate pseudo masks")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--cv", action="store_true", help="Run K-Fold cross-validation")
    parser.add_argument("--failures", action="store_true", help="Run failure analysis")
    parser.add_argument("--figures", action="store_true", help="Generate IEEE figures")
    parser.add_argument("--model", type=str, default=None, help="Specific model name")
    parser.add_argument("--models", nargs="+", default=None, help="List of models")
    parser.add_argument("--skip-training", action="store_true", help="Skip training in comparison")
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    default_models = ["efficientnet", "convnext", "swin"]
    models = args.models or default_models

    if args.all or args.train:
        print("\n" + "=" * 70)
        print("  STAGE 1: TRAINING ALL CLASSIFICATION MODELS")
        print("=" * 70)
        from src.evaluation.model_comparison import run_model_comparison
        run_model_comparison(models=models, skip_training=args.skip_training)

    if args.all or args.pseudo_masks:
        print("\n" + "=" * 70)
        print("  STAGE 2: GENERATING PSEUDO MASKS")
        print("=" * 70)
        from src.data.generate_pseudo_masks import generate_masks
        generate_masks()

    if args.all or args.segmentation:
        print("\n" + "=" * 70)
        print("  STAGE 3: TRAINING SEGMENTATION MODEL")
        print("=" * 70)
        from src.training.train_segmentation import train_segmentation
        train_segmentation()

        print("\n  Evaluating segmentation...")
        from src.evaluation.segmentation_eval import evaluate_segmentation
        evaluate_segmentation()

    if args.all or args.evaluate:
        print("\n" + "=" * 70)
        print("  STAGE 4: EVALUATING ALL MODELS")
        print("=" * 70)
        from src.evaluation.evaluate_advanced import evaluate_model
        for model_name in models:
            evaluate_model(model_name)

    if args.all or args.ablation:
        print("\n" + "=" * 70)
        print("  STAGE 5: ABLATION STUDY")
        print("=" * 70)
        from src.evaluation.ablation_study import run_full_ablation
        run_full_ablation()

    if args.all or args.cv:
        print("\n" + "=" * 70)
        print("  STAGE 6: K-FOLD CROSS-VALIDATION")
        print("=" * 70)
        from src.evaluation.statistical_report import run_statistical_cv
        cv_model = args.model or "efficientnet"
        run_statistical_cv(cv_model)

    if args.all or args.failures:
        print("\n" + "=" * 70)
        print("  STAGE 7: FAILURE ANALYSIS")
        print("=" * 70)
        from src.evaluation.failure_analysis import analyze_failures
        failure_model = args.model or "efficientnet"
        analyze_failures(failure_model)

    if args.all or args.figures:
        print("\n" + "=" * 70)
        print("  STAGE 8: GENERATING IEEE FIGURES")
        print("=" * 70)
        from src.evaluation.generate_ieee_results import generate_all_ieee_figures
        generate_all_ieee_figures()

    print("\n" + "=" * 70)
    print("  EXPERIMENT PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nResults saved under: experiments/")
    print(f"IEEE figures saved under: experiments/ieee_results/")


if __name__ == "__main__":
    main()
