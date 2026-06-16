from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_YAML = SCRIPT_DIR / "YOLO Waste Detection.v2-add-new-photos.yolov8" / "data.yaml"
FILTERED_DATASET_DIR = SCRIPT_DIR / "filtered_datasets" / "waste_no_organic.yolov8"
SAMPLED_FILTERED_DATASET_DIR = SCRIPT_DIR / "filtered_datasets" / "waste_no_organic_train_1of3.yolov8"
ORGANIC_CLASS_NAME = "Organic"
SPLIT_DIRS = {"train": "train", "val": "valid", "test": "test"}
DEFAULT_RUN_NAME = "waste_yolov8n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune an Ultralytics YOLOv8 model on a Roboflow YOLOv8 waste dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_YAML),
        help="Path to Roboflow YOLOv8 data.yaml",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="Base model or checkpoint path")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=416, help="Training image size")
    parser.add_argument(
        "--batch",
        default="16",
        help='Training batch size. Use "auto" for CUDA AutoBatch, or an integer such as 32.',
    )
    parser.add_argument("--name", default=DEFAULT_RUN_NAME, help="Run name under runs/detect")
    parser.add_argument(
        "--exclude-organic",
        action="store_true",
        help="Use a generated 9-class dataset that removes Organic. By default, the original full dataset is used.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume from a checkpoint path. If no path is provided, uses runs/detect/{name}/weights/last.pt.",
    )
    parser.add_argument(
        "--rebuild-filtered-data",
        action="store_true",
        help="Rebuild the generated 9-class dataset that excludes Organic and applies train sampling.",
    )
    parser.add_argument(
        "--sample-train-one-per-group",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="When --exclude-organic is used, keep one image per Roboflow augmentation group in train split only.",
    )
    parser.add_argument(
        "--augmentation",
        choices=("auto", "none", "light"),
        default="auto",
        help=(
            "Online augmentation mode. auto keeps the previous behavior: light only for the reduced "
            "1-of-3 filtered train split, none for the original full dataset."
        ),
    )
    parser.add_argument(
        "--continue-from-run",
        default=None,
        help="Continue fine-tuning from runs/detect/{run}/weights/{last,best}.pt as a new run.",
    )
    parser.add_argument(
        "--continue-to-epochs",
        type=int,
        default=150,
        help="Target total epoch count when --continue-from-run is used.",
    )
    parser.add_argument(
        "--continue-weights",
        choices=("last", "best"),
        default="last",
        help="Checkpoint to use from --continue-from-run.",
    )

    args = parser.parse_args()
    args.name_was_set = any(arg == "--name" or arg.startswith("--name=") for arg in sys.argv[1:])
    return args


def require_dependencies() -> tuple[Any, Any, Any]:
    """Import training dependencies and print a helpful virtualenv hint if missing."""
    # Keep Ultralytics settings away from AppData permission issues. Ultralytics
    # appends an "Ultralytics" subfolder to YOLO_CONFIG_DIR, so create the root.
    if "YOLO_CONFIG_DIR" not in os.environ:
        for config_root in (
            Path(__file__).resolve().parent / ".ultralytics",
            Path(tempfile.gettempdir()) / "yolotune-ultralytics",
        ):
            try:
                config_root.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue
            os.environ["YOLO_CONFIG_DIR"] = str(config_root)
            break

    try:
        import torch
        import yaml
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        print(f"[ERROR] Missing Python package: {exc.name}", file=sys.stderr)
        print("Install dependencies inside the project virtual environment:", file=sys.stderr)
        print(r"  python -m venv .venv", file=sys.stderr)
        print(r"  .\.venv\Scripts\python.exe -m pip install --upgrade pip", file=sys.stderr)
        print(r"  .\.venv\Scripts\python.exe -m pip install ultralytics", file=sys.stderr)
        raise SystemExit(1) from exc
    return torch, YOLO, yaml


def validate_data_yaml(data_path: str) -> Path:
    data_yaml = Path(data_path).expanduser()
    if not data_yaml.exists():
        print(f"[ERROR] data.yaml not found: {data_yaml}", file=sys.stderr)
        print("Pass a valid Roboflow YOLOv8 data.yaml path with --data.", file=sys.stderr)
        raise SystemExit(1)
    if not data_yaml.is_file():
        print(f"[ERROR] --data must point to a file, not a directory: {data_yaml}", file=sys.stderr)
        raise SystemExit(1)
    return data_yaml.resolve()


def normalize_names(names: Any) -> list[str]:
    if isinstance(names, dict):
        return [names[key] for key in sorted(names, key=lambda value: int(value))]
    if isinstance(names, list):
        return names
    raise ValueError("data.yaml names must be a list or dictionary.")


def resolve_split_dir(data_yaml: Path, data: dict[str, Any], split: str) -> Path | None:
    split_value = data.get(split)
    if not split_value:
        return None
    if isinstance(split_value, list):
        raise ValueError(f"This script expects a single image directory for '{split}', not a list.")

    root = Path(data.get("path") or data_yaml.parent)
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    image_dir = (root / split_value).resolve()
    if not image_dir.exists() and isinstance(split_value, str) and split_value.startswith("../"):
        image_dir = (root / split_value[3:]).resolve()
    return image_dir


def image_to_label_path(image_path: Path, image_dir: Path) -> Path:
    return image_dir.parent / "labels" / f"{image_path.stem}.txt"


def roboflow_group_key(image_path: Path) -> str:
    """Group Roboflow augmentations by the original filename before the .rf. hash."""
    if ".rf." in image_path.name:
        return image_path.name.split(".rf.", 1)[0]
    return image_path.stem


def link_or_copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def build_filtered_dataset(
    source_yaml: Path,
    output_dir: Path,
    yaml_module: Any,
    rebuild: bool,
    sample_train_one_per_group: bool,
) -> Path:
    filtered_yaml = output_dir / "data.yaml"
    if filtered_yaml.exists() and not rebuild:
        print(f"Filtered dataset: {filtered_yaml}")
        return filtered_yaml.resolve()

    if output_dir.exists() and rebuild:
        shutil.rmtree(output_dir)

    source_data = yaml_module.safe_load(source_yaml.read_text(encoding="utf-8"))
    names = normalize_names(source_data["names"])
    if ORGANIC_CLASS_NAME not in names:
        print(f"Class '{ORGANIC_CLASS_NAME}' not found. Using original dataset YAML.")
        return source_yaml

    organic_id = names.index(ORGANIC_CLASS_NAME)
    kept_names = [name for index, name in enumerate(names) if index != organic_id]
    class_id_map = {
        old_id: new_id
        for new_id, old_id in enumerate(index for index in range(len(names)) if index != organic_id)
    }

    print(f"Building filtered dataset without '{ORGANIC_CLASS_NAME}'...")
    print(f"Filtered dataset directory: {output_dir}")

    stats: dict[str, dict[str, int]] = {}
    for yaml_split, output_split in SPLIT_DIRS.items():
        source_image_dir = resolve_split_dir(source_yaml, source_data, yaml_split)
        if source_image_dir is None:
            continue
        if not source_image_dir.exists():
            raise FileNotFoundError(f"Image directory for '{yaml_split}' was not found: {source_image_dir}")

        target_image_dir = output_dir / output_split / "images"
        target_label_dir = output_dir / output_split / "labels"
        split_stats = {
            "images_total": 0,
            "candidate_images_after_organic_filter": 0,
            "images_kept": 0,
            "images_dropped": 0,
            "labels_kept": 0,
            "organic_labels_removed": 0,
            "groups": 0,
            "sampled_augmented_images_skipped": 0,
        }
        candidates: list[tuple[Path, Path, list[str]]] = []

        for image_path in sorted(source_image_dir.iterdir()):
            if not image_path.is_file():
                continue
            source_label = image_to_label_path(image_path, source_image_dir)
            if not source_label.exists():
                continue

            split_stats["images_total"] += 1
            kept_label_lines: list[str] = []
            removed_organic = 0
            for raw_line in source_label.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                parts = line.split()
                old_class_id = int(float(parts[0]))
                if old_class_id == organic_id:
                    removed_organic += 1
                    continue
                parts[0] = str(class_id_map[old_class_id])
                kept_label_lines.append(" ".join(parts))

            split_stats["organic_labels_removed"] += removed_organic
            if not kept_label_lines:
                split_stats["images_dropped"] += 1
                continue

            candidates.append((image_path, source_label, kept_label_lines))

        split_stats["candidate_images_after_organic_filter"] = len(candidates)
        if yaml_split == "train" and sample_train_one_per_group:
            selected_candidates: list[tuple[Path, Path, list[str]]] = []
            seen_groups: set[str] = set()
            for candidate in candidates:
                image_path = candidate[0]
                group_key = roboflow_group_key(image_path)
                if group_key in seen_groups:
                    split_stats["sampled_augmented_images_skipped"] += 1
                    continue
                selected_candidates.append(candidate)
                seen_groups.add(group_key)
            split_stats["groups"] = len(selected_candidates)
            print(
                f"train groups={split_stats['groups']}, "
                f"kept={len(selected_candidates)}, "
                f"skipped_augmented={split_stats['sampled_augmented_images_skipped']}"
            )
        else:
            selected_candidates = candidates

        for image_path, source_label, kept_label_lines in selected_candidates:
            target_image = target_image_dir / image_path.name
            target_label = target_label_dir / source_label.name
            link_or_copy_file(image_path, target_image)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            target_label.write_text("\n".join(kept_label_lines) + "\n", encoding="utf-8")

            split_stats["images_kept"] += 1
            split_stats["labels_kept"] += len(kept_label_lines)

        stats[output_split] = split_stats
        print(
            f"{output_split}: kept {split_stats['images_kept']}/{split_stats['images_total']} images, "
            f"dropped {split_stats['images_dropped']}, "
            f"removed {split_stats['organic_labels_removed']} Organic labels"
        )

    filtered_data = {
        "path": str(output_dir.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(kept_names),
        "names": kept_names,
        "source_data_yaml": str(source_yaml),
        "removed_class": ORGANIC_CLASS_NAME,
        "sample_train_one_per_group": sample_train_one_per_group,
        "filter_stats": stats,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered_yaml.write_text(
        yaml_module.safe_dump(filtered_data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"Filtered data.yaml: {filtered_yaml}")
    return filtered_yaml.resolve()


def resolve_batch(batch_value: str, cuda_available: bool) -> int | float:
    value = str(batch_value).strip().lower()
    if value == "auto":
        if cuda_available:
            print("Batch: auto - using Ultralytics AutoBatch at 70% GPU memory")
            return 0.7
        print("Batch: auto requested, but CUDA is not available - using batch=16")
        return 16
    try:
        parsed = int(value)
    except ValueError:
        try:
            parsed_float = float(value)
        except ValueError as exc:
            raise ValueError('--batch must be "auto", an integer, or a GPU memory fraction such as 0.7') from exc
        if parsed_float <= 0:
            raise ValueError("--batch must be greater than 0")
        return parsed_float
    if parsed <= 0:
        raise ValueError("--batch must be greater than 0")
    return parsed


def resolve_resume_checkpoint(resume_arg: str | None, project_dir: Path, run_name: str) -> str | None:
    if resume_arg is None:
        return None

    resume_path = project_dir / run_name / "weights" / "last.pt" if resume_arg == "auto" else Path(resume_arg)
    if not resume_path.is_absolute():
        resume_path = (SCRIPT_DIR / resume_path).resolve()
    if not resume_path.exists():
        print(f"[ERROR] Resume checkpoint not found: {resume_path}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Resume checkpoint: {resume_path}")
    return str(resume_path)


def count_completed_epochs(results_csv: Path) -> int:
    if not results_csv.exists():
        print(f"[ERROR] Previous run results.csv not found: {results_csv}", file=sys.stderr)
        raise SystemExit(1)

    try:
        with results_csv.open(newline="", encoding="utf-8") as csv_file:
            rows = [row for row in csv.DictReader(csv_file) if row.get("epoch")]
    except OSError as exc:
        print(f"[ERROR] Could not read previous run results.csv: {results_csv}", file=sys.stderr)
        raise SystemExit(1) from exc

    if not rows:
        print(f"[ERROR] Previous run has no completed epoch rows: {results_csv}", file=sys.stderr)
        raise SystemExit(1)
    return len(rows)


def resolve_continue_training(args: argparse.Namespace, project_dir: Path) -> tuple[str, int] | None:
    if not args.continue_from_run:
        return None

    if args.resume is not None:
        print("[ERROR] Use either --resume or --continue-from-run, not both.", file=sys.stderr)
        raise SystemExit(1)

    if args.continue_to_epochs <= 0:
        print("[ERROR] --continue-to-epochs must be greater than 0.", file=sys.stderr)
        raise SystemExit(1)

    source_run_dir = project_dir / args.continue_from_run
    if not source_run_dir.exists():
        print(f"[ERROR] Previous run directory not found: {source_run_dir}", file=sys.stderr)
        raise SystemExit(1)

    completed_epochs = count_completed_epochs(source_run_dir / "results.csv")
    if args.continue_to_epochs <= completed_epochs:
        print(
            f"[ERROR] --continue-to-epochs ({args.continue_to_epochs}) must be greater than "
            f"completed epochs ({completed_epochs}).",
            file=sys.stderr,
        )
        raise SystemExit(1)

    checkpoint_path = source_run_dir / "weights" / f"{args.continue_weights}.pt"
    if not checkpoint_path.exists():
        print(f"[ERROR] Continue checkpoint not found: {checkpoint_path}", file=sys.stderr)
        raise SystemExit(1)

    if not args.name_was_set:
        args.name = f"{args.continue_from_run}_continue_to{args.continue_to_epochs}"

    additional_epochs = args.continue_to_epochs - completed_epochs
    print("Continue training mode:")
    print(f"  Previous run: {source_run_dir}")
    print(f"  Completed epochs: {completed_epochs}")
    print(f"  Target total epochs: {args.continue_to_epochs}")
    print(f"  Additional epochs this run: {additional_epochs}")
    print(f"  Starting weights: {checkpoint_path}")
    print(f"  New output directory: {project_dir / args.name}")
    return str(checkpoint_path.resolve()), additional_epochs


def resolve_augmentation_args(args: argparse.Namespace) -> tuple[str, dict[str, float]]:
    mode = args.augmentation
    if mode == "auto":
        mode = "light" if args.exclude_organic and args.sample_train_one_per_group else "none"

    if mode == "light":
        return mode, {
            "degrees": 5.0,
            "translate": 0.05,
            "scale": 0.2,
            "fliplr": 0.5,
        }

    return mode, {
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "fliplr": 0.0,
    }


def metric_value(metrics: Any, keys: tuple[str, ...], attr_name: str | None = None) -> float | None:
    """Read a metric across Ultralytics versions, which expose metrics slightly differently."""
    if attr_name and hasattr(metrics, "box") and hasattr(metrics.box, attr_name):
        value = getattr(metrics.box, attr_name)
        return float(value() if callable(value) else value)

    results_dict = getattr(metrics, "results_dict", {}) or {}
    for key in keys:
        if key in results_dict:
            return float(results_dict[key])
    return None


def print_metric(label: str, value: float | None) -> None:
    if value is None:
        print(f"{label}: unavailable")
    else:
        print(f"{label}: {value:.4f}")


def main() -> None:
    args = parse_args()
    data_yaml = validate_data_yaml(args.data)
    torch, YOLO, yaml = require_dependencies()

    cuda_available = torch.cuda.is_available()
    device = 0 if cuda_available else "cpu"
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: available - using CUDA device 0 ({gpu_name})")
    else:
        print("GPU: not available - using CPU")

    if args.exclude_organic:
        filtered_dataset_dir = SAMPLED_FILTERED_DATASET_DIR if args.sample_train_one_per_group else FILTERED_DATASET_DIR
        train_data_yaml = build_filtered_dataset(
            data_yaml,
            filtered_dataset_dir,
            yaml,
            args.rebuild_filtered_data,
            args.sample_train_one_per_group,
        )
    else:
        train_data_yaml = data_yaml
        if args.rebuild_filtered_data:
            print("--rebuild-filtered-data ignored because --exclude-organic is not enabled.")

    batch = resolve_batch(args.batch, cuda_available)
    project_dir = SCRIPT_DIR / "runs" / "detect"
    continue_training = resolve_continue_training(args, project_dir)
    if continue_training:
        model_path, train_epochs = continue_training
        resume_checkpoint = None
    else:
        resume_checkpoint = resolve_resume_checkpoint(args.resume, project_dir, args.name)
        model_path = resume_checkpoint or args.model
        train_epochs = args.epochs

    print(f"Dataset mode: {'Organic excluded' if args.exclude_organic else 'original full dataset'}")
    print(f"Source dataset YAML: {data_yaml}")
    print(f"Training dataset YAML: {train_data_yaml}")
    print(f"Model: {model_path}")
    print(f"Output directory: {project_dir / args.name}")
    print(f"Epochs this run: {train_epochs}")
    print(f"Batch: {batch}")

    model = YOLO(model_path)

    augmentation_mode, augmentation_args = resolve_augmentation_args(args)
    print(f"Augmentation: {augmentation_mode}")

    # Roboflow already generated augmented images. Use light online augmentation
    # only when explicitly requested or when using the reduced 1-of-3 train split.
    train_results = model.train(
        data=str(train_data_yaml),
        epochs=train_epochs,
        imgsz=args.imgsz,
        batch=batch,
        project=str(project_dir),
        name=args.name,
        exist_ok=True,
        device=device,
        resume=bool(resume_checkpoint),
        save_period=1,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        degrees=augmentation_args["degrees"],
        translate=augmentation_args["translate"],
        scale=augmentation_args["scale"],
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=augmentation_args["fliplr"],
    )

    save_dir = Path(getattr(getattr(model, "trainer", None), "save_dir", project_dir / args.name))
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        print(f"[ERROR] Training finished, but best.pt was not found: {best_pt}", file=sys.stderr)
        raise SystemExit(1)

    print(f"\nBest weights: {best_pt.resolve()}")

    print("\nRunning validation with best.pt...")
    best_model = YOLO(str(best_pt))
    val_batch = getattr(getattr(model, "trainer", None), "batch_size", batch)
    metrics = best_model.val(data=str(train_data_yaml), imgsz=args.imgsz, batch=val_batch, device=device)

    print("\nValidation metrics")
    print_metric("Precision", metric_value(metrics, ("metrics/precision(B)", "metrics/precision"), "mp"))
    print_metric("Recall", metric_value(metrics, ("metrics/recall(B)", "metrics/recall"), "mr"))
    print_metric("mAP50", metric_value(metrics, ("metrics/mAP50(B)", "metrics/mAP50"), "map50"))
    print_metric("mAP50-95", metric_value(metrics, ("metrics/mAP50-95(B)", "metrics/mAP50-95"), "map"))

    print(f"\nUse this file on Raspberry Pi for local inference:\n{best_pt.resolve()}")


if __name__ == "__main__":
    main()
