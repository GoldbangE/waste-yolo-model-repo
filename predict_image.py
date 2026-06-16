from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_INFO = REPO_DIR / "models" / "model_info.json"
DEFAULT_OUTPUT_DIR = REPO_DIR / "outputs"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict waste objects with the packaged YOLOv8n model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", required=True, help="Image file or folder to predict")
    parser.add_argument("--model-info", default=str(DEFAULT_MODEL_INFO), help="Path to model_info.json")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Folder for rendered result images")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=None, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=None, help="Inference image size")
    parser.add_argument("--fill-alpha", type=float, default=0.18, help="Bounding box fill opacity from 0.0 to 1.0")
    parser.add_argument("--open", action="store_true", help="Open saved result images with the default viewer")
    return parser.parse_args()


def require_dependencies() -> tuple[Any, Any]:
    try:
        import cv2
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        print(f"[ERROR] Missing package: {exc.name}", file=sys.stderr)
        print("Install dependencies first: python -m pip install -r requirements.txt", file=sys.stderr)
        raise SystemExit(1) from exc
    return cv2, YOLO


def load_model_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        print(f"[ERROR] model_info.json not found: {path}", file=sys.stderr)
        raise SystemExit(1)
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_model_path(info: dict[str, Any], info_path: Path) -> Path:
    weights_path = Path(info["model"]["weights_path"])
    if not weights_path.is_absolute():
        weights_path = info_path.parent.parent / weights_path
    if not weights_path.exists():
        print(f"[ERROR] Model weights not found: {weights_path}", file=sys.stderr)
        raise SystemExit(1)
    return weights_path.resolve()


def image_paths(source: Path) -> list[Path]:
    if not source.exists():
        print(f"[ERROR] Source not found: {source}", file=sys.stderr)
        raise SystemExit(1)
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            print(f"[ERROR] Unsupported image file: {source}", file=sys.stderr)
            raise SystemExit(1)
        return [source.resolve()]

    images = sorted(path.resolve() for path in source.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        print(f"[ERROR] No supported images found in folder: {source}", file=sys.stderr)
        raise SystemExit(1)
    return images


def class_names(info: dict[str, Any]) -> list[str]:
    names = info.get("classes", {}).get("names", [])
    return [str(name) for name in names]


def color_for_class(class_id: int) -> tuple[int, int, int]:
    colors = [
        (40, 160, 240),
        (80, 200, 120),
        (220, 120, 80),
        (180, 120, 220),
        (80, 180, 220),
        (230, 180, 60),
        (120, 170, 255),
        (100, 220, 220),
        (220, 90, 130),
        (150, 210, 80),
    ]
    return colors[class_id % len(colors)]


def draw_result(cv2: Any, image: Any, detections: list[dict[str, Any]], fill_alpha: float) -> Any:
    fill_alpha = max(0.0, min(fill_alpha, 1.0))
    overlay = image.copy()

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), det["color"], thickness=-1)

    rendered = cv2.addWeighted(overlay, fill_alpha, image, 1.0 - fill_alpha, 0)

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        label = f"{det['class_name']} {det['confidence']:.2f}"
        color = det["color"]
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, thickness=2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_y1 = max(0, y1 - text_h - baseline - 8)
        label_y2 = label_y1 + text_h + baseline + 8
        label_x2 = min(rendered.shape[1] - 1, x1 + text_w + 8)
        cv2.rectangle(rendered, (x1, label_y1), (label_x2, label_y2), color, thickness=-1)
        cv2.putText(
            rendered,
            label,
            (x1 + 4, label_y2 - baseline - 4),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return rendered


def detections_from_result(result: Any, names: list[str]) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        xyxy = [int(round(value)) for value in box.xyxy[0].tolist()]
        class_name = names[class_id] if 0 <= class_id < len(names) else str(class_id)
        detections.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "box": xyxy,
                "color": color_for_class(class_id),
            }
        )
    return detections


def print_result(image_path: Path, output_path: Path, detections: list[dict[str, Any]]) -> None:
    print(f"\nImage: {image_path}")
    if not detections:
        print("Detections: none")
    else:
        print("Detections:")
        print("  #  class_id  class_name                              conf   box_xyxy")
        for index, det in enumerate(detections, start=1):
            box_text = ", ".join(str(value) for value in det["box"])
            print(
                f"  {index:<2} {det['class_id']:<8} "
                f"{det['class_name'][:36]:<36} "
                f"{det['confidence']:.3f}  [{box_text}]"
            )
    print(f"Saved: {output_path}")


def main() -> None:
    args = parse_args()
    cv2, YOLO = require_dependencies()

    model_info_path = Path(args.model_info).expanduser().resolve()
    info = load_model_info(model_info_path)
    model_path = resolve_model_path(info, model_info_path)
    names = class_names(info)

    thresholds = info.get("thresholds", {})
    input_info = info.get("input", {})
    conf = args.conf if args.conf is not None else float(thresholds.get("default_confidence", 0.25))
    iou = args.iou if args.iou is not None else float(thresholds.get("default_iou", 0.7))
    imgsz = args.imgsz if args.imgsz is not None else int(input_info.get("image_size", 416))
    max_det = int(thresholds.get("max_detections", 300))

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))

    print(f"Model: {model_path}")
    print(f"Classes: {len(names)}")
    print(f"Confidence: {conf}")
    print(f"Output: {output_dir}")

    for path in image_paths(Path(args.source).expanduser()):
        image = cv2.imread(str(path))
        if image is None:
            print(f"[WARN] Could not read image: {path}")
            continue

        results = model.predict(source=str(path), imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, verbose=False)
        detections = detections_from_result(results[0], names)
        rendered = draw_result(cv2, image, detections, args.fill_alpha)

        output_path = output_dir / f"{path.stem}_pred{path.suffix}"
        cv2.imwrite(str(output_path), rendered)
        print_result(path, output_path, detections)

        if args.open:
            os.startfile(output_path)  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
