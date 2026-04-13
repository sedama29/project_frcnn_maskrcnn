import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from pycocotools.coco import COCO
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.ops import box_iou
from torchvision.transforms import functional as F
from PIL import Image


def xywh_to_xyxy(boxes_xywh: List[List[float]]) -> torch.Tensor:
    if not boxes_xywh:
        return torch.zeros((0, 4), dtype=torch.float32)
    t = torch.tensor(boxes_xywh, dtype=torch.float32)
    t[:, 2] = t[:, 0] + t[:, 2]
    t[:, 3] = t[:, 1] + t[:, 3]
    return t


def compute_recall(
    proposals: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_thresholds: List[float],
    topk_values: List[int],
) -> Dict[str, float]:
    recalls = {}
    if gt_boxes.numel() == 0:
        for k in topk_values:
            for thr in iou_thresholds:
                recalls[f"recall@{k}_iou{thr}"] = 1.0
        return recalls

    for k in topk_values:
        p = proposals[:k]
        if p.numel() == 0:
            for thr in iou_thresholds:
                recalls[f"recall@{k}_iou{thr}"] = 0.0
            continue
        ious = box_iou(gt_boxes, p)
        max_iou_per_gt, _ = ious.max(dim=1)
        for thr in iou_thresholds:
            recalls[f"recall@{k}_iou{thr}"] = float((max_iou_per_gt >= thr).float().mean().item())
    return recalls


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze RPN proposal recall on COCO subset.")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument(
        "--ann_json",
        type=Path,
        default=None,
        help="COCO instances JSON (default: <data_root>/annotations/instances_val2017_subset.json)",
    )
    parser.add_argument("--output_json", type=Path, default=Path("outputs/frcnn/rpn_recall.json"))
    parser.add_argument("--max_images", type=int, default=200)
    parser.add_argument("--topk", type=int, nargs="+", default=[100, 300, 1000])
    parser.add_argument("--iou", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    args = parser.parse_args()
    if args.ann_json is None:
        args.ann_json = args.data_root / "annotations" / "instances_val2017_subset.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device)
    model.eval()

    coco = COCO(str(args.ann_json))
    img_ids = coco.getImgIds()[: args.max_images]

    agg = {f"recall@{k}_iou{thr}": [] for k in args.topk for thr in args.iou}

    with torch.inference_mode():
        for img_id in img_ids:
            img_info = coco.loadImgs(img_id)[0]
            img_path = args.data_root / "val2017" / img_info["file_name"]
            image = Image.open(img_path).convert("RGB")
            image_t = F.to_tensor(image).to(device)

            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            gt_xywh = [a["bbox"] for a in anns if a["bbox"][2] > 1 and a["bbox"][3] > 1]
            gt_boxes = xywh_to_xyxy(gt_xywh).to(device)

            images, _ = model.transform([image_t], None)
            features = model.backbone(images.tensors)
            if isinstance(features, torch.Tensor):
                features = {"0": features}
            proposals, _ = model.rpn(images, features, None)
            props = proposals[0]

            rec = compute_recall(props, gt_boxes, args.iou, args.topk)
            for k, v in rec.items():
                agg[k].append(v)

    summary = {k: float(sum(v) / len(v)) if v else 0.0 for k, v in agg.items()}
    out = {
        "num_images": len(img_ids),
        "topk": args.topk,
        "iou_thresholds": args.iou,
        "mean_recall": summary,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Saved proposal recall analysis to {args.output_json}")
    for key in sorted(summary):
        print(f"{key}: {summary[key]:.4f}")


if __name__ == "__main__":
    main()
