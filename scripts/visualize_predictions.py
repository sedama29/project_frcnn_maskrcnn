import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from pycocotools.coco import COCO
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    MaskRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
    maskrcnn_resnet50_fpn,
)
from torchvision.transforms import functional as F
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks


def load_model(model_type: str, ckpt: Path, device: torch.device):
    if model_type == "frcnn":
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    else:
        model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize detections/masks for report figures.")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument(
        "--ann_json",
        type=Path,
        default=None,
        help="COCO instances JSON (default: <data_root>/annotations/instances_val2017_subset.json)",
    )
    parser.add_argument("--model_type", choices=["frcnn", "maskrcnn"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--num_images", type=int, default=8)
    parser.add_argument("--score_thresh", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.ann_json is None:
        args.ann_json = args.data_root / "annotations" / "instances_val2017_subset.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_type, args.checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    coco = COCO(str(args.ann_json))
    img_ids = coco.getImgIds()
    rng = random.Random(args.seed)
    chosen = rng.sample(img_ids, min(args.num_images, len(img_ids)))

    for img_id in chosen:
        info = coco.loadImgs(img_id)[0]
        image_path = args.data_root / "val2017" / info["file_name"]
        pil_img = Image.open(image_path).convert("RGB")
        img_t = F.to_tensor(pil_img).to(device)

        with torch.inference_mode():
            out = model([img_t])[0]

        keep = out["scores"] >= args.score_thresh
        boxes = out["boxes"][keep].detach().cpu()
        labels = out["labels"][keep].detach().cpu()
        scores = out["scores"][keep].detach().cpu()

        img_u8 = (img_t.detach().cpu() * 255).to(torch.uint8)
        label_text = [f"{int(l)}:{float(s):.2f}" for l, s in zip(labels, scores)]
        drawn = draw_bounding_boxes(img_u8, boxes=boxes, labels=label_text, width=2)

        if args.model_type == "maskrcnn" and "masks" in out:
            masks = out["masks"][keep][:, 0] > 0.5
            if masks.shape[0] > 0:
                drawn = draw_segmentation_masks(drawn, masks=masks.detach().cpu(), alpha=0.35)

        fig = drawn.permute(1, 2, 0).numpy()
        save_path = args.output_dir / f"{args.model_type}_{img_id}.png"
        plt.figure(figsize=(10, 8))
        plt.axis("off")
        plt.imshow(fig)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
