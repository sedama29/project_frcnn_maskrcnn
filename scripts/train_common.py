import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F


class CocoDetectionDataset(Dataset):
    def __init__(self, image_dir: Path, ann_file: Path, include_masks: bool):
        self.image_dir = image_dir
        self.coco = COCO(str(ann_file))
        self.ids = sorted(self.coco.getImgIds())
        self.include_masks = include_masks

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        image = Image.open(self.image_dir / img_info["file_name"]).convert("RGB")
        image_tensor = F.to_tensor(image)

        boxes = []
        labels = []
        area = []
        iscrowd = []
        masks = []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 1 or h <= 1:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(a["category_id"])
            area.append(a.get("area", w * h))
            iscrowd.append(a.get("iscrowd", 0))
            if self.include_masks:
                masks.append(self.coco.annToMask(a))

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            area_t = torch.zeros((0,), dtype=torch.float32)
            crowd_t = torch.zeros((0,), dtype=torch.int64)
            masks_t = torch.zeros((0, image_tensor.shape[1], image_tensor.shape[2]), dtype=torch.uint8)
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            area_t = torch.tensor(area, dtype=torch.float32)
            crowd_t = torch.tensor(iscrowd, dtype=torch.int64)
            masks_t = torch.tensor(np.stack(masks, axis=0), dtype=torch.uint8) if self.include_masks else None

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([img_id]),
            "area": area_t,
            "iscrowd": crowd_t,
        }
        if self.include_masks:
            target["masks"] = masks_t

        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def train_one_epoch(model, loader, optimizer, device) -> Dict[str, float]:
    model.train()
    losses = []
    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return {"loss": float(np.mean(losses)) if losses else 0.0}


@torch.inference_mode()
def evaluate_coco(model, loader, dataset: CocoDetectionDataset, device, iou_type: str = "bbox") -> Dict[str, float]:
    model.eval()
    coco_gt = dataset.coco
    results = []

    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        for out, tgt in zip(outputs, targets):
            image_id = int(tgt["image_id"].item())
            boxes = out["boxes"].detach().cpu().numpy()
            scores = out["scores"].detach().cpu().numpy()
            labels = out["labels"].detach().cpu().numpy()

            if iou_type == "bbox":
                for b, s, l in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = b.tolist()
                    results.append(
                        {
                            "image_id": image_id,
                            "category_id": int(l),
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(s),
                        }
                    )
            else:
                masks = out["masks"].detach().cpu().numpy()
                for b, s, l, m in zip(boxes, scores, labels, masks):
                    x1, y1, x2, y2 = b.tolist()
                    bin_mask = (m[0] > 0.5).astype(np.uint8)
                    enc = mask_utils.encode(np.asfortranarray(bin_mask))
                    enc["counts"] = enc["counts"].decode("utf-8")
                    results.append(
                        {
                            "image_id": image_id,
                            "category_id": int(l),
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "segmentation": enc,
                            "score": float(s),
                        }
                    )

    if len(results) == 0:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "APs": 0.0, "APm": 0.0, "APl": 0.0}

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    return {
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "APs": float(stats[3]),
        "APm": float(stats[4]),
        "APl": float(stats[5]),
    }


def build_loaders(data_root: Path, batch_size: int, num_workers: int, include_masks: bool):
    train_ds = CocoDetectionDataset(
        image_dir=data_root / "train2017",
        ann_file=data_root / "annotations" / "instances_train2017_subset.json",
        include_masks=include_masks,
    )
    val_ds = CocoDetectionDataset(
        image_dir=data_root / "val2017",
        ann_file=data_root / "annotations" / "instances_val2017_subset.json",
        include_masks=include_masks,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return train_ds, val_ds, train_loader, val_loader


def run_training(model, train_loader, val_loader, val_ds, optimizer, scheduler, device, epochs: int, output_dir: Path, iou_types: List[str]):
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_key = "bbox_AP"
    best_score = -1.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()

        row = {"epoch": epoch, **train_metrics}
        for iou_type in iou_types:
            m = evaluate_coco(model, val_loader, val_ds, device, iou_type=iou_type)
            for k, v in m.items():
                row[f"{iou_type}_{k}"] = v

        row["epoch_time_sec"] = time.time() - t0
        history.append(row)

        if row.get(best_key, row.get("AP", -1.0)) > best_score:
            best_score = row.get(best_key, row.get("AP", -1.0))
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        torch.save(model.state_dict(), output_dir / "last_model.pt")

        with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        print(f"Epoch {epoch}: {row}")

    return history

