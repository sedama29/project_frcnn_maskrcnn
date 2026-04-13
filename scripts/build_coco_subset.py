import argparse
import json
import random
import shutil
from pathlib import Path


def create_subset(annotation_file: Path, src_img_dir: Path, out_img_dir: Path, out_ann_file: Path, n_images: int, rng: random.Random) -> None:
    with annotation_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images = data["images"]
    annotations = data["annotations"]
    categories = data["categories"]

    if n_images > len(images):
        raise ValueError(f"Requested {n_images} images but only {len(images)} available in {annotation_file}")

    sampled_images = rng.sample(images, n_images)
    sampled_ids = {img["id"] for img in sampled_images}
    filtered_annotations = [ann for ann in annotations if ann["image_id"] in sampled_ids]

    out_img_dir.mkdir(parents=True, exist_ok=True)
    for img in sampled_images:
        src = src_img_dir / img["file_name"]
        dst = out_img_dir / img["file_name"]
        if not src.exists():
            continue
        shutil.copy2(src, dst)

    subset_data = {
        "images": sampled_images,
        "annotations": filtered_annotations,
        "categories": categories,
    }
    out_ann_file.parent.mkdir(parents=True, exist_ok=True)
    with out_ann_file.open("w", encoding="utf-8") as f:
        json.dump(subset_data, f)

    print(f"Saved subset: {out_ann_file}")
    print(f"  images: {len(sampled_images)}")
    print(f"  annotations: {len(filtered_annotations)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create sampled COCO train/val subsets with filtered annotation JSON.")
    parser.add_argument("--coco_root", type=Path, default=Path("coco"))
    parser.add_argument("--out_root", type=Path, default=Path("coco_subset"))
    parser.add_argument("--train_size", type=int, default=3000)
    parser.add_argument("--val_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    ann_dir = args.coco_root / "annotations"
    train_ann = ann_dir / "instances_train2017.json"
    val_ann = ann_dir / "instances_val2017.json"
    train_imgs = args.coco_root / "train2017"
    val_imgs = args.coco_root / "val2017"

    out_ann_dir = args.out_root / "annotations"
    out_train_ann = out_ann_dir / "instances_train2017_subset.json"
    out_val_ann = out_ann_dir / "instances_val2017_subset.json"

    create_subset(train_ann, train_imgs, args.out_root / "train2017", out_train_ann, args.train_size, rng)
    create_subset(val_ann, val_imgs, args.out_root / "val2017", out_val_ann, args.val_size, rng)
    print("COCO subset creation complete.")


if __name__ == "__main__":
    main()
