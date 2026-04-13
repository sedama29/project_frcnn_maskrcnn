import argparse
from pathlib import Path

import torch
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn

from train_common import build_loaders, run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Mask R-CNN on COCO subset.")
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/maskrcnn"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, val_ds, train_loader, val_loader = build_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        include_masks=True,
    )

    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_ds=val_ds,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        output_dir=args.output_dir,
        iou_types=["bbox", "segm"],
    )


if __name__ == "__main__":
    main()
