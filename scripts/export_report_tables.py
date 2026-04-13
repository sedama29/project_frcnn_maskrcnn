"""
Build report-ready tables and summaries from training outputs.

Run after training (and after visualize_predictions.py + rpn_proposal_analysis.py if you use them).

Writes under --output_dir (default: outputs/report):
  - report_tables.md           Markdown tables (paste into Google Docs / convert to PDF)
  - report_narrative_draft.md  Editable intro / method / results / analysis / conclusion (you refine)
  - report_tables.tex          LaTeX tabular snippets
  - metrics_summary.csv        Flat CSV for Excel
  - rpn_recall_table.csv       If rpn_recall.json exists
  - REPORT_CHECKLIST.txt       Assignment-oriented checklist + file paths
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_metrics(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def best_row(history: List[Dict[str, Any]], key: str = "bbox_AP") -> Optional[Dict[str, Any]]:
    if not history:
        return None
    return max(history, key=lambda r: float(r.get(key, -1.0)))


def fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def bbox_keys(prefix: str = "bbox_") -> List[str]:
    return [f"{prefix}AP", f"{prefix}AP50", f"{prefix}AP75", f"{prefix}APs", f"{prefix}APm", f"{prefix}APl"]


def segm_keys() -> List[str]:
    return ["segm_AP", "segm_AP50", "segm_AP75", "segm_APs", "segm_APm", "segm_APl"]


def write_md(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_narrative_draft(
    best_f: Optional[Dict[str, Any]],
    best_m: Optional[Dict[str, Any]],
    hist_f: List[Dict[str, Any]],
    hist_m: List[Dict[str, Any]],
    rpn_exists: bool,
    out_dir: Path,
) -> str:
    """First-person report draft; numbers from metrics.json."""
    nf = len(hist_f)
    nm = len(hist_m)
    ap_f = fmt(best_f.get("bbox_AP")) if best_f else "[missing metrics]"
    ap_m_box = fmt(best_m.get("bbox_AP")) if best_m else "[missing metrics]"
    ap_m_mask = fmt(best_m.get("segm_AP")) if best_m and best_m.get("segm_AP") is not None else "[n/a]"

    lines: List[str] = []
    lines.append("# Report draft (first person)\n\n")
    lines.append(
        "_Auto-filled from `metrics.json` by `export_report_tables.py`. Trim or merge with `report_tables.md` for your PDF._\n\n"
    )

    lines.append("## Abstract\n\n")
    lines.append(
        f"I fine-tuned **Faster R-CNN** and **Mask R-CNN** (ResNet-50-FPN, torchvision, COCO-pretrained weights) "
        f"on a **3,000 / 500** COCO 2017 image subset. "
        f"At the best validation epoch by bounding-box AP, I obtained **bbox AP {ap_f}** (Faster R-CNN), "
        f"**bbox AP {ap_m_box}** and **mask AP {ap_m_mask}** (Mask R-CNN). "
        f"I compare detection and segmentation quality and discuss tradeoffs below.\n\n"
    )

    lines.append("## 1. Introduction\n\n")
    lines.append(
        "I focus on **object detection** and **instance segmentation** on COCO-style data. "
        "COCO is difficult because of scale variation, clutter, and overlap. "
        "I implement **Faster R-CNN** for boxes and **Mask R-CNN** for masks, and I analyze the **RPN**, **FPN**, "
        "and **RoI Align** (vs. RoI Pooling) as required by the assignment.\n\n"
    )

    lines.append("## 2. Methodology\n\n")
    lines.append("### 2.1 Faster R-CNN\n\n")
    lines.append(
        "- **Backbone:** ResNet-50; **FPN** gives a multi-scale pyramid.\n"
        "- **RPN:** Dense anchors per level; objectness + box regression to propose regions.\n"
        "- **Head:** Class scores and box refinements per RoI.\n"
        "- **Loss:** RPN + detection terms; box regression uses **Smooth L1** (common choice vs. pure L2 for stability).\n\n"
    )
    lines.append("### 2.2 Mask R-CNN\n\n")
    lines.append(
        "- Adds a **mask branch** (FCN on RoI features) for a binary mask per instance.\n"
        "- **RoI Align** avoids harsh quantization of RoI corners; masks align better at boundaries than with RoI Pooling alone.\n"
        "- **Loss:** \\(L = L_{\\mathrm{cls}} + L_{\\mathrm{box}} + L_{\\mathrm{mask}}\\) "
        "(mask branch: per-pixel sigmoid / BCE-style supervision in the torchvision implementation).\n\n"
    )

    lines.append("## 3. Experimental setup\n\n")
    lines.append(
        "- **Data:** I built a subset of **3,000 train** and **500 val** images with filtered JSON (`build_coco_subset.py`, default seed **42**).\n"
        "- **Models:** `fasterrcnn_resnet50_fpn` and `maskrcnn_resnet50_fpn` with **DEFAULT** weights; **SGD** (momentum 0.9, weight decay 5e-4), **StepLR** every 3 epochs (×0.1).\n"
        f"- **Epochs logged:** Faster R-CNN **{nf}**, Mask R-CNN **{nm}**.\n"
        "- **Cluster:** I trained on **UTRGV Cradle** via Slurm (`run_frcnn.slurm`, `run_maskrcnn.slurm`): conda env **`maskrcnn`**, Python 3.10, PyTorch **cu121** wheels, **`module load cuda/12.3`** in the job. Exact batch size, learning rate, and partition match those scripts and my Slurm output logs.\n"
        "- **Reproducibility:** The subset draw is seeded; training shuffle is not fully fixed unless a global seed is set in code.\n\n"
    )

    lines.append("## 4. Results\n\n")
    lines.append(
        "I report **COCO-style AP** (AP@[0.50:0.95], AP50, AP75, AP by size). "
        f"Full tables: **`{(out_dir / 'report_tables.md').as_posix()}`** and **`metrics_summary.csv`**.\n\n"
    )
    lines.append("### 4.1 Detection\n\n")
    lines.append(
        f"**Faster R-CNN** (epoch with best bbox AP): **{ap_f}** AP. "
        f"**Mask R-CNN** (same rule for the row I quote): **{ap_m_box}** AP. "
        "I use the detailed table for AP50, AP75, and small/medium/large breakdown and comment on where each model struggles (often small objects or crowding).\n\n"
    )
    lines.append("### 4.2 Segmentation (Mask R-CNN)\n\n")
    lines.append(
        f"**Mask AP** on the same quoted epoch: **{ap_m_mask}**. "
        "I relate this to box AP and to the qualitative masks in my figures.\n\n"
    )
    lines.append("### 4.3 Qualitative examples\n\n")
    lines.append(
        "I include figures from **`outputs/figures/frcnn/`** and **`outputs/figures/maskrcnn/`** "
        "(from `visualize_predictions.py`). I caption **good predictions**, **false positives**, "
        "**mislocalized boxes**, and **missed small objects** where they appear.\n\n"
    )

    lines.append("## 5. RPN proposal analysis\n\n")
    if rpn_exists:
        lines.append(
            f"I ran `rpn_proposal_analysis.py`; mean recall is in **`{(out_dir / 'rpn_recall_table.csv').as_posix()}`** "
            f"and **`report_tables.md` §6**. "
            "I summarize how recall drops at **higher IoU** and with **fewer top-k proposals**, and I connect that to **downstream AP**.\n\n"
        )
    else:
        lines.append(
            "_I did not generate `rpn_recall.json` yet; I will run `rpn_proposal_analysis.py` and re-export._\n\n"
        )

    lines.append("## 6. Analysis and tradeoffs\n\n")
    lines.append(
        "- **Compute:** Mask R-CNN costs more per step than Faster R-CNN because of the mask head and segmentation eval.\n"
        "- **Accuracy:** I weigh the **mask AP gain** against that cost using my tables and figures.\n"
        "- **Limits:** Subset size ≠ full COCO; I may tune LR or epochs further; CUDA nondeterminism can add small run-to-run noise.\n\n"
    )

    lines.append("## 7. Conclusion\n\n")
    lines.append(
        f"I trained Faster R-CNN and Mask R-CNN on a fixed COCO subset and reported **bbox AP {ap_f}** vs **{ap_m_box}** "
        f"and **mask AP {ap_m_mask}**. "
        "Future work could include longer training, stronger augmentation, or another backbone, on the same pipeline.\n\n"
    )

    lines.append(
        "---\n\n**Attachments:** Slurm `.out`/`.err`, `run_*.slurm`, code repo, and this PDF.\n"
    )
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export report tables from metrics.json and RPN JSON.")
    parser.add_argument("--frcnn_metrics", type=Path, default=Path("outputs/frcnn/metrics.json"))
    parser.add_argument("--maskrcnn_metrics", type=Path, default=Path("outputs/maskrcnn/metrics.json"))
    parser.add_argument("--rpn_json", type=Path, default=Path("outputs/frcnn/rpn_recall.json"))
    parser.add_argument("--figures_frcnn", type=Path, default=Path("outputs/figures/frcnn"))
    parser.add_argument("--figures_mask", type=Path, default=Path("outputs/figures/maskrcnn"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/report"))
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    hist_f = load_metrics(args.frcnn_metrics)
    hist_m = load_metrics(args.maskrcnn_metrics)

    last_f = hist_f[-1] if hist_f else None
    last_m = hist_m[-1] if hist_m else None
    best_f = best_row(hist_f, "bbox_AP")
    best_m = best_row(hist_m, "bbox_AP")

    labels = ["AP @[.50:.95]", "AP50", "AP75", "AP small", "AP medium", "AP large"]
    bkeys = bbox_keys()

    # --- Markdown ---
    lines: List[str] = []
    lines.append("# COCO subset — results for report\n")
    lines.append("Generated by `scripts/export_report_tables.py`.\n")

    lines.append("## 1) Faster R-CNN — bounding box metrics (best epoch by bbox AP)\n")
    if best_f:
        lines.append(f"- **Epoch:** {best_f.get('epoch', '')}  |  **train loss:** {fmt(best_f.get('loss'))}\n")
        lines.append("\n| Metric | Value |\n|--------|-------|\n")
        for lab, k in zip(labels, bkeys):
            lines.append(f"| {lab} | {fmt(best_f.get(k))} |\n")
    else:
        lines.append("_No data: missing `outputs/frcnn/metrics.json`._\n")

    lines.append("\n## 2) Mask R-CNN — bounding box metrics (best epoch by bbox AP)\n")
    if best_m:
        lines.append(f"- **Epoch:** {best_m.get('epoch', '')}  |  **train loss:** {fmt(best_m.get('loss'))}\n")
        lines.append("\n| Metric | Value |\n|--------|-------|\n")
        for lab, k in zip(labels, bkeys):
            lines.append(f"| {lab} | {fmt(best_m.get(k))} |\n")
    else:
        lines.append("_No data: missing `outputs/maskrcnn/metrics.json`._\n")

    lines.append("\n## 3) Mask R-CNN — mask / segmentation metrics (same epoch as §2 best bbox)\n")
    if best_m:
        sk = segm_keys()
        slabels = ["Mask AP @[.50:.95]", "Mask AP50", "Mask AP75", "Mask APs", "Mask APm", "Mask APl"]
        lines.append("\n| Metric | Value |\n|--------|-------|\n")
        for lab, k in zip(slabels, sk):
            lines.append(f"| {lab} | {fmt(best_m.get(k))} |\n")
    else:
        lines.append("_No Mask R-CNN metrics._\n")

    lines.append("\n## 4) Side-by-side comparison (best bbox epoch per model)\n")
    lines.append("\n| Metric | Faster R-CNN | Mask R-CNN (bbox) |\n|--------|--------------|-------------------|\n")
    for lab, k in zip(labels, bkeys):
        vf = fmt(best_f.get(k)) if best_f else "—"
        vm = fmt(best_m.get(k)) if best_m else "—"
        lines.append(f"| {lab} | {vf} | {vm} |\n")

    lines.append("\n## 5) Last epoch snapshot (for convergence discussion)\n")
    lines.append("\n| Model | Epoch | loss | bbox AP |\n|-------|-------|------|--------|\n")
    if last_f:
        lines.append(f"| Faster R-CNN | {last_f.get('epoch')} | {fmt(last_f.get('loss'))} | {fmt(last_f.get('bbox_AP'))} |\n")
    if last_m:
        lines.append(
            f"| Mask R-CNN | {last_m.get('epoch')} | {fmt(last_m.get('loss'))} | {fmt(last_m.get('bbox_AP'))} | "
            f"(segm AP {fmt(last_m.get('segm_AP'))})\n"
        )

    # RPN section
    if args.rpn_json.exists():
        with args.rpn_json.open(encoding="utf-8") as f:
            rpn = json.load(f)
        mr = rpn.get("mean_recall", {})
        lines.append("\n## 6) RPN proposal recall (mean over val images)\n")
        lines.append(f"- Images used: **{rpn.get('num_images', '?')}**\n")
        lines.append("\n| Setting | Mean recall |\n|---------|-------------|\n")
        for key in sorted(mr.keys()):
            lines.append(f"| `{key}` | {fmt(mr[key])} |\n")
    else:
        lines.append("\n## 6) RPN proposal recall\n")
        lines.append("_No `rpn_recall.json` — run:_\n")
        lines.append("```bash\npython scripts/rpn_proposal_analysis.py \\\n")
        lines.append("  --data_root data/coco_subset \\\n")
        lines.append("  --ann_json data/coco_subset/annotations/instances_val2017_subset.json \\\n")
        lines.append("  --output_json outputs/frcnn/rpn_recall.json\n```\n")

    # Figures
    lines.append("\n## 7) Qualitative figures (paths to include in PDF)\n")
    for label, p in [("Faster R-CNN", args.figures_frcnn), ("Mask R-CNN", args.figures_mask)]:
        pngs = sorted(p.glob("*.png")) if p.exists() else []
        lines.append(f"\n**{label}** (`{p}`): {len(pngs)} file(s)\n")
        for png in pngs[:20]:
            lines.append(f"- `{png.as_posix()}`\n")
        if len(pngs) > 20:
            lines.append(f"- _… and {len(pngs) - 20} more_\n")
        if not pngs:
            mt = "frcnn" if "frcnn" in p.as_posix() else "maskrcnn"
            ckpt = f"outputs/{mt}/best_model.pt"
            lines.append(
                f"_None — run:_ `python scripts/visualize_predictions.py --data_root data/coco_subset "
                f"--model_type {mt} --checkpoint {ckpt} --output_dir {p.as_posix()}`\n"
            )

    md_path = out_dir / "report_tables.md"
    write_md(md_path, "".join(lines))
    print(f"Wrote {md_path}")

    narrative_path = out_dir / "report_narrative_draft.md"
    write_md(
        narrative_path,
        build_narrative_draft(best_f, best_m, hist_f, hist_m, args.rpn_json.exists(), out_dir),
    )
    print(f"Wrote {narrative_path}")

    # --- LaTeX ---
    tex: List[str] = []
    tex.append("% --- Faster R-CNN (best bbox epoch) ---\n")
    if best_f:
        tex.append("\\begin{tabular}{lr}\n\\hline\n")
        for lab, k in zip(labels, bkeys):
            tex.append(f"{lab} & {fmt(best_f.get(k))} \\\\\n")
        tex.append("\\hline\n\\end{tabular}\n\n")

    tex.append("% --- Mask R-CNN bbox + mask (best bbox epoch) ---\n")
    if best_m:
        tex.append("\\begin{tabular}{lr}\n\\hline\n")
        tex.append("\\multicolumn{2}{c}{Bounding box} \\\\\n\\hline\n")
        for lab, k in zip(labels, bkeys):
            tex.append(f"{lab} & {fmt(best_m.get(k))} \\\\\n")
        tex.append("\\hline\n\\multicolumn{2}{c}{Instance segmentation} \\\\\n\\hline\n")
        for lab, k in zip(
            ["Mask AP", "Mask AP50", "Mask AP75", "Mask APs", "Mask APm", "Mask APl"],
            segm_keys(),
        ):
            tex.append(f"{lab} & {fmt(best_m.get(k))} \\\\\n")
        tex.append("\\hline\n\\end{tabular}\n")

    tex_path = out_dir / "report_tables.tex"
    tex_path.write_text("".join(tex), encoding="utf-8")
    print(f"Wrote {tex_path}")

    # --- CSV flat ---
    csv_path = out_dir / "metrics_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "epoch_type", "epoch", "loss"] + bkeys + segm_keys())
        for name, best, last in [
            ("faster_rcnn", best_f, last_f),
            ("mask_rcnn", best_m, last_m),
        ]:
            if best:
                row = [name, "best_bbox_ap", best.get("epoch"), best.get("loss")]
                row += [best.get(k) for k in bkeys]
                row += [best.get(k) for k in segm_keys()]
                w.writerow(row)
            if last and last is not best:
                row = [name, "last", last.get("epoch"), last.get("loss")]
                row += [last.get(k) for k in bkeys]
                row += [last.get(k) for k in segm_keys()]
                w.writerow(row)
    print(f"Wrote {csv_path}")

    # RPN CSV
    if args.rpn_json.exists():
        with args.rpn_json.open(encoding="utf-8") as f:
            rpn = json.load(f)
        rpn_csv = out_dir / "rpn_recall_table.csv"
        with rpn_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["metric", "mean_recall"])
            for key in sorted(rpn.get("mean_recall", {}).keys()):
                w.writerow([key, rpn["mean_recall"][key]])
        print(f"Wrote {rpn_csv}")

    # --- Checklist ---
    chk: List[str] = []
    chk.append("CASE STUDY 2 — REPORT CHECKLIST (align with course PDF)\n")
    chk.append("=" * 60 + "\n\n")
    chk.append("[ ] Abstract (max 1 page per assignment)\n")
    chk.append("[ ] Introduction / motivation\n")
    chk.append("[ ] Methodology (Faster R-CNN: backbone, FPN, RPN, anchors, RoI head)\n")
    chk.append("[ ] Methodology (Mask R-CNN: RoI Align, mask branch, L = Lcls+Lbox+Lmask)\n")
    chk.append("[ ] Experimental setup: hardware, conda env, PyTorch/CUDA build, Slurm partition\n")
    chk.append("[ ] Data: COCO subset 3000/500, seed, how subset was built (build_coco_subset.py)\n")
    chk.append("[ ] Hyperparameters: epochs, batch size, LR, optimizer (from Slurm / train scripts)\n")
    chk.append("[ ] Table: bbox AP, AP50, AP75, APs/m/l — Faster vs Mask (use metrics_summary.csv)\n")
    chk.append("[ ] Table: mask AP metrics — Mask R-CNN only\n")
    chk.append("[ ] RPN analysis: recall vs IoU and vs # proposals (rpn_recall_table.csv + discussion)\n")
    chk.append("[ ] Figures: qualitative examples (outputs/figures/...)\n")
    chk.append("[ ] Tradeoffs: time, memory, accuracy, mask branch cost\n")
    chk.append("[ ] Conclusion + limitations\n")
    chk.append("[ ] Attach: training logs (.out), Slurm scripts, code link or zip\n\n")
    chk.append("Generated files:\n")
    chk.append(f"  - {md_path.resolve()}\n")
    chk.append(f"  - {(out_dir / 'report_narrative_draft.md').resolve()}\n")
    chk.append(f"  - {tex_path.resolve()}\n")
    chk.append(f"  - {csv_path.resolve()}\n")
    if args.rpn_json.exists():
        chk.append(f"  - {(out_dir / 'rpn_recall_table.csv').resolve()}\n")
    chk.append("\nSource metrics:\n")
    chk.append(f"  - {args.frcnn_metrics}\n")
    chk.append(f"  - {args.maskrcnn_metrics}\n")

    chk_path = out_dir / "REPORT_CHECKLIST.txt"
    chk_path.write_text("".join(chk), encoding="utf-8")
    print(f"Wrote {chk_path}")

    print("\nNext: edit report_narrative_draft.md, copy report_tables.md + figures into your PDF.")


if __name__ == "__main__":
    main()
