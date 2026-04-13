# COCO Case Study 2 Starter

This project scaffold helps you complete:
- Faster R-CNN object detection on a COCO subset
- Mask R-CNN instance segmentation on the same subset
- Training, evaluation, and result logging for your report

## 1) Quick local setup (optional)

If you want to test on your own machine first:

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio pycocotools pillow matplotlib tqdm
```

## 2) Full cluster workflow (recommended)

This section is the end-to-end process for UTRGV/TACC-style clusters.

### 2.1 Create project folders on cluster

```bash
mkdir -p ~/project_frcnn_maskrcnn/{data,scripts,outputs,logs}
cd ~/project_frcnn_maskrcnn
```

### 2.2 Copy your code to cluster

From your local machine, copy this repository contents:

```bash
scp -r /path/to/COCO/* your_username@cluster_host:~/project_frcnn_maskrcnn/
```

If your cluster requires a gateway/jump host, use your institution's SSH instructions.

### 2.3 Create Python environment on cluster

```bash
module load python/3.10
python -m venv vision_env
source vision_env/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio pycocotools pillow matplotlib tqdm
```

### 2.4 Where to download COCO data

Download from official COCO 2017 URLs:
- `http://images.cocodataset.org/zips/train2017.zip`
- `http://images.cocodataset.org/zips/val2017.zip`
- `http://images.cocodataset.org/annotations/annotations_trainval2017.zip`

Download on cluster (if outbound internet is allowed):

```bash
cd ~/project_frcnn_maskrcnn/data
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

If cluster download is blocked, download locally and upload with `scp`.

### 2.5 Extract data on cluster

```bash
cd ~/project_frcnn_maskrcnn/data
mkdir -p coco
unzip -q train2017.zip -d coco
unzip -q val2017.zip -d coco
unzip -q annotations_trainval2017.zip -d coco
```

Expected structure:

```text
~/project_frcnn_maskrcnn/data/coco/
  train2017/
  val2017/
  annotations/
    instances_train2017.json
    instances_val2017.json
```

### 2.6 Build required subset (3000 train / 500 val)

```bash
cd ~/project_frcnn_maskrcnn
source vision_env/bin/activate
python scripts/build_coco_subset.py \
  --coco_root data/coco \
  --out_root data/coco_subset \
  --train_size 3000 \
  --val_size 500 \
  --seed 42
```

## 3) Run training on cluster with Slurm

### 3.1 Faster R-CNN Slurm script

Create `run_frcnn.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=frcnn_coco
#SBATCH --output=logs/frcnn_%j.out
#SBATCH --error=logs/frcnn_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

cd ~/project_frcnn_maskrcnn
module load python/3.10
source vision_env/bin/activate

python scripts/train_frcnn.py \
  --data_root data/coco_subset \
  --epochs 10 \
  --batch_size 2 \
  --lr 0.005 \
  --num_workers 2 \
  --output_dir outputs/frcnn
```

Submit and monitor:

```bash
sbatch run_frcnn.slurm
squeue -u your_username
tail -f logs/frcnn_JOBID.out
```

### 3.2 Mask R-CNN Slurm script

Create `run_maskrcnn.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=maskrcnn_coco
#SBATCH --output=logs/maskrcnn_%j.out
#SBATCH --error=logs/maskrcnn_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

cd ~/project_frcnn_maskrcnn
module load python/3.10
source vision_env/bin/activate

python scripts/train_maskrcnn.py \
  --data_root data/coco_subset \
  --epochs 10 \
  --batch_size 2 \
  --lr 0.005 \
  --num_workers 2 \
  --output_dir outputs/maskrcnn
```

Submit and monitor:

```bash
sbatch run_maskrcnn.slurm
squeue -u your_username
tail -f logs/maskrcnn_JOBID.out
```

## 4) Post-training analysis commands

### 4.1 Generate report figures

Faster R-CNN:

```bash
python scripts/visualize_predictions.py \
  --data_root data/coco_subset \
  --model_type frcnn \
  --checkpoint outputs/frcnn/best_model.pt \
  --output_dir outputs/figures/frcnn \
  --num_images 8 \
  --score_thresh 0.5
```

Mask R-CNN:

```bash
python scripts/visualize_predictions.py \
  --data_root data/coco_subset \
  --model_type maskrcnn \
  --checkpoint outputs/maskrcnn/best_model.pt \
  --output_dir outputs/figures/maskrcnn \
  --num_images 8 \
  --score_thresh 0.5
```

### 4.2 RPN proposal recall analysis (Part I-D)

```bash
python scripts/rpn_proposal_analysis.py \
  --data_root data/coco_subset \
  --ann_json data/coco_subset/annotations/instances_val2017_subset.json \
  --output_json outputs/frcnn/rpn_recall.json \
  --max_images 200 \
  --topk 100 300 1000 \
  --iou 0.5 0.7 0.9
```

This writes `outputs/frcnn/rpn_recall.json` for:
- Recall vs IoU threshold (0.5 / 0.7 / 0.9)
- Recall vs number of proposals (100 / 300 / 1000)

## 5) What gets saved

Each training script saves:
- `outputs/frcnn/metrics.json` or `outputs/maskrcnn/metrics.json`
- `best_model.pt` (best validation AP)
- `last_model.pt` (last epoch checkpoint)

## 6) Report checklist mapping to assignment

- Architecture explanation: backbone, FPN, RPN, anchors, RoI heads, RoI Align
- Math formulation: `Lcls + Lbox (+ Lmask)` and Smooth L1 motivation
- Quantitative tables: AP, AP50, AP75, APsmall/APmedium/APlarge
- Proposal analysis: recall vs IoU and recall vs number of proposals
- Qualitative examples: correct, FP, mislocalization, missed small objects
- Tradeoff analysis: training time, memory, speed, AP gains

## 7) Practical cluster tips

- Start with `--epochs 1` first to validate paths and logs.
- Use `batch_size 1` if GPU memory is tight.
- Save and keep all Slurm logs for reproducibility in your report.
- If `module load python/3.10` is unavailable, use the version your cluster provides.
