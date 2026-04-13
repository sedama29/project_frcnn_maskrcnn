# COCO Case Study — Faster R-CNN & Mask R-CNN (UTRGV Cradle)

Train on a **COCO subset** using **`scripts/train_frcnn.py`** and **`scripts/train_maskrcnn.py`**.  
This README assumes you work on **Cradle** (`login.cradle.utrgv.edu`) with **Miniconda** (recommended by the course).

**Extra docs in this repo**

| File | What it is |
|------|------------|
| **`mask_rcnn_guide.txt`** | Professor’s HPC guide (Miniconda, **cu121** PyTorch, **`gpul40q`**, CUDA checks, NaN tips). Same practices; their `train.py` / Penn-Fudan names differ from this repo. |
| **`guidelines.txt`** | General UTRGV Slurm tips (`dos2unix`, `sbatch`, **`gpua30q`** example). |

**GPU partition:** use **`gpul40q`** or **`gpua30q`** depending on your account—edit **`#SBATCH -p`** in **`run_frcnn.slurm`** and **`run_maskrcnn.slurm`**.

---

## Fast path: you already have `data/` and `data/coco_subset/`

### 1) Layout check (must match)

Project root (example **`~/project_frcnn_maskrcnn`**) should look like:

```text
project_frcnn_maskrcnn/
  data/
    coco_subset/
      train2017/          ← many .jpg
      val2017/
      annotations/
        instances_train2017_subset.json
        instances_val2017_subset.json
  scripts/
  run_frcnn.slurm
  run_maskrcnn.slurm
```

Quick check on the server:

```bash
cd ~/project_frcnn_maskrcnn
ls data/coco_subset/train2017 | head
ls data/coco_subset/val2017 | head
ls data/coco_subset/annotations/
```

If that is correct, **you do not need to download COCO or run `build_coco_subset.py`**.

Training always uses **`--data_root data/coco_subset`**.

---

### 2) Log in

```bash
ssh YOUR_USERNAME@login.cradle.utrgv.edu
cd ~/project_frcnn_maskrcnn
```

---

### 3) Miniconda (install once if you do not have it)

Skip if **`~/miniconda3`** already exists.

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# accept license, default install path, say yes to conda init
source ~/.bashrc
conda --version
```

**Conda Terms of Service error?** (from professor guide)

```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

---

### 4) Create the **`maskrcnn`** environment (Python 3.10)

This matches **`mask_rcnn_guide.txt`** and the **`run_*.slurm`** scripts in this repo.

```bash
cd ~/project_frcnn_maskrcnn
conda create -n maskrcnn python=3.10 -y
source ~/.bashrc
conda activate maskrcnn
python --version
```

Expect **`Python 3.10.x`**.  
**Do not use** the login node’s default **`python3`** (often **3.6**) for this project.

---

### 5) Install PyTorch (CUDA 12.1 wheels) + project packages

On Cradle, **avoid** plain `pip install torch`—use the **cu121** index (see **`mask_rcnn_guide.txt`**):

```bash
conda activate maskrcnn
pip install --upgrade pip setuptools wheel
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
pip cache purge 2>/dev/null || true
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install matplotlib numpy pillow tqdm pycocotools
```

**Package name is `matplotlib`** (not `matplotli`).

---

### 6) Test imports (on the login node)

```bash
conda activate maskrcnn
python -c "import torch; print('torch', torch.__version__, 'CUDA build', torch.version.cuda)"
python -c "import torchvision; print('torchvision', torchvision.__version__)"
```

---

### 7) Check that the GPU works (pick what matches your access)

Some schools **do not** give you a GPU on the login node—you must use an interactive compute session (**§7B**).  
Others **explicitly allow GPU on the login node** for your class. If that is you, use **§7A** and you can **skip §7B**.

#### 7A — GPU check on the login node (if your admins said login has GPU)

```bash
module load cuda/12.3
nvidia-smi

source ~/.bashrc
conda activate maskrcnn
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

If **`CUDA available: True`** here, your environment is fine. For **long** training, **`sbatch`** (step 8) is still best when you can use it: it reserves a node cleanly and avoids hogging a shared login machine.

#### 7B — GPU check via interactive job (if login has no GPU / standard HPC)

Only if **`nvidia-smi`** on login fails or shows no GPU:

```bash
srun -p gpul40q --gres=gpu:1 --cpus-per-task=4 --mem=16G -t 01:00:00 --pty bash
# if that fails, try: -p gpua30q

module load cuda/12.3
nvidia-smi
source ~/.bashrc
conda activate maskrcnn
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"

python -c "import torch; print(torch.cuda.is_available())"


If **`CUDA available: False`**, fix PyTorch install (§10) before long training.

---

### 8) Submit training (Slurm)

The repo’s **`run_frcnn.slurm`** / **`run_maskrcnn.slurm`** use **`conda activate maskrcnn`**.

```bash
cd ~/project_frcnn_maskrcnn
conda activate maskrcnn
dos2unix run_frcnn.slurm run_maskrcnn.slurm 2>/dev/null || true
chmod +x run_frcnn.slurm run_maskrcnn.slurm
mkdir -p logs

sbatch run_frcnn.slurm
# later:
# sbatch run_maskrcnn.slurm
```

Monitor:

```bash
squeue -u $USER
tail -f logs/frcnn_JOBID.out
```

**Outputs:** `outputs/frcnn/` or `outputs/maskrcnn/` → `metrics.json`, `best_model.pt`, `last_model.pt`.

---

## If you only have full COCO under `data/coco/` (no subset yet)

1. You need **`data/coco/train2017`**, **`val2017`**, **`annotations/instances_*.json`** (see **`mask_rcnn_guide.txt`** / assignment for download—prefer compute node or `scp` for large zips).

2. Build the assignment subset:

```bash
cd ~/project_frcnn_maskrcnn
conda activate maskrcnn
python scripts/build_coco_subset.py \
  --coco_root data/coco \
  --out_root data/coco_subset \
  --train_size 3000 \
  --val_size 500 \
  --seed 42
```

Then continue from **§8**.

---

## After training (same env)

```bash
cd ~/project_frcnn_maskrcnn
conda activate maskrcnn

# Figures — Faster R-CNN
python scripts/visualize_predictions.py \
  --data_root data/coco_subset \
  --model_type frcnn \
  --checkpoint outputs/frcnn/best_model.pt \
  --output_dir outputs/figures/frcnn \
  --num_images 8 \
  --score_thresh 0.5

# Figures — Mask R-CNN
python scripts/visualize_predictions.py \
  --data_root data/coco_subset \
  --model_type maskrcnn \
  --checkpoint outputs/maskrcnn/best_model.pt \
  --output_dir outputs/figures/maskrcnn \
  --num_images 8 \
  --score_thresh 0.5

# RPN proposal recall (Part I-D)
python scripts/rpn_proposal_analysis.py \
  --data_root data/coco_subset \
  --ann_json data/coco_subset/annotations/instances_val2017_subset.json \
  --output_json outputs/frcnn/rpn_recall.json \
  --max_images 200 \
  --topk 100 300 1000 \
  --iou 0.5 0.7 0.9
```

### Export tables & checklist for the PDF (no LaTeX required)

After the steps above (figures + RPN optional but recommended), generate **Markdown / LaTeX snippets / CSV** and a **submission checklist**:

```bash
cd ~/project_frcnn_maskrcnn
conda activate maskrcnn
python scripts/export_report_tables.py
```

**Creates `outputs/report/`:**

| File | Use |
|------|-----|
| **`report_narrative_draft.md`** | **Full outline + draft prose** (abstract, intro, method, setup, results, RPN, analysis, conclusion). Auto-fills AP/epochs from metrics; **`[EDIT: ...]`** tags mark what you must personalize. **Edit this file** and merge with tables/figures into your final PDF. |
| **`report_tables.md`** | Copy tables into Word / Google Docs → export PDF |
| **`report_tables.tex`** | Paste tabular blocks into LaTeX / Overleaf |
| **`metrics_summary.csv`** | Excel-friendly numbers |
| **`rpn_recall_table.csv`** | RPN recall (only if `rpn_recall.json` exists) |
| **`REPORT_CHECKLIST.txt`** | Submission checklist + paths |

Default inputs: `outputs/frcnn/metrics.json`, `outputs/maskrcnn/metrics.json`, `outputs/frcnn/rpn_recall.json`. Override with `--frcnn_metrics`, `--maskrcnn_metrics`, `--rpn_json`, `--output_dir` if needed.

Re-run **`export_report_tables.py`** after training (or after new metrics) to refresh numbers; your **edited** narrative can stay in a **copy** if you do not want it overwritten—in that case save your version as e.g. `my_report.md` outside `outputs/report/` or disable overwriting by copying the draft aside first.

---

## Python version

| | |
|--|--|
| **Use on Cradle** | **Python 3.10** in conda env **`maskrcnn`** |
| **Minimum** | 3.8+ (3.10 recommended) |
| **Avoid** | Login-node **`python3`** at **3.6** for installs/training |

---

## 10) Troubleshooting

| Problem | What to do |
|--------|------------|
| **`Python.h` / `pycocotools` build error** | You are not in **3.10** conda env, or pip is compiling. Use **`conda activate maskrcnn`**, **`python --version`**, then **`pip install --upgrade pip setuptools wheel`** and reinstall **`pycocotools`**. Or: **`conda install -c conda-forge pycocotools`**. |
| **`torch.cuda.is_available()` False on GPU node** | Reinstall torch with **§5** **`cu121`** line; try **`module load cuda/12.3`** or **`cuda/11.6`** to match **`module avail cuda`**. |
| **Loss = NaN** | Lower **`--lr`** in the Slurm script (try **`0.0005`** or **`0.0001`**), **`--batch_size 1`**. See **`mask_rcnn_guide.txt` Part O**. |
| **`module load python/3.10` missing** | Normal on Cradle—**ignore**; use **Miniconda** only. |
| **Jobs pending forever** | Wrong partition; switch **`#SBATCH -p`** between **`gpul40q`** / **`gpua30q`** or ask the instructor. |

---

## 11) Report checklist (assignment)

- Architecture: backbone, FPN, RPN, anchors, RoI heads, RoI Align  
- Losses: `Lcls + Lbox (+ Lmask)`, Smooth L1  
- Tables: AP, AP50, AP75, APs / APm / APl  
- RPN recall analysis; qualitative figures; tradeoffs (time, memory, accuracy)  
- Slurm logs + how data/subset were produced  

---

## 12) Optional: local PC (not Cradle)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio pycocotools pillow matplotlib tqdm
```

---

## 13) Alternative: `venv` named `vision_env` (no conda)

If you prefer **`~/miniconda3/bin/python3 -m venv vision_env`**: create the venv, **`source vision_env/bin/activate`**, run the same **§5** pip commands, and edit **`run_*.slurm`** to use **`source vision_env/bin/activate`** instead of **`conda activate maskrcnn`**.
