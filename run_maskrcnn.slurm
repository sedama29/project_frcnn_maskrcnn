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
