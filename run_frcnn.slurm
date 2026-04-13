#!/bin/bash
# UTRGV Cradle — pick ONE partition: gpua30q (guidelines) or gpul40q (professor L40S)
#SBATCH --job-name=frcnn_coco
#SBATCH --output=logs/frcnn_%j.out
#SBATCH --error=logs/frcnn_%j.err
#SBATCH --time=24:00:00
#SBATCH -p gpua30q
# #SBATCH -p gpul40q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

cd ~/project_frcnn_maskrcnn || exit 1
module load cuda/12.3

source ~/.bashrc
conda activate maskrcnn
# If you use venv instead of conda, comment the two lines above and use:
# source vision_env/bin/activate

echo "Python: $(which python)"
python --version
nvidia-smi

python scripts/train_frcnn.py \
  --data_root data/coco_subset \
  --epochs 10 \
  --batch_size 2 \
  --lr 0.001 \
  --num_workers 2 \
  --output_dir outputs/frcnn
