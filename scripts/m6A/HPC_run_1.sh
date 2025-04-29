#!/bin/bash
#SBATCH --job-name=biom-m6-0.1
#SBATCH --nodes=1 
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --output=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_m6a101_4-0.1.out
#SBATCH --error=/public/home/shenninggroup/yny/code/biombenchmark/logs/%j_m6a101-4_0.1.err
#SBATCH --partition=gpu
#SBATCH -w, --nodelist=gpu02
#SBATCH --gres=gpu:1

cd /public/home/shenninggroup/yny/code/biombenchmark

echo "m6A-101bp-miclip-1e-4-0.1"
bash scripts/m6A/m6a_cls_batch.sh 1e-4 101bp 0.1

echo ""
echo "m6A-101bp-miclip-1e-5-0.1"
bash scripts/m6A/m6a_cls_batch.sh 1e-5 101bp 0.1

echo ""
echo "m6A-509bp-miclip-1e-4-0.1"
bash scripts/m6A/m6a_cls_batch.sh 1e-4 509bp 0.1

echo ""
echo "m6A-509bp-miclip-1e-5-0.1"
bash scripts/m6A/m6a_cls_batch.sh 1e-5 509bp 0.1

# Keep the job running until manually terminated
echo "Main task completed. Job will remain active until manually terminated."
tail -f /dev/null  # This command will wait indefinitely
