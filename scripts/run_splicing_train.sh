#!bash

## Train SpliceBERT on splicing task
python trainer.py --resume False --debug True --model_path 'model/pretrained/SpliceBERT/models/SpliceBERT.510nt' --patience 5 \
    --batch_size 6 --lr 1e-4 --output_dir 'model/fine_tuned/Splicing/SpliceBERT'

## Train DNABERT1(3mer) on splicing task
python trainer.py --method DNABERT1 --task splicing --resume False --debug True --model_path 'model/pretrained/DNABERT/DNABERT1' --patience 5 \
    --batch_size 6 --lr 1e-4 --output_dir 'model/fine_tuned/Splicing/DNABERT1'