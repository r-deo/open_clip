cd ..
# python3 
torchrun --nproc_per_node 1 -m open_clip_train.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data="/data/train.csv"  \
    --val-data="/data/test.csv"  \
    --csv-img-key filepath \
    --csv-caption-key title \
    --warmup 2000 \
    --batch-size=128 \
    --lr=1e-6 \
    --wd=0.5 \
    --grad-clip-norm 50\
    --epochs=10 \
    --workers=8 \
    --model 'ViT-B-32'\
    --pretrained 'openai'\
    --accum-freq 10 \
    --log-every-n-steps 5\
    --train-num-samples 40000

    # --train-data="/home/ubuntu/geof/ITRA/pretrain/all_output_train.csv"  \
    # --val-data="/home/ubuntu/geof/ITRA/pretrain/all_output_val.csv"  \