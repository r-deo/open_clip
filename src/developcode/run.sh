cd ..
# python3 
torchrun --nproc_per_node 8 -m open_clip_train.main \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --report-to tensorboard \
    --train-data="/home/ubuntu/geof/ITRA/pretrain/all_output_train.csv"  \
    --val-data="/home/ubuntu/geof/ITRA/pretrain/all_output_val.csv"  \
    --csv-img-key filepath \
    --csv-caption-key title \
    --warmup 600 \
    --batch-size=288 \
    --lr=1e-6 \
    --wd=0.5 \
    --grad-clip-norm 50\
    --epochs=20 \
    --workers=8 \
    --model 'ViT-B-32'\
    --pretrained 'openai'\
    --accum-freq 10 \
    --log-every-n-steps 5
    # --train-num-samples 40000

    # --train-data="/home/ubuntu/geof/ITRA/pretrain/all_output_train.csv"  \
    # --val-data="/home/ubuntu/geof/ITRA/pretrain/all_output_val.csv"  \