import numpy as np

FIXED_PARAMS = dict(
    # data
    data_dir_train='/workspace/ImageNet1K/lmdb/train.lmdb',
    data_dir_val='/workspace/ImageNet1K/lmdb/val.lmdb',
    num_classes=1000,
    # model
    model='vit_pe_core_gigantic_patch14_448.fb',
    amp_dtype='bfloat16',
    partial_finetune=2,
    drop_path_rate=0.0,
    clip_grad=1.0,
    # training
    seed=0,
    dist_eval=1,
    partial_train=1,
    partial_eval=1,
    lr_scheduler='CosineAnnealing',
    lr_min_rate=0.0,
    lr_M_mult=0.9,
    lr_cosine_mul=0.1,
    stop_epoch=20,
    keep_ckpt=0,
    # augmentation
    augment='Trivial',
    crop_pct=1.0,
    rand_crop_ratio=0.75,
    # others
    ema_start=0,
    ema_interval=1,
    ema_ratio=0,
)

SEARCH_PARAMS = dict(
    # basic
    epochs=np.linspace(25, 50, 26, dtype=int),
    warmup_epochs=[0, 1, 5],
    blr={
        3.2e-6: 0.3,
        4.6e-6: 0.4,
        6.8e-6: 0.3,
    },
    weight_decay=2 ** np.linspace(0, 1, 11) * 0.004, # 11 values from 0.004 to 0.008
    drop_rate={
        0.0: 0.4,
        0.01: 0.6,
    },
    # augmentation
    rand_crop_scale={
        (1.0, 1.0): 1/3,
        (0.9, 1.0): 1/3,
        (0.7, 1.0): 1/3,
    },
    re_prob=np.linspace(0.15, 0.45, 7),
    smoothing={
        0.2: 0.4,
        0.15: 0.2,
        0.1: 0.4,
    },
    mixup={
        0.02: 0.3,
        0.05: 0.4,
        0.1: 0.3,
    },
    cutmix={
        0.0: 0.6,
        0.02: 0.4,
    },
)

HARDWARE_PARAMS = dict(
    RTX4090=dict(
        NUM_GPUS=8,
        NUM_PROC=1,
        batch_size=27,
        accum_iter=19,
        num_workers=8,
    ),
    A800=dict(
        NUM_GPUS=8,
        NUM_PROC=2,
        batch_size=256,
        accum_iter=1,
        num_workers=8,
    ),
)

TASK_PARAMS = dict(
    WORK_DIR='/mae',
    OUTPUT_ROOT='/mae/finetune/classify/output.in1k.tmp',
    OUTPUT_DIR='{OUTPUT_ROOT}/{MODEL_IDX}.{EXP_IDX}',
    MODEL_IDX=1311,
    EXP_IDX='0000',  # the initial index, will be set dynamically
    CUDA_DEVICES=0,  # will be set dynamically
    CPU_AFFINITY='0-15',  # will be set dynamically
    MASTER_PORT=29540,  # will be set dynamically
)

TRAIN_SCRIPT = '''#!/bin/bash
################################################
# Environment setup

cd $WORK_DIR
export HF_HOME='/workspace/cache/huggingface'
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

################################################
# Training task

mkdir -p $OUTPUT_DIR
taskset -c $CPU_AFFINITY torchrun --nnodes 1 --master_port $MASTER_PORT --nproc_per_node $NUM_PROC \\
    -m finetune.classify.main --task CLASSIFY $FIXED_PARAMS $HARDWARE_PARAMS $SEARCH_PARAMS \\
    --output_dir $OUTPUT_DIR --log_dir $OUTPUT_DIR/summary \\
    1>$OUTPUT_DIR/train.log 2>&1
'''
