import numpy as np

FIXED_PARAMS = dict(
    # data
    data_dir_train='/workspace/ImageNet1K/lmdb/train.lmdb',
    data_dir_val='/workspace/ImageNet1K/lmdb/val.lmdb',
    num_classes=1000,
    # model
    model='vit_pe_core_large_patch14_336.fb',
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
    warmup_epochs=0,
    keep_ckpt=0,
    # augmentation
    crop_pct=1.0,
    # others
    ema_start=0,
    ema_interval=1,
)

SEARCH_PARAMS = dict(
    # basic
    epochs=np.linspace(5, 15, 11, dtype=int),
    blr={
        4.6e-5: 0.2,
        6.8e-5: 0.3,
        1.0e-4: 0.3,
        1.5e-4: 0.2,
    },
    weight_decay=10 ** np.linspace(-3, -1, 20), # 20 values from 0.001 to 0.1
    drop_rate=[0.0, 0.01, 0.02, 0.05, 0.1, 0.2],
    # augmentation
    augment={
        'Trivial': 0.7,
        'Rand:num_ops=2:magnitude=9': 0.1,
        'Rand:num_ops=2:magnitude=15': 0.1,
        'Rand:num_ops=2:magnitude=20': 0.1,
    },
    rand_crop_scale={
        (1.0, 1.0): 0.3,
        (0.9, 1.0): 0.3,
        (0.8, 1.0): 0.2,
        (0.7, 1.0): 0.2,
    },
    rand_crop_ratio={
        0.75: 0.5,
        0.95: 0.3,
        1.0: 0.2,
    },
    re_prob=np.linspace(0.0, 0.45, 10),
    smoothing=np.linspace(0.0, 0.25, 26),
    mixup={
        0.0: 0.5,
        0.1: 0.25,
        0.2: 0.25,
    },
    cutmix={
        0.0: 0.5,
        0.1: 0.25,
        0.2: 0.25,
    },
    ema_ratio=np.linspace(1e-4, 1e-3, 10),
)

HARDWARE_PARAMS = dict(
    RTX4090=dict(
        NUM_GPUS=8,
        NUM_PROC=1,
        batch_size=256,
        accum_iter=2,
        num_workers=8,
    ),
    A800=dict(
        NUM_GPUS=8,
        NUM_PROC=1,
        batch_size=512,
        accum_iter=1,
        num_workers=8,
    ),
)

TASK_PARAMS = dict(
    WORK_DIR='/mae',
    OUTPUT_ROOT='/mae/finetune/classify/output.in1k.tmp',
    OUTPUT_DIR='{OUTPUT_ROOT}/{MODEL_IDX}.{EXP_IDX}',
    MODEL_IDX=1301,
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
