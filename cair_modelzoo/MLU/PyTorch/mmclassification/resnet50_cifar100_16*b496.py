model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet_CIFAR',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch'),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='MultiLabelLinearClsHead',
        num_classes=100,
        in_channels=2048,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0, use_soft=True)),
    train_cfg=dict(augments=[
        dict(type='BatchMixup', alpha=0.2, num_classes=100, prob=0.2),
        dict(type='BatchCutMix', alpha=1.0, num_classes=100, prob=0.8)
    ]))
sampler = dict(type='RepeatAugSampler')
dataset_type = 'CIFAR100'
img_norm_cfg = dict(
    mean=[129.304, 124.07, 112.434], std=[68.17, 65.392, 70.418], to_rgb=False)
train_pipeline = [
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(
        type='Normalize',
        mean=[129.304, 124.07, 112.434],
        std=[68.17, 65.392, 70.418],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(
        type='Normalize',
        mean=[129.304, 124.07, 112.434],
        std=[68.17, 65.392, 70.418],
        to_rgb=False),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=496,
    workers_per_gpu=12,
    train=dict(
        type='CIFAR100',
        data_prefix='data/cifar100',
        pipeline=[
            dict(type='RandomCrop', size=32, padding=4),
            dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
            dict(
                type='Normalize',
                mean=[129.304, 124.07, 112.434],
                std=[68.17, 65.392, 70.418],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='ToTensor', keys=['gt_label']),
            dict(type='Collect', keys=['img', 'gt_label'])
        ]),
    val=dict(
        type='CIFAR100',
        data_prefix='data/cifar100',
        pipeline=[
            dict(
                type='Normalize',
                mean=[129.304, 124.07, 112.434],
                std=[68.17, 65.392, 70.418],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        test_mode=True),
    test=dict(
        type='CIFAR100',
        data_prefix='data/cifar100',
        pipeline=[
            dict(
                type='Normalize',
                mean=[129.304, 124.07, 112.434],
                std=[68.17, 65.392, 70.418],
                to_rgb=False),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ],
        test_mode=True))
optimizer = dict(
    type='SGD', lr=0.3, momentum=0.99, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=2500,
    warmup_ratio=0.25)
runner = dict(type='EpochBasedRunner', max_epochs=300)
checkpoint_config = dict(interval=1)
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
fp16 = dict(loss_scale=512.0)
work_dir = './work_dirs/resnet50_8xb16_cifar100'
gpu_ids = range(0, 16)

