_base_ = './fcn_hr48-M2MRF-C_40k_idrid_full.py'

runner = dict(type='IterBasedRunner', max_iters=2)
checkpoint_config = dict(by_epoch=False, interval=2)
evaluation = dict(interval=2, metric='mIoU')
log_config = dict(
    interval=1,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
    ])

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
)

work_dir = 'work_dirs/la_m2mrf/debug'
