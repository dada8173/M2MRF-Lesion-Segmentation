_base_ = './fcn_hr48-M2MRF-C_40k_idrid_baseline_copy.py'

model = dict(
    decode_head=dict(
        loss_decode=dict(
            type='WeightedBinaryDiceLoss',
            loss_type='dice',
            class_weight=[1.0, 1.0, 1.5, 2.0],
            smooth=1e-5,
            loss_weight=1.0)))

work_dir = 'work_dirs/la_m2mrf/idrid_weighted_dice'
