_base_ = [
    '../_base_/models/pspnet_r50-d8_aspp.py',
    '../_base_/datasets/cityscapes_test.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_test.py'
]
model = dict(
    decode_head=dict(align_corners=True),
    auxiliary_head=dict(align_corners=True))
test_cfg = dict(mode='slide', crop_size=(300, 300), stride=(44, 44))
