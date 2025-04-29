# training schedule for 1x
_base_ = [
    '../_base_/datasets/simplesynth_martianmono_variablefont.py',
    '../_base_/datasets/simplesynth_bigshouldersinline.py',
    '../_base_/datasets/simplesynth_rubikmicrobe.py',
    '../_base_/datasets/simplesynth_wireone.py',
    '../_base_/datasets/svt.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_adadelta_5e.py',
    '_base_crnn_mini-vgg.py',
]
# dataset settings
train_list = [_base_.martianmono_variablefont_textrecog_train, _base_.bigshouldersinline_textrecog_train, _base_.rubikmicrobe_textrecog_train, _base_.wireone_textrecog_train]
# test_list = [
#     _base_.cute80_textrecog_test, _base_.iiit5k_textrecog_test,
#     _base_.svt_textrecog_test, _base_.svtp_textrecog_test,
#     _base_.icdar2013_textrecog_test, _base_.icdar2015_textrecog_test
# ]
# test_list = [_base_.svt_textrecog_test]
test_list = [_base_.martianmono_variablefont_textrecog_train]

default_hooks = dict(logger=dict(type='LoggerHook', interval=50), )
train_dataloader = dict(
    batch_size=64,
    num_workers=24,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=train_list,
        pipeline=_base_.train_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=test_list,
        pipeline=_base_.test_pipeline))
val_dataloader = test_dataloader

# val_evaluator = dict(
#     dataset_prefixes=['CUTE80', 'IIIT5K', 'SVT', 'SVTP', 'IC13', 'IC15'])
val_evaluator = dict(
    dataset_prefixes=['SVT'])
test_evaluator = val_evaluator

auto_scale_lr = dict(base_batch_size=64 * 4)
