rubikmicrobe_textrecog_data_root = 'data/rubikmicrobe'

rubikmicrobe_textrecog_train = dict(
    type='RecogTextDataset',
    data_root=rubikmicrobe_textrecog_data_root,
    ann_file='textrecog_train.jsonl',
    pipeline=None)

# mjsynth_sub_textrecog_train = dict(
#     type='OCRDataset',
#     data_root=mjsynth_textrecog_data_root,
#     ann_file='subset_textrecog_train.json',
#     pipeline=None)
