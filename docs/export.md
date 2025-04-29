# export

修改mmengine中的infer/infer.py文件

```
checkpoint: Optional[dict] = None
if weights is not None:
    checkpoint = _load_checkpoint(weights, map_location='cpu')

if not cfg:
    assert checkpoint is not None
    try:
        # Prefer to get config from `message_hub` since `message_hub`
        # is a more stable module to store all runtime information.
        # However, the early version of MMEngine will not save config
        # in `message_hub`, so we will try to load config from `meta`.
        cfg_string = checkpoint['message_hub']['runtime_info']['cfg']
    except KeyError:
        assert 'meta' in checkpoint, (
            'If model(config) is not provided, the checkpoint must'
            'contain the config string in `meta` or `message_hub`, '
            'but both `meta` and `message_hub` are not found in the '
            'checkpoint.')
        meta = checkpoint['meta']
        if 'cfg' in meta:
            cfg_string = meta['cfg']
        else:
            raise ValueError(
                'Cannot find the config in the checkpoint.')
    cfg.update(
        Config.fromstring(cfg_string, file_format='.py')._cfg_dict)

# Delete the `pretrained` field to prevent model from loading the
# the pretrained weights unnecessarily.
if cfg.model.get('pretrained') is not None:
    del cfg.model.pretrained

model = MODELS.build(cfg.model)
model.cfg = cfg
self._load_weights_to_model(model, checkpoint, cfg)
model.to(device)
model.eval()
print("开始导出onnx模型，用完后记得删除该代码")
import torch
torch.onnx.export(model, (torch.randn(1, 1, 32, 48, device=next(model.parameters()).device),), "crnn.onnx", input_names=["input"], dynamic_axes={'input' : {0 : 'batch_size', 3: "width"}})
return model
```