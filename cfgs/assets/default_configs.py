from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.IMAGE = CN()
_C.HYPERPARAMS = CN()
_C.CLASSES = CN()

def get_cfg_defaults():
    return _C.clone()
