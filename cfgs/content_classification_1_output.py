from .assets.default_configs import get_cfg_defaults

_C = get_cfg_defaults()

_C.MODEL.STRUCTURE = [{0: {'type': 'Genre', 'outNeurons': 6, 'outActivation': 'sigmoid', 'loss': 'categorical_crossentropy', 'weight': 1, 'metric': 'accuracy'}}]

_C.MODEL.BACKBONE = "InceptionV3"


_C.IMAGE.RESOLUTION = (200, 150, 3)

_C.HYPERPARAMS.TEST_SPLIT = 0.9
_C.HYPERPARAMS.VAL_SPLIT = 0.9
_C.HYPERPARAMS.LR = 1e-4
_C.HYPERPARAMS.EPOCHS = 30
_C.HYPERPARAMS.BATCH_SIZE_TR = 64
_C.HYPERPARAMS.BATCH_SIZE_VAL = 64
_C.HYPERPARAMS.BATCH_SIZE_TEST = 64

_C.CLASSES = [{'Genre': {0: 'Comedy', 1: 'Action', 2: 'Documentary', 3: 'Crime', 4: 'Animation', 5: 'Horror'}}]

def get_cfg():
    return _C.clone()
