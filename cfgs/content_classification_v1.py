from .assets.default_configs import get_cfg_defaults

_C = get_cfg_defaults()

_C.MODEL.STRUCTURE = [{0: {'type': 'genre', 'outNeurons': 3, 'outActivation': 'sigmoid', 'loss': 'categorical_crossentropy', 'weight': 1, 'metric': 'accuracy'}, 
                      1: {'type': 'rating', 'outNeurons': 1, 'outActivation': 'linear', 'loss': 'mse', 'weight': 1, 'metric': 'mae'}}]

_C.MODEL.BACKBONE = "InceptionV3"


_C.IMAGE.RESOLUTION = (224, 224, 3)

_C.HYPERPARAMS.TEST_SPLIT = 0.7
_C.HYPERPARAMS.LR = 1e-4
_C.HYPERPARAMS.EPOCHS = 100

_C.CLASSES.BRANCH_1 = ['Comedy', 'Drama', 'Action', 'Documentary', 'Crime', 'Animation', 'Horror', 'Adventure']
_C.CLASSES.BRANCH_2 = ['Rating']
_C.CLASSES.BRANCH_2 = ['Year']


def get_cfg():
    return _C.clone()
