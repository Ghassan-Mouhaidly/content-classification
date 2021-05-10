from .assets.default_configs import get_cfg_defaults

_C = get_cfg_defaults()

_C.MODEL.STRUCTURE = [{0: {'type': 'Genre', 'outNeurons': 3, 'outActivation': 'sigmoid', 'loss': 'categorical_crossentropy', 'weight': 1, 'metric': 'accuracy'}, 
                      1: {'type': 'Rating', 'outNeurons': 1, 'outActivation': 'linear', 'loss': 'mse', 'weight': 1, 'metric': 'mae'},
                      2: {'type': 'Year', 'outNeurons': 1, 'outActivation': 'linear', 'loss': 'mse', 'weight': 1, 'metric': 'mae'}}]

_C.MODEL.BACKBONE = "InceptionV3"


_C.IMAGE.RESOLUTION = (224, 224, 3)

_C.HYPERPARAMS.TEST_SPLIT = 0.8
_C.HYPERPARAMS.VAL_SPLIT = 0.8
_C.HYPERPARAMS.LR = 1e-4
_C.HYPERPARAMS.EPOCHS = 100

_C.CLASSES = [{'Genre': {0: 'Comedy', 1: 'Drama', 2: 'Action', 3: 'Documentary', 4: 'Crime', 5: 'Animation', 6: 'Horror', 7:'Adventure'},
                        'Rating': {0: 'value'},
                        'Year': {0: 'value'}}]

def get_cfg():
    return _C.clone()
