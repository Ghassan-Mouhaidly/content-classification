import importlib

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Dropout, Activation, Input, Flatten

class MultiOutputModel(object):
    """
    """
    def __init__(self, width, height, channels, outputs):
        """
        """
        self.width = width
        self.width = height
        self.outputs = outputs
        self.model = None
        self.backbone = None
        self.optimizer = None
        self.input_tensor = Input(shape=(width, height, channels))

    def _generic_layer(self, inputs):
        """
        """
        x = Dense(128)(inputs)
        x = Activation("relu")(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        # x = BatchNormalization()(x)
        x = Dropout(0.2)(x)        
        
        return x 
   
    def _generic_branch(self, base_model, output):
        """
        """
        # x = Lambda(lambda c: tf.image.rgb_to_grayscale(c))(inputs)        
        x = self._generic_layer(base_model)
        x = Dense(output['outNeurons'])(x)
        x = Activation(output['outActivation'], name=output['type'])(x)       
        
        return x
    
    def _base_model(self, backbone, input_tensor):
        """
        """
        feature_extractor = importlib.import_module("tensorflow.keras.applications").__getattribute__(backbone)

        base_model = feature_extractor(input_tensor=input_tensor, weights='imagenet', include_top=False)
        base_model.trainable = False

        return base_model
    
    def build_model(self, backbone):
        """
        """
        self.backbone = backbone
        base_model = self._base_model(self.backbone, self.input_tensor)
        
        x = base_model.output
        # x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)

        branches = []

        for key, output in self.outputs.items():
            branches.append(self._generic_branch(x, output))
        
        self.model = Model(inputs=self.input_tensor,
                     outputs = branches,
                     name="content_classification")
    
    def compile_model(self, opt):
        """
        """
        self.optimizer = opt

        loss_dict = {}
        loss_weights_dict = {}
        metrics_dict = {}

        for key, output in self.outputs.items():
            loss_dict[output['type']] = output['loss']
            loss_weights_dict[output['type']] = output['weight']
            metrics_dict[output['type']] = output['metric']
 
        self.model.compile(optimizer=self.optimizer, 
              loss=loss_dict,
              loss_weights=loss_weights_dict,
              metrics=metrics_dict)
