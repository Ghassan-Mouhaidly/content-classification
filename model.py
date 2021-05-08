import importlib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.layers import Dense, Lambda, Dropout, Activation, Flatten, Input

class MultiOutputModel():
    """
    """
    def _generic_layer(self, inputs):
        """
        """
        x = Dense(256)(inputs)
        x = Activation("relu")(x)
        x = Dense(128)(x)
        x = Activation("relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)        
        
        return x 
   
    def _generic_branch(self, base_model, structure):
        """
        """
        # x = Lambda(lambda c: tf.image.rgb_to_grayscale(c))(inputs)        
        x = self._generic_layer(base_model)
        x = Dense(structure[1])(x)
        x = Activation(structure[2], name=structure[0])(x)       
        
        return x
    
    def _base_model(self, base, input_tensor):
        """
        """
        feature_extractor = importlib.import_module("tensorflow.keras.applications").__getattribute__(base)

        base_model = feature_extractor(input_tensor=input_tensor, weights='imagenet', include_top=False)
        base_model.trainable = False

        return base_model
    
    def build_model(self, width, height, base, structures):
        """
        """
        input_tensor = Input(shape=(width, height, 3))
        
        base_model = self._base_model(base, input_tensor)
        
        x = base_model.output
        x = GlobalAveragePooling2D()(x)

        branches = []

        for key, structure in structures.items():
            branches.append(self._generic_branch(x, structure))
        
        model = Model(inputs=input_tensor,
                     outputs = branches,
                     name="content_classification")        
        
        return model

IM_WIDTH = 224
IM_HEIGHT = 224

structures = {0: ('genre', 3, 'sigmoid'), 1: ('rating', 1, 'linear')}

model = MultiOutputModel().build_model(IM_WIDTH, IM_HEIGHT, "InceptionV3", structures)
keras.utils.plot_model(model, show_shapes=True)
