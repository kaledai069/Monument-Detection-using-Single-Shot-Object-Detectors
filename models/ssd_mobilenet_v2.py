import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from .header import get_head_from_outputs

def get_model(hyper_params):
    """Creating MobileNetV2-SSDLite with Tensorflow Functional API for Model Building
    
    Inputs:
        hyper_params = dictionary

    Outputs:
        ssd_model = tf.keras.model
        
    """
    ## Setting Hyperparameters required by the Model ##
    reg_factor = 4e-3
    img_size = hyper_params["img_size"]
    
    ## Loading pre-trained MobileNet-V2 model on Sinlge Monument Image Dataset ##
    pre_trained_model = tf.keras.models.load_model("/Saved_Model/mobilenetv2_512x512_fine-tuned.h5")
    base_model = pre_trained_model.layers[1]
    
    ## Input Layer ##
    input = base_model.input
    
    ## First Feature Map ##
    first_conv = base_model.get_layer("block_13_expand_relu").output
    
    ## Second Feature Map ##
    second_conv = base_model.output
    
    ## Third Feature Map ##
    extra1_1 = Conv2D(256, (1, 1), strides=(1, 1), padding="valid", activation="relu", name="extra1_1", kernel_regularizer = l2(reg_factor))(second_conv)
    extra1_2 = Conv2D(512, (3, 3), strides=(2, 2), padding="same", activation="relu", name="extra1_2", kernel_regularizer = l2(reg_factor))(extra1_1)
    
    ## Fourth Feature Map ##
    extra2_1 = Conv2D(128, (1, 1), strides=(1, 1), padding="valid", activation="relu", name="extra2_1", kernel_regularizer = l2(reg_factor))(extra1_2)
    extra2_2 = Conv2D(256, (3, 3), strides=(2, 2), padding="same", activation="relu", name="extra2_2", kernel_regularizer = l2(reg_factor))(extra2_1)
    
    ## Fifth Feature Map ##
    extra3_1 = Conv2D(128, (1, 1), strides=(1, 1), padding="valid", activation="relu", name="extra3_1", kernel_regularizer = l2(reg_factor))(extra2_2)
    extra3_2 = Conv2D(256, (3, 3), strides=(2, 2), padding="same", activation="relu", name="extra3_2", kernel_regularizer = l2(reg_factor))(extra3_1)
    
    ## Sixth Feature Map ##
    extra4_1 = Conv2D(128, (1, 1), strides=(1, 1), padding="valid", activation="relu", name="extra4_1", kernel_regularizer = l2(reg_factor))(extra3_2)
    extra4_2 = Conv2D(256, (3, 3), strides=(2, 2), padding="same", activation="relu", name="extra4_2", kernel_regularizer = l2(reg_factor))(extra4_1)

    ## Attaching 6-Feature Maps into a Single-Custom Detection Head Layer ##
    pred_deltas, pred_labels = get_head_from_outputs(hyper_params, [first_conv, second_conv, extra1_2, extra2_2, extra3_2, extra4_2])
    return Model(inputs=input, outputs=[pred_deltas, pred_labels])

def init_model(model):
    """
    Initializing model with dummy data for load weights with optimizer state and also graph construction.
    inputs:
        model = tf.keras.model
    """
    model(tf.random.uniform((1, 512, 512, 3)))