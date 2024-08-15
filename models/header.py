import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Conv2D, MaxPool2D, Activation, SeparableConv2D
from tensorflow.keras.regularizers import l2

class HeadWrapper(Layer):
    """
    Merging all feature maps for detections.
    Inputs:
        first_conv layer = (batch_size, (layer_shape x aspect_ratios), last_dimension)
            shape => (32 x 32 x 4) = 4096
        second_conv layer = (batch_size, (layer_shape x aspect_ratios), last_dimension)
            shape => (16 x 16 x 6) = 1536
        extra1_2 = (batch_size, (layer_shape x aspect_ratios), last_dimension)
            shape => (8 x 8 x 6) = 384
        extra2_2 = (batch_size, (layer_shape x aspect_ratios), last_dimension)
            shape => (4 x 4 x 6) = 96
        extra3_2 = (batch_size, (layer_shape x aspect_ratios), last_dimension)
            shape => (2 x 2 x 4) = 16
        extra4_2 = (batch_size, (layer_shape x aspect_ratios), last_dimension)
            shape => (1 x 1 x 4) = 4
                                           Total = 6132 default box
    Outputs:
        merged_head = (batch_size, total_prior_boxes, last_dimension)
    """

    def __init__(self, last_dimension, **kwargs):
        super(HeadWrapper, self).__init__(**kwargs)
        self.last_dimension = last_dimension

    def get_config(self):
        config = super(HeadWrapper, self).get_config()
        config.update({"last_dimension": self.last_dimension})
        return config

    def call(self, inputs):
        last_dimension = self.last_dimension
        batch_size = tf.shape(inputs[0])[0]
        outputs = []
        for conv_layer in inputs:
            outputs.append(tf.reshape(conv_layer, (batch_size, -1, last_dimension)))
        return tf.concat(outputs, axis=1)

def get_head_from_outputs(hyper_params, outputs):
    """
    Generating SSDLite Bounding Box deltas and label heads.
    
    Inputs:
        hyper_params = dictionary
        outputs = list of ssd layers output to be used for prediction

    Outputs:
        pred_deltas = merged outputs for bbox delta head
        pred_labels = merged outputs for bbox label head
        
    """
    
    ## Setting up regularization hyperparameters for Training the model ##
    label_reg_factor = 5e-3
    box_reg_factor = 1e-3
    total_labels = hyper_params["total_labels"]
    
    # +1 for ratio 1
    len_aspect_ratios = [len(x) + 1 for x in hyper_params["aspect_ratios"]]
    labels_head = []
    boxes_head = []
    
    ## Stacking and Replacing Separable Depthwise Convolutional Block instead of Standard Convolution in Detection Head Layer ##
    for i, output in enumerate(outputs):
        aspect_ratio = len_aspect_ratios[i]
        
        labels_head.append(SeparableConv2D(filters = aspect_ratio * total_labels, kernel_size = (3, 3), padding = "same", name = f"{i+1}_separable_conv_label_output", depthwise_regularizer = l2(label_reg_factor), pointwise_regularizer = l2(label_reg_factor))(output))
        
        boxes_head.append(SeparableConv2D(filters = aspect_ratio * 4, kernel_size = (3, 3), padding = "same", name = f"{i+1}_separable_conv_boxes_output", depthwise_regularizer = l2(box_reg_factor), pointwise_regularizer = l2(box_reg_factor))(output))
    
    ## Custom Layer to wrap all the layers in Detection Head into a single layer ##
    pred_labels = HeadWrapper(total_labels, name="labels_head")(labels_head)
    pred_labels = Activation("softmax", name="conf")(pred_labels)
    
    pred_deltas = HeadWrapper(4, name="loc")(boxes_head)
    return pred_deltas, pred_labels