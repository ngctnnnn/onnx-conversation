import os
import shutil
import sys

import cv2
import numpy as np 
import onnx
import torch
import tensorflow as tf 
from PIL import Image
from torchvision import transforms
from torchvision.models import *
from torchsummary import summary
from onnx_tf.backend import prepare

from utils.util import (
    get_torch_model,
    get_example_input
)

def convert_tf_to_tflite(tf_path, tf_lite_path):
    """
    Converts TF saved model into TFLite model
    :param tf_path: TF saved model path to load
    :param tf_lite_path: TFLite model path to save
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
    tflite_model  = converter.convert()
    with open(tf_lite_path, 'wb') as f:
        f.write(tflite_model)

tf_model_path = "model/model_tf"
tflite_model_path ="model/model.tflite"

convert_tf_to_tflite(tf_model_path, tflite_model_path)
tflite_model_path = 'model/model.tflite'
# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()

# Get input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# get_tensor() returns a copy of the tensor data
# use tensor() in order to get a pointer to the tensor
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)