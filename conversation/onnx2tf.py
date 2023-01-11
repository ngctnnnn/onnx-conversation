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

def convert_onnx_to_tf(onnx_path, tf_path):
    """
    Converts ONNX model to TF 2.X saved file
    :param onnx_path: ONNX model path to load
    :param tf_path: TF model path to save
    """
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    tf_rep = prepare(onnx_model)  #Prepare TF representation
    tf_rep.export_graph(tf_path)  #Export the model
    
if __name__ == "__main__":
    onnx_path ="model/model.onnx"
    tf_path = "model/model_tf"

    convert_onnx_to_tf(onnx_path, tf_path)
    
    tf_model_path ="model/model_tf"
    image_test_path = "test.png"

    _, _, input_tensor = get_example_input(image_test_path) 

    model = tf.saved_model.load(tf_model_path)
    model.trainable = False

    out = model(**{'input':input_tensor})
    print(out)