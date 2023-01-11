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

def convert_torch_to_onnx(onnx_path, image_path, torch_path=None):
    """
    Coverts Pytorch model file to ONNX
    :param torch_path: Torch model path to load
    :param onnx_path: ONNX model path to save
    :param image_path: Path to test image to use in export progress
    """
    pytorch_model = get_torch_model()
    
    image, _, torch_image = get_example_input(image_path)

    torch.onnx.export(
        model = pytorch_model,
        args = torch_image,
        f = onnx_path,
        verbose = False,
        export_params=True,
        do_constant_folding = False,
        input_names = ['input'],
        opset_version = 10,
        output_names = ['output'])
    
onnx_model_path ='model/model.onnx'
image_path = "test.png"

convert_torch_to_onnx(onnx_model_path,image_path)