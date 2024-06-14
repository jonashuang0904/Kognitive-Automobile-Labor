import pytest
import numpy as np
from numpy.testing import assert_allclose
from pathlib import Path
import rospkg

from matplotlib import pyplot as plt
import cv2 as cv

from kal2_perception.preprocessing import OpenVinoInferenceSession, OnnxInferenceSession, UnetPreprocessor

def get_model_path():
    rospack = rospkg.RosPack()
    package_path = Path(rospack.get_path('kal2_perception'))

    return package_path / "models/segmentation_model_v2.fixed_224_224_3.onnx"

def test_openvino_inference_session(benchmark):
    model_path = get_model_path()

    input_shape = (1, 224, 224, 3)
    session = OpenVinoInferenceSession(model_path=model_path, input_shape=input_shape, device='GPU')

    input_data = np.zeros(input_shape, dtype=np.uint8)
    input_data[0] = cv.line(input_data[0, :, :, :], (112, 0), (112, 224), (255, 255, 255), 10)
    input_data = input_data.astype(np.float32) / 255.0

    output_data = session.run(input_data)
    benchmark(session.run, input_data)

    assert output_data.shape == (1, 224, 224, 1), "Output shape is not correct."
    assert output_data[0, :, 112-5:112+5, 0].sum() > 0, "Line is not detected."
    assert output_data[0, :, :100, 0].sum() < 10, "Background is should not be classified as line."
    assert output_data[0, :, 120:, 0].sum() < 10, "Background is should not be classified as line."


def test_onnx_inference_session(benchmark):
    model_path = get_model_path()

    input_shape = (1, 224, 224, 3)
    session = OnnxInferenceSession(model_path=model_path, input_shape=input_shape)

    input_data = np.zeros(input_shape, dtype=np.uint8)
    input_data[0] = cv.line(input_data[0, :, :, :], (112, 0), (112, 224), (255, 255, 255), 10)
    input_data = input_data.astype(np.float32) / 255.0

    output_data = session.run(input_data)
    benchmark(session.run, input_data)

    assert output_data.shape == (1, 224, 224, 1), "Output shape is not correct."
    assert output_data[0, :, 112-5:112+5, 0].sum() > 0, "Line is not detected."
    assert output_data[0, :, :100, 0].sum() < 10, "Background is should not be classified as line."
    assert output_data[0, :, 120:, 0].sum() < 10, "Background is should not be classified as line."

def test_unet_preprocessor():
    model_path = get_model_path()

    preprocessor = UnetPreprocessor(model_path=model_path, runtime="onnx")

    input_data = np.zeros((720, 1280, 3), dtype=np.uint8)
    output_data = preprocessor.process(input_data, np.zeros((720, 1280, 1), dtype=np.uint8))

    assert output_data.shape == (720, 1280), "Output shape is not correct."

