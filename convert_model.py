#!/usr/bin/python3

import tensorflow as tf
from tensorflow import python as tf_python

######## Conversion #########

####### FP 32 ########
tf_model = tf.keras.models.load_model('/Users/bahareh/Desktop/crash_pattern_wrAven/OpenCL_StridedSlice/tflite_inference_tool/model_files/sample_model.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)

tflite_model_quantized = converter.convert()
tflite_model_quantized_file = 'sample_model_fp32.tflite'

with open(tflite_model_quantized_file, 'wb') as f:
    f.write(tflite_model_quantized)

####### FP 16 ########
converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model_quantized = converter.convert()
tflite_model_quantized_file = 'sample_model_fp16.tflite'

with open(tflite_model_quantized_file, 'wb') as f:
    f.write(tflite_model_quantized)
