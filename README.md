# openCL delegate issue with a sequence of Dense/FullyConnected nodes 

This repo contains scripts and a tool to reproduce the `openCL` delegate issue with a sequence of Dense/FullyConnected nodes. Our experiments revealed that if we use a sequence of Dense layers in a special pattern (see the following image), the corresponding tflite version of this model will generate a bunch of `nan` and `inf` values for certain random indices in certain runs. This issue happens with both FP16 and FP32 tflite versions. This issue can't be reproduced with the `XNNPACK` delegate.

<img width="153" alt="Screenshot 2023-01-16 at 2 43 50 PM" src="https://user-images.githubusercontent.com/45400368/212793054-8a85b2af-3a8b-47ee-9c90-9e8d58247f4a.png">

## Converting the model
* `model_files` folder contains the above-mentioned pattern (`sample_model.h5`) and its corresponding tflite versions (`sample_model_fp32.tflite`, and `sample_model_fp16.tflite`). 
  * You can also use `convert_model.py` to convert this pattern to tflite.
  
  Note: `sample_model.h5` is extracted from a large trained model.

## tflite_inference tool 
We have implemented a small tool to feed a random input to our sample tflite model using `openCL` and `XNNPACK` delegates. Run the tool multiple times. You will see that for some runs, there are `inf` values in the output from the `openCL` delegate. 

### PREREQUISITES: ###
* Linux or Mac host computer
* Connectivity to the target device via adb
* Android NDK, version 22 or later
* CMake 3.18 or later

### BUILD INSTRUCTIONS ###
* Unzip the `tensorflow_lite_cpp_2_10_1_patched_static.zip` file inside the `tflite_inference_tool` folder.
* In a terminal, from `tflite_inference_tool` folder:
```console
$ mkdir build
$ cd build
$ cmake -G "Unix Makefiles"
        -DCMAKE_SYSTEM_NAME=Android 
        -DANDROID_ABI=arm64-v8a 
        -DANDROID_STL=c++_shared 
        -DANDROID_NATIVE_API_LEVEL=27 
        -DCMAKE_VERBOSE_MAKEFILE=ON 
        -DCMAKE_TOOLCHAIN_FILE=<path-to-ndk>/build/cmake/android.toolchain.cmake 
        -DCMAKE_BUILD_TYPE=Release
        -DTensorFlowLite_ROOT=../tensorflow_lite_cpp_2_10_1_patched_static ..
$ make
```
* Here, you must replace <path-to-ndk> with the absolute path of the ndk installed on your computer. If you installed NDK through Android studio, it is typically located at:
    `/home/<username>/Android/Sdk/ndk/<version>/` on Linux

* `tensorflow_lite_cpp_2_10_1_patched_static` is TensorflowFlow Lite library (nightly version) package.
### Run INSTRUCTIONS ###
WARNING: This step will write to your `/data/local/tmp` folder on device. Please make sure existing files in that folder are backed up as needed.

In a terminal, from `tflite_inference_tool` folder:
```console
$ adb push ./build/model_test /data/local/tmp
$ adb push ./model_files /data/local/tmp
```

To run the tool you can use the FP32 or the FP16 versions. 
```console
$ adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=. ./model_test --model_a=model_files/sample_model_fp32.tflite --model_b=model_files/sample_model_fp32.tflite --input_shape=1,81 --output_shape=1,78"
```

The output should be something like this:
```console
INFO: Created TensorFlow Lite delegate for GPU.
INFO: Initialized TensorFlow Lite runtime.
VERBOSE: Replacing 13 node(s) with delegate (TfLiteGpuDelegateV2) node, yielding 1 partitions.
INFO: Initialized OpenCL-based API.
INFO: Created 1 GPU delegate kernels.
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
VERBOSE: Replacing 13 node(s) with delegate (TfLiteXNNPackDelegate) node, yielding 1 partitions.
OpenCL output:
inf, 34048, 6448, inf, -inf, -29936, 10336, -inf, -35360, inf, inf, inf, 40320, inf, inf, inf, inf, 39616, -13680, inf, 52160, inf, inf, inf, -12776, inf, 61440, -9648, -inf, -30272, -26624, -inf, -17568, -31776, -inf, -8600, 43872, -50752, 44416, -27328, 56512, -30656, inf, 14768, 31968, inf, -4944, inf, inf, 29536, 27376, 23056, 33024, 38880, 39232, 35968, 30656, 34112, 5168, 5176, 62944, 30608, inf, -10464, -28832, -43136, -41824, 33536, 37184, inf, inf, -11872, 16296, inf, 47808, inf, 30816, -35232, 
xnnpack output:
159092, 34167.3, 6507.01, 30145.7, -16468.5, -30071.8, 10657.3, -213111, -35455.5, 59236.4, 187745, 109088, 40159.7, 189144, 115387, 15455.3, 165877, 39716.5, -13815.6, 211712, 52249.9, 66429.5, 140783, 130572, -12844, 241188, 61518.1, -9514.39, -190658, -30316.6, -26623.5, -140136, -17629.3, -31893.8, -73718.8, -8607.2, 44218.3, -50711.9, 44557.1, -27426.1, 56461.4, -30854.3, 138278, 14813.5, 31880.9, 68724.8, -4587.04, 87265.3, 81735.9, 29943.9, 27338, 23023.4, 33413.6, 38988.4, 39238.6, 35762.5, 30767.5, 34308.8, 4932.98, 5111.61, 63130.2, 30683.3, 75698.6, -10461.2, -28918.4, -43359.6, -42086.3, 33322.9, 37274.8, 91469.1, 107788, -11963.6, 16518.4, 90442.4, 48030.6, 70459, 30802.7, -35454.6,  
```
### IMPORTANT UPDATE ###
We have noticed that sometimes the above-mentioned pattern does not lead into wrong results. Therefore, we think in addition to the pattern structure, the values of weight and bias are also influential factors. In `model_files/correct_results` you can find a pattern instance that does not have the issue of generating `inf` values. 

It is worth noting that both of the pattern instances (the one that leads into wrong `inf` values and the one that does not have this issue) are extracted from trained models.

