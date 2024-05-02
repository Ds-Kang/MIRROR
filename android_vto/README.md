# MIRROR android application

## Overview
This is an application for the paper on Android.

### Model
We use onnxruntime format model by converting model trained with pytorch in this app.

## Requirements
- Android Studio 17.0.6
- Android SDK 33+
- Android NDK r21+
- Gradle 7.2
- OpenCV 4.5.5

## Build And Run (TODO)

### Step 0. Import OpenCV
1. Download [OpenCV 4.5.5](https://github.com/opencv/opencv/releases/download/4.5.5/opencv-4.5.5-android-sdk.zip).
2. Import module of `sdk` folder in downloaded folder with module name `:opencv`.
3. Add `opencv` as dependency of `app` module
   or add the following line to `build.gradle (Module :app)`
   ```
   dependencies {
       implementation project(path: ':opencv')
   }
   ```

### Step 1. Prepare NDK
1. Recommended version: 21.4.7075529
2. Add the following line to `local.properties`
   ```
   ndk.dir=/path/to/ndk
   ```

### Step 2. Prepare the ORT models
Download pretrained and converted onnxruntime model and locate it at `app/src/main/assets`.

[Parsing Model](https://drive.google.com/file/d/11vSIMje8ZZVVm4YGXHmNgwStOUoh89EQ/view?usp=sharing) 
[Warping Module](https://drive.google.com/file/d/1alssLq7uJOzgHnIpFUhyQp1ovVYcci0c/view?usp=sharing) 
[Generation Module](https://drive.google.com/file/d/1OFSx6-eTGstV0okGtJI_NC_zfq3Nm0Eh/view?usp=sharing)

Model names should be `parsing.ort`, `warp.ort`, `gen.ort` respectively.

### Step 3. Connect your Android Device and run the app

**Must allow storage permission to use the app**



