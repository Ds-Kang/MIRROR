
#include <jni.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "models.h"
#include "imageUtils.h"

extern "C" JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_example_virtualtryon_core_TotalModel_newModels(JNIEnv *env, jobject);

extern "C" JNIEXPORT void JNICALL
Java_ai_onnxruntime_example_virtualtryon_core_TotalModel_parsingModelInit(JNIEnv *env, jobject, jlong modelsAddr, jobject mgr);

extern "C" JNIEXPORT void JNICALL
Java_ai_onnxruntime_example_virtualtryon_core_TotalModel_vtoModelInit(JNIEnv *env, jobject, jlong modelsAddr, jobject mgr);

extern "C" JNIEXPORT void JNICALL
Java_ai_onnxruntime_example_virtualtryon_core_TotalModel_deleteModels(JNIEnv *env, jobject, jlong modelsAddr);

extern "C" JNIEXPORT jbyteArray JNICALL
Java_ai_onnxruntime_example_virtualtryon_core_TotalModel_inferenceParsing(JNIEnv *env, jobject, jlong modelsAddr, jbyteArray input1);

extern "C" JNIEXPORT jobjectArray JNICALL
Java_ai_onnxruntime_example_virtualtryon_core_TotalModel_inferenceVTO(JNIEnv *env, jobject,
                                                                 jlong modelsAddr, jbyteArray input1,
                                                                 jbyteArray input2, jbyteArray input3,
                                                                 jbyteArray parsingJava, jintArray input4,
                                                                 jintArray newSizeJava, jintArray paddingsJava,
                                                                 jintArray corrSizeJava);