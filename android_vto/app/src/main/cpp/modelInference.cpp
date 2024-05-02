
#include "modelInference.h"

extern "C" JNIEXPORT jlong JNICALL
Java_ai_onnxruntime_example_virtualtryon_core_TotalModel_newModels(JNIEnv *env, jobject) {
    TotalModel *totalModel = new TotalModel();
    return (jlong) totalModel;
}

extern "C" JNIEXPORT void JNICALL
Java_ai_onnxruntime_example_virtualtryon_core_TotalModel_parsingModelInit(JNIEnv *env, jobject, jlong modelsAddr, jobject assetManager) {
    std::unique_ptr<Ort::Env> environment(new Ort::Env(ORT_LOGGING_LEVEL_VERBOSE, "test"));
    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);

    if (modelsAddr != 0) {
        TotalModel *totalModel = (TotalModel *) modelsAddr;
        totalModel->parsingModelInit(environment, mgr);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_ai_onnxruntime_example_virtualtryon_core_TotalModel_vtoModelInit(JNIEnv *env, jobject, jlong modelsAddr, jobject assetManager) {
    std::unique_ptr<Ort::Env> environment(new Ort::Env(ORT_LOGGING_LEVEL_VERBOSE, "test"));
    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);

    if (modelsAddr != 0) {
        TotalModel *totalModel = (TotalModel *) modelsAddr;
        totalModel->vtoModelInit(environment, mgr);
    }
}

extern "C" JNIEXPORT void JNICALL
Java_ai_onnxruntime_example_virtualtryon_core_TotalModel_deleteModels(JNIEnv *env, jobject, jlong modelsAddr) {
    if (modelsAddr != 0) {
        TotalModel *totalModel = (TotalModel *) modelsAddr;
        delete totalModel;
    }
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_ai_onnxruntime_example_virtualtryon_core_TotalModel_inferenceParsing(JNIEnv *env, jobject, jlong modelsAddr,
                                                                     jbyteArray input1) {
    if (modelsAddr != 0) {

        TotalModel *totalModel = (TotalModel *) modelsAddr;

        uint8_t *frame_uint8 = (uint8_t *) env->GetByteArrayElements(input1, NULL);

        cv::Mat rgb_frame = cv::Mat(960, 540, CV_8UC3, frame_uint8);

        int64_t frameSize = 3 * 960 * 540;
        int64_t parsingInputSize = 3 * 512 * 512;

        cv::Mat resized;
        cv::resize(rgb_frame, resized, cv::Size(288, 512), 0, 0, cv::INTER_LINEAR);

        cv::Mat resizedParsing_with_border;
        cv::copyMakeBorder(resized, resizedParsing_with_border, 0, 0, 112, 112, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

        totalModel->runParsing(resizedParsing_with_border.data);

        cv::Mat parsing_resized;
        cv::Mat parsing_out = cv::Mat(512, 288, CV_8U, totalModel->uint8ParsingOutput.get());
        cv::resize(parsing_out, parsing_resized, cv::Size(540, 960), 0, 0, cv::INTER_NEAREST);

        int64_t outputTensorSize = 960 * 540;

        jbyteArray result;
        result = env->NewByteArray(outputTensorSize);

        env->SetByteArrayRegion(result, 0, outputTensorSize, (jbyte *) parsing_resized.data);

        env->ReleaseByteArrayElements(input1, (jbyte *) frame_uint8, 0);

        return result;
    }
    return env->NewByteArray(0);
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_ai_onnxruntime_example_virtualtryon_core_TotalModel_inferenceVTO(JNIEnv *env, jobject, jlong modelsAddr,
                                                                 jbyteArray input1,
                                                                 jbyteArray input2,
                                                                 jbyteArray input3,
                                                                 jbyteArray parsingJava,
                                                                 jintArray input4,
                                                                 jintArray newSizeJava,
                                                                 jintArray paddingsJava,
                                                                 jintArray corrSizeJava) {

    TotalModel *totalModel = (TotalModel *) modelsAddr;

    uint8_t *frame_uint8 = (uint8_t *) env->GetByteArrayElements(input1, NULL);
    uint8_t *cloth_uint8 = (uint8_t *) env->GetByteArrayElements(input2, NULL);
    uint8_t *edge_uint8 = (uint8_t *) env->GetByteArrayElements(input3, NULL);
    uint8_t *parsingArray = (uint8_t *) env->GetByteArrayElements(parsingJava, NULL);
    int *bb = env->GetIntArrayElements(input4, NULL);
    int *newSize = env->GetIntArrayElements(newSizeJava, NULL);
    int *paddings = env->GetIntArrayElements(paddingsJava, NULL);
    int *corrSize = env->GetIntArrayElements(corrSizeJava, NULL);

    cv::Mat rgb_cloth = cv::Mat(256, 192, CV_8UC3, cloth_uint8);
    cv::Mat edge = cv::Mat(256, 192, CV_8U, edge_uint8);

    cv::Mat parseMat = cv::Mat((bb[3] - bb[2]), (bb[1] - bb[0]), CV_8U, parsingArray);

    uint8_t *allCroppedImg = new uint8_t[3 * (bb[1] - bb[0]) * (bb[3] - bb[2])];
    cropImage(frame_uint8, 3, 960, 540, bb, allCroppedImg);

    cv::Mat crop_frameMat = cv::Mat((bb[3] - bb[2]), (bb[1] - bb[0]), CV_8UC3, allCroppedImg);

    uint8_t *frameRemoveBackground = new uint8_t[3 * (bb[1] - bb[0]) * (bb[3] - bb[2])];
    cv::Mat frameRemoveBackgroundMat;

    rmBackground(crop_frameMat.data, 3, (bb[3] - bb[2]), (bb[1] - bb[0]), parsingArray, frameRemoveBackground);
    frameRemoveBackgroundMat = cv::Mat((bb[3] - bb[2]), (bb[1] - bb[0]), CV_8UC3, frameRemoveBackground);

    cv::Mat resized;
    cv::Mat resizedParsing;

    cv::resize(frameRemoveBackgroundMat, resized, cv::Size(newSize[0], newSize[1]), 0, 0, cv::INTER_CUBIC);
    cv::resize(parseMat, resizedParsing, cv::Size(newSize[0], newSize[1]), 0, 0, cv::INTER_NEAREST);

    cv::Mat resized_withPadding;
    cv::Mat resizedParsing_withPadding;
    cv::copyMakeBorder(resized, resized_withPadding, paddings[0], paddings[1], paddings[2], paddings[3], cv::BORDER_CONSTANT,
                       cv::Scalar(255, 255, 255));
    cv::copyMakeBorder(resizedParsing, resizedParsing_withPadding, paddings[0], paddings[1], paddings[2], paddings[3], cv::BORDER_CONSTANT, 0);

    totalModel->runVto(resized_withPadding.data, resizedParsing_withPadding.data, rgb_cloth.data, edge.data);

    int cropValuePad[4] = {paddings[0], 256 - paddings[1], paddings[2], 192 - paddings[3]};
    float *cropPadGen = new float[3 * (cropValuePad[1] - cropValuePad[0]) * (cropValuePad[3] - cropValuePad[2])];
    float *cropPadMask = new float[(cropValuePad[1] - cropValuePad[0]) * (cropValuePad[3] - cropValuePad[2])];
    cropTensor(totalModel->modelOutput.get(), 3, 256, 192, cropValuePad, cropPadGen);
    cropTensor(totalModel->maskOutput.get(), 1, 256, 192, cropValuePad, cropPadMask);

    int cropValueCorr[4] = {corrSize[2], corrSize[3], corrSize[0], corrSize[1]};
    float *cropCorrGen = new float[3 * (corrSize[3] - corrSize[2]) * (corrSize[1] - corrSize[0])];
    float *cropCorrMask = new float[(corrSize[3] - corrSize[2]) * (corrSize[1] - corrSize[0])];
    cropTensor(cropPadGen, 3, cropValuePad[1] - cropValuePad[0], cropValuePad[3] - cropValuePad[2], cropValueCorr, cropCorrGen);
    cropTensor(cropPadMask, 1, cropValuePad[1] - cropValuePad[0], cropValuePad[3] - cropValuePad[2], cropValueCorr, cropCorrMask);

    uint8_t *gen_uint8 = new uint8_t[3 * (corrSize[3] - corrSize[2]) * (corrSize[1] - corrSize[0])];
    toImage(cropCorrGen, 3, (corrSize[3] - corrSize[2]), (corrSize[1] - corrSize[0]), gen_uint8);
    cv::Mat genMat = cv::Mat((corrSize[3] - corrSize[2]), (corrSize[1] - corrSize[0]), CV_8UC3, gen_uint8);

    uint8_t *mask_uint8 = new uint8_t[(corrSize[3] - corrSize[2]) * (corrSize[1] - corrSize[0])];
    floatToUint8(cropCorrMask, (corrSize[3] - corrSize[2]) * (corrSize[1] - corrSize[0]), mask_uint8);
    cv::Mat maskMat = cv::Mat((corrSize[3] - corrSize[2]), (corrSize[1] - corrSize[0]), CV_8U, mask_uint8);

    cv::Mat resizedGen;
    cv::Mat resizedMask;
    cv::resize(genMat, resizedGen, cv::Size((bb[5] - bb[4]), (bb[7] - bb[6])), 0, 0, cv::INTER_CUBIC);
    cv::resize(maskMat, resizedMask, cv::Size((bb[5] - bb[4]), (bb[7] - bb[6])), 0, 0, cv::INTER_NEAREST);

    uint8_t *croppedParse = new uint8_t[(bb[5] - bb[4]) * (bb[7] - bb[6])];
    const int cropDiffValues[4] = {bb[4] - bb[0], bb[5] - bb[0], bb[6] - bb[2], bb[7] - bb[2]};
    cropImage(parseMat.data, 1, (bb[3] - bb[2]), (bb[1] - bb[0]), cropDiffValues, croppedParse);

    uint8_t *outputMask = new uint8_t[(bb[5] - bb[4]) * (bb[7] - bb[6])];
    makeMask(croppedParse, resizedMask.data, (bb[7] - bb[6]), (bb[5] - bb[4]), outputMask);

    int64_t outputTensorSize = (bb[5] - bb[4]) * (bb[7] - bb[6]);

    jobjectArray result = env->NewObjectArray(2, env->FindClass("[B"), NULL);

    jbyteArray genOutput = env->NewByteArray(3 * outputTensorSize);
    jbyteArray maskOutput = env->NewByteArray(outputTensorSize);

    env->SetByteArrayRegion(genOutput, 0, 3 * outputTensorSize, (jbyte *) resizedGen.data);

    env->SetByteArrayRegion(maskOutput, 0, outputTensorSize, (jbyte *) resizedMask.data);

    env->SetObjectArrayElement(result, 0, genOutput);
    env->SetObjectArrayElement(result, 1, maskOutput);


    env->ReleaseByteArrayElements(input1, (jbyte *) frame_uint8, 0);
    env->ReleaseByteArrayElements(input2, (jbyte *) cloth_uint8, 0);
    env->ReleaseByteArrayElements(input3, (jbyte *) edge_uint8, 0);
    env->ReleaseByteArrayElements(parsingJava, (jbyte *) parsingArray, 0);
    env->ReleaseIntArrayElements(input4, bb, 0);
    env->ReleaseIntArrayElements(newSizeJava, newSize, 0);
    env->ReleaseIntArrayElements(paddingsJava, paddings, 0);
    env->ReleaseIntArrayElements(corrSizeJava, corrSize, 0);

    delete[] frameRemoveBackground;
    delete[] cropPadGen;
    delete[] cropCorrGen;
    delete[] cropPadMask;
    delete[] cropCorrMask;
    delete[] croppedParse;
    delete[] outputMask;
    delete[] gen_uint8;
    delete[] allCroppedImg;

    return result;
}