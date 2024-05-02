
#include <jni.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "imageUtils.h"
#include "modelInference.h"

extern "C" JNIEXPORT jintArray JNICALL
Java_ai_onnxruntime_example_virtualtryon_core_VTO_getBoundingBox(JNIEnv *env, jobject, jbyteArray parsingJava, jint parsingSize) {
    uint8_t *parsingArray = (uint8_t *) env->GetByteArrayElements(parsingJava, NULL);

    cv::Mat parseMat = cv::Mat(960, 540, CV_8U, parsingArray);

    const int allPaddings[8] = {20, 20, 20, 30, 10, 10, 20, 20};
    int bb[8];
    makeBoundingBox(parseMat.data, 960, 540, allPaddings, bb, parsingSize);

    jintArray result;
    result = env->NewIntArray(8);

    env->SetIntArrayRegion(result, 0, 8, bb);
    env->ReleaseByteArrayElements(parsingJava, (jbyte *) parsingArray, 0);

    return result;
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_ai_onnxruntime_example_virtualtryon_core_VTO_cropParse(JNIEnv *env, jobject, jbyteArray parsingJava, jintArray bbJava) {


    uint8_t *parsingArray = (uint8_t *) env->GetByteArrayElements(parsingJava, NULL);
    int *bb = env->GetIntArrayElements(bbJava, NULL);

    cv::Mat parseMat = cv::Mat(960, 540, CV_8U, parsingArray);

    uint8_t *allCroppedParsing = new uint8_t[(bb[1] - bb[0]) * (bb[3] - bb[2])];
    cropImage(parseMat.data, 1, 960, 540, bb, allCroppedParsing);

    jbyteArray result;

    int64_t outputTensorSize = (bb[1] - bb[0]) * (bb[3] - bb[2]);
    result = env->NewByteArray(outputTensorSize);

    env->SetByteArrayRegion(result, 0, outputTensorSize, (jbyte *) allCroppedParsing);

    env->ReleaseByteArrayElements(parsingJava, (jbyte *) parsingArray, 0);
    env->ReleaseIntArrayElements(bbJava, bb, 0);

    delete[] allCroppedParsing;

    return result;
}


extern "C" JNIEXPORT jbyteArray JNICALL
Java_ai_onnxruntime_example_virtualtryon_core_VTO_cropTorsoParse(JNIEnv *env, jobject, jbyteArray parsingJava, jintArray bbJava) {


    uint8_t *parsingArray = (uint8_t *) env->GetByteArrayElements(parsingJava, NULL);
    int *bb = env->GetIntArrayElements(bbJava, NULL);

    cv::Mat parseMat = cv::Mat(960, 540, CV_8U, parsingArray);

    uint8_t *allCroppedParsing = new uint8_t[(bb[5] - bb[4]) * (bb[7] - bb[6])];
    cropImage(parseMat.data, 1, 960, 540, bb + 4, allCroppedParsing);

    jbyteArray result;

    int64_t outputTensorSize = (bb[5] - bb[4]) * (bb[7] - bb[6]);
    result = env->NewByteArray(outputTensorSize);

    env->SetByteArrayRegion(result, 0, outputTensorSize, (jbyte *) allCroppedParsing);

    env->ReleaseByteArrayElements(parsingJava, (jbyte *) parsingArray, 0);
    env->ReleaseIntArrayElements(bbJava, bb, 0);

    delete[] allCroppedParsing;

    return result;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_ai_onnxruntime_example_virtualtryon_core_VTO_orbExtraction(JNIEnv *env, jobject, jintArray bbJava,
                                                                      jbyteArray frameJava,
                                                                      jbyteArray parsingJava) {
    int *bb = env->GetIntArrayElements(bbJava, NULL);
    //    const char* path = env->GetStringUTFChars(javaPath, 0);
    uint8_t *frameUint8 = (uint8_t *) env->GetByteArrayElements(frameJava, NULL);
    uint8_t *parsingArray = (uint8_t *) env->GetByteArrayElements(parsingJava, NULL);


    cv::Mat rgbFrame = cv::Mat(960, 540, CV_8UC3, frameUint8);

    cv::Mat parseMat = cv::Mat(960, 540, CV_8U, parsingArray);

    uint8_t *src = new uint8_t[3 * 960 * 540];
    makeOrbSource(rgbFrame.data, 3, 960, 540, bb, src);
    cv::Mat srcMat = cv::Mat(960, 540, CV_8UC3, src);
    auto orb = cv::ORB::create(300);


    cv::Mat dilParse;
    cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);

    cv::dilate(parseMat, dilParse, kernel, cv::Point(-1, -1), 3);
    std::vector<cv::KeyPoint> kps;
    orb->detect(srcMat, kps, dilParse);

    std::sort(kps.begin(), kps.end(), compare);
    std::vector<cv::Point2f> p0;
    cv::KeyPoint::convert(kps, p0);

    std::vector<float> dP0;
    if (p0.size() != 0) {
        dP0.push_back(p0[0].x);
        dP0.push_back(p0[0].y);
        for (auto p :p0) {
            if (dP0.size() >= 60) break;
            else {
                bool acc = true;
                for (int i = 0; i < dP0.size(); i += 2) {
                    if (calcDist(dP0[i], dP0[i + 1], p) < 20) {
                        acc = false;
                        break;
                    }
                }
                if (acc == false) {
                    continue;
                }
                dP0.push_back(p.x);
                dP0.push_back(p.y);
            }
        }
    }

    jfloatArray result;
    result = env->NewFloatArray(dP0.size());

    env->SetFloatArrayRegion(result, 0, dP0.size(), &dP0[0]);
    env->ReleaseIntArrayElements(bbJava, bb, 0);
    env->ReleaseByteArrayElements(frameJava, (jbyte *) frameUint8, 0);
    env->ReleaseByteArrayElements(parsingJava, (jbyte *) parsingArray, 0);

    delete[] src;

    return result;
}



extern "C" JNIEXPORT jfloatArray JNICALL
Java_ai_onnxruntime_example_virtualtryon_core_VTO_opticalFlow(JNIEnv *env, jobject,
                                                                    jfloatArray p0Java,
                                                                    jfloatArray firstP0Java,
                                                                    jbyteArray oldFrameJava,
                                                                    jbyteArray frameJava) {
    float *p0 = env->GetFloatArrayElements(p0Java, NULL);
    float *firstP0 = env->GetFloatArrayElements(firstP0Java, NULL);
    uint8_t *oldFrameUint8 = (uint8_t *) env->GetByteArrayElements(oldFrameJava, NULL);
    uint8_t *frameUint8 = (uint8_t *) env->GetByteArrayElements(frameJava, NULL);

    cv::Mat oldFrame = cv::Mat(960, 540, CV_8UC3, oldFrameUint8);
    cv::Mat frame = cv::Mat(960, 540, CV_8UC3, frameUint8);

    jsize size = env->GetArrayLength(p0Java);

    cv::Mat oldGray;
    cv::cvtColor(oldFrame, oldGray, cv::COLOR_BGR2GRAY);

    cv::Mat frameGray;
    cv::cvtColor(frame, frameGray, cv::COLOR_BGR2GRAY);

    float p1[size];
    std::vector<cv::Point2f> p0_vec, p1_vec;
    for (int i = 0; i < size; i += 2) {
        p0_vec.push_back(cv::Point2f(p0[i], p0[i + 1]));
    }

    std::vector<uint8_t> st;
    std::vector<float> err;

    cv::calcOpticalFlowPyrLK(oldGray, frameGray, p0_vec, p1_vec, st, err, cv::Size(15, 15), 3,
                             cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 10, 0.03));

    std::vector<float> goodPoints;
    for (int i = 0; i < size / 2; i++) {
        if (st[i] == 1) {
            goodPoints.push_back(p1_vec[i].x);
            goodPoints.push_back(p1_vec[i].y);
        }
    }

    for (int i = 0; i < size / 2; i++) {
        if (st[i] == 1) {
            goodPoints.push_back(firstP0[i * 2]);
            goodPoints.push_back(firstP0[i * 2 + 1]);
        }
    }

    jfloatArray result;
    result = env->NewFloatArray(goodPoints.size());

    env->SetFloatArrayRegion(result, 0, goodPoints.size(), &goodPoints[0]);
    env->ReleaseFloatArrayElements(p0Java, p0, 0);
    env->ReleaseFloatArrayElements(firstP0Java, firstP0, 0);
    env->ReleaseByteArrayElements(frameJava, (jbyte *) frameUint8, 0);
    env->ReleaseByteArrayElements(oldFrameJava, (jbyte *) oldFrameUint8, 0);

    return result;
}


extern "C" JNIEXPORT jobjectArray JNICALL
Java_ai_onnxruntime_example_virtualtryon_core_VTO_gridGen(JNIEnv *env, jobject,
                                                                jfloatArray p0Java,
                                                                jfloatArray firstP0Java,
                                                                jintArray diffJava,
                                                                jintArray bbJava
) {
    float *p0 = env->GetFloatArrayElements(p0Java, NULL);
    float *firstP0 = env->GetFloatArrayElements(firstP0Java, NULL);
    int *diff = env->GetIntArrayElements(diffJava, NULL);
    int *bb = env->GetIntArrayElements(bbJava, NULL);
    int size = env->GetArrayLength(p0Java);

    int N = size / 2;

    float new_norm[size];
    float first_norm[size];

    normalize(p0, size, bb, diff, new_norm);
    normalize(firstP0, size, bb, first_norm);

    int target_height = bb[7] - bb[6];
    int target_width = bb[5] - bb[4];


    float *forwardKernel = new float[(N + 3) * (N + 3)];
    float *targetControlPartialRepr = new float[N * N];
    computePartialRepr(new_norm, new_norm, N, N, targetControlPartialRepr);
    makeForwardKernel(targetControlPartialRepr, new_norm, N, forwardKernel);


    cv::Mat forwardKernelMat = cv::Mat(N + 3, N + 3, CV_32F, forwardKernel);
    cv::Mat inverseKernelMat = forwardKernelMat.inv();

    int HW = target_height * target_width;
    float *targetCoordinate = new float[HW * 2];

    makeTargetCoordinate(targetCoordinate, target_height, target_width);
    float *targetCoordinatePartialRepr = new float[HW * N];
    computePartialRepr(targetCoordinate, new_norm, HW, N, targetCoordinatePartialRepr);

    float *targetCoordinateRepr = new float[HW * (N + 3)];
    makeTargetCoordinateRepr(targetCoordinatePartialRepr, targetCoordinate, HW, N, targetCoordinateRepr);

    cv::Mat targetCoordinateReprMat = cv::Mat(HW, N + 3, CV_32F, targetCoordinateRepr);

    cv::Mat sourceControlPointsMat = cv::Mat(N, 2, CV_32F, first_norm);
    cv::Mat YMat(N + 3, 2, CV_32F);
    cv::copyMakeBorder(sourceControlPointsMat, YMat, 0, 3, 0, 0, cv::BORDER_CONSTANT, 0);

    cv::Mat mappingMatrix(N + 3, 2, CV_32F);
    mappingMatrix = inverseKernelMat * YMat;
    cv::Mat sourceCoordinate(HW, 2, CV_32F);
    sourceCoordinate = targetCoordinateReprMat * mappingMatrix;

    float *tpsPartialRepr = new float[N * N];
    computePartialRepr(first_norm, new_norm, N, N, tpsPartialRepr);

    float *tpsRepr = new float[N * (N + 3)];
    makeTargetCoordinateRepr(tpsPartialRepr, first_norm, N, N, tpsRepr);

    cv::Mat tpsReprMat = cv::Mat(N, N + 3, CV_32F, tpsRepr);
    cv::Mat tps_pred(N, 2, CV_32F);
    tps_pred = tpsReprMat * mappingMatrix;

    float *err = new float[1];
    calcErr((float *) tps_pred.data, first_norm, 2 * N, err, target_height, target_width);

    jobjectArray result = env->NewObjectArray(2, env->FindClass("[F"), NULL);

    jfloatArray grid = env->NewFloatArray(2 * HW);
    jfloatArray errJava = env->NewFloatArray(1);

    env->SetFloatArrayRegion(grid, 0, 2 * HW, (float *) sourceCoordinate.data);
    env->SetFloatArrayRegion(errJava, 0, 1, err);

    env->SetObjectArrayElement(result, 0, grid);
    env->SetObjectArrayElement(result, 1, errJava);

    env->ReleaseFloatArrayElements(p0Java, p0, 0);
    env->ReleaseFloatArrayElements(firstP0Java, firstP0, 0);
    env->ReleaseIntArrayElements(diffJava, diff, 0);
    env->ReleaseIntArrayElements(bbJava, bb, 0);

    delete[] forwardKernel;
    delete[] targetControlPartialRepr;
    delete[] targetCoordinateRepr;
    delete[] targetCoordinate;
    delete[] targetCoordinatePartialRepr;
    delete[] tpsRepr;
    delete[] tpsPartialRepr;

    return result;
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_ai_onnxruntime_example_virtualtryon_core_VTO_getOFrame(JNIEnv *env, jobject,
                                                                  jbyteArray input1,
                                                                  jbyteArray maskJava,
                                                                  jintArray input4) {
    uint8_t *frameUint8 = (uint8_t *) env->GetByteArrayElements(input1, NULL);
    uint8_t *maskArray = (uint8_t *) env->GetByteArrayElements(maskJava, NULL);
    int *bb = env->GetIntArrayElements(input4, NULL);

    uint8_t *allCroppedImg = new uint8_t[3 * (bb[5] - bb[4]) * (bb[7] - bb[6])];
    cropImage(frameUint8, 3, 960, 540, bb + 4, allCroppedImg);

    uint8_t *frameRemoveBack = new uint8_t[3 * (bb[7] - bb[6]) * (bb[5] - bb[4])];
    rmBackground(allCroppedImg, 3, (bb[7] - bb[6]), (bb[5] - bb[4]), maskArray, frameRemoveBack);

    int64_t outputTensorSize = 3 * (bb[5] - bb[4]) * (bb[7] - bb[6]);

    jbyteArray result;
    result = env->NewByteArray(outputTensorSize);

    env->SetByteArrayRegion(result, 0, outputTensorSize, (jbyte *) frameRemoveBack);

    env->ReleaseByteArrayElements(input1, (jbyte *) frameUint8, 0);
    env->ReleaseByteArrayElements(maskJava, (jbyte *) maskArray, 0);
    env->ReleaseIntArrayElements(input4, bb, 0);

    delete[] allCroppedImg;
    delete[] frameRemoveBack;

    return result;
}


extern "C" JNIEXPORT jintArray JNICALL
Java_ai_onnxruntime_example_virtualtryon_core_VTO_byteToInt(JNIEnv *env, jobject,
                                                                  jbyteArray genResultJava) {

    uint8_t *genResult = (uint8_t *) env->GetByteArrayElements(genResultJava, NULL);

    int *outputResult = new int[540 * 960 * 3];
    uint8ToInt(genResult, 540 * 960 * 3, outputResult);

    jintArray result;
    result = env->NewIntArray(540 * 960 * 3);

    env->SetIntArrayRegion(result, 0, 540 * 960 * 3, outputResult);

    env->ReleaseByteArrayElements(genResultJava, (jbyte *) genResult, 0);

    delete[] outputResult;

    return result;
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_ai_onnxruntime_example_virtualtryon_core_VTO_genToFrameMask(JNIEnv *env, jobject,
                                                                       jbyteArray genResultJava,
                                                                       jintArray bbJava,
                                                                       jintArray paddingsJava,
                                                                       jintArray corrSizeJava,
                                                                       jbyteArray frameJava,
                                                                       jbyteArray maskJava) {

    uint8_t *genResult = (uint8_t *) env->GetByteArrayElements(genResultJava, NULL);
    int *bb = env->GetIntArrayElements(bbJava, NULL);
    int *paddings = env->GetIntArrayElements(paddingsJava, NULL);
    int *corrSize = env->GetIntArrayElements(corrSizeJava, NULL);
    uint8_t *frameUint8 = (uint8_t *) env->GetByteArrayElements(frameJava, NULL);

    uint8_t *maskArray = (uint8_t *) env->GetByteArrayElements(maskJava, NULL);
    cv::Mat rgbFrame = cv::Mat(960, 540, CV_8UC3, frameUint8);

    int64_t outputTensorSize = 540 * 960 * 3;
    uint8_t *outputFrame = new uint8_t[540 * 960 * 3];
    putOutputToFrame(rgbFrame.data, 3, 960, 540, genResult, maskArray, bb, outputFrame);


    jbyteArray result;
    result = env->NewByteArray(outputTensorSize);

    env->SetByteArrayRegion(result, 0, outputTensorSize, (jbyte *) outputFrame);

    env->ReleaseByteArrayElements(genResultJava, (jbyte *) genResult, 0);
    env->ReleaseIntArrayElements(bbJava, bb, 0);
    env->ReleaseIntArrayElements(paddingsJava, paddings, 0);
    env->ReleaseIntArrayElements(corrSizeJava, corrSize, 0);
    env->ReleaseByteArrayElements(frameJava, (jbyte *) frameUint8, 0);
    env->ReleaseByteArrayElements(maskJava, (jbyte *) maskArray, 0);

    delete[] outputFrame;

    return result;
}


extern "C" JNIEXPORT jfloatArray JNICALL
Java_ai_onnxruntime_example_virtualtryon_core_VTO_calcMap(JNIEnv *env, jobject,
                                                                jintArray bbJava,
                                                                jintArray paddingsJava,
                                                                jintArray corrSizeJava,
                                                                jfloatArray gridJava) {

    int *bb = env->GetIntArrayElements(bbJava, NULL);
    int *paddings = env->GetIntArrayElements(paddingsJava, NULL);
    int *corrSize = env->GetIntArrayElements(corrSizeJava, NULL);
    float *grid = env->GetFloatArrayElements(gridJava, NULL);

    float *mapXYFloat = new float[(bb[5] - bb[4]) * (bb[7] - bb[6]) * 2];
    getMap(grid, (bb[5] - bb[4]), (bb[7] - bb[6]), mapXYFloat);

    int64_t outputTensorSize = (bb[5] - bb[4]) * (bb[7] - bb[6]) * 2;

    jfloatArray result;
    result = env->NewFloatArray(outputTensorSize);

    env->SetFloatArrayRegion(result, 0, outputTensorSize, mapXYFloat);

    env->ReleaseIntArrayElements(bbJava, bb, 0);
    env->ReleaseIntArrayElements(paddingsJava, paddings, 0);
    env->ReleaseIntArrayElements(corrSizeJava, corrSize, 0);
    env->ReleaseFloatArrayElements(gridJava, grid, 0);

    delete[] mapXYFloat;

    return result;
}
extern "C" JNIEXPORT jbyteArray JNICALL
Java_ai_onnxruntime_example_virtualtryon_core_VTO_genWithMapMask(JNIEnv *env, jobject,
                                                                       jbyteArray genResultJava,
                                                                       jintArray bbJava,
                                                                       jintArray paddingsJava,
                                                                       jintArray corrSizeJava,
                                                                       jfloatArray mapXYJava,
                                                                       jbyteArray frameJava,
                                                                       jbyteArray maskJava) {

    uint8_t *genResult = (uint8_t *) env->GetByteArrayElements(genResultJava, NULL);
    int *bb = env->GetIntArrayElements(bbJava, NULL);
    int *paddings = env->GetIntArrayElements(paddingsJava, NULL);
    int *corrSize = env->GetIntArrayElements(corrSizeJava, NULL);
    float *mapXYFloat = env->GetFloatArrayElements(mapXYJava, NULL);

    uint8_t *frameUint8 = (uint8_t *) env->GetByteArrayElements(frameJava, NULL);
    uint8_t *maskArray = (uint8_t *) env->GetByteArrayElements(maskJava, NULL);

    auto start = std::chrono::high_resolution_clock::now();

    cv::Mat rgbFrame = cv::Mat(960, 540, CV_8UC3, frameUint8);

    cv::Mat genMat = cv::Mat((bb[7] - bb[6]), (bb[5] - bb[4]), CV_8UC3, genResult);
    cv::Mat parseMat = cv::Mat((bb[7] - bb[6]), (bb[5] - bb[4]), CV_8U, maskArray);

    cv::Mat mapX = cv::Mat((bb[7] - bb[6]), (bb[5] - bb[4]), CV_32F, mapXYFloat);
    cv::Mat mapY = cv::Mat((bb[7] - bb[6]), (bb[5] - bb[4]), CV_32F, mapXYFloat + (bb[5] - bb[4]) * (bb[7] - bb[6]));

    cv::Mat remapGen;
    cv::remap(genMat, remapGen, mapX, mapY, cv::INTER_CUBIC);

    cv::Mat remapParsing;
    cv::remap(parseMat, remapParsing, mapX, mapY, cv::INTER_NEAREST);

    int64_t outputTensorSize = 540 * 960 * 3;
    uint8_t *outputFrame = new uint8_t[540 * 960 * 3];
    putOutputToFrame(rgbFrame.data, 3, 960, 540, remapGen.data, remapParsing.data, bb, outputFrame);

    jbyteArray result;
    result = env->NewByteArray(outputTensorSize);

    env->SetByteArrayRegion(result, 0, outputTensorSize, (jbyte *) outputFrame);

    env->ReleaseByteArrayElements(genResultJava, (jbyte *) genResult, 0);
    env->ReleaseIntArrayElements(bbJava, bb, 0);
    env->ReleaseIntArrayElements(paddingsJava, paddings, 0);
    env->ReleaseIntArrayElements(corrSizeJava, corrSize, 0);
    env->ReleaseFloatArrayElements(mapXYJava, mapXYFloat, 0);
    env->ReleaseByteArrayElements(frameJava, (jbyte *) frameUint8, 0);
    env->ReleaseByteArrayElements(maskJava, (jbyte *) maskArray, 0);

    delete[] outputFrame;

    return result;
}

extern "C" JNIEXPORT jfloat JNICALL
Java_ai_onnxruntime_example_virtualtryon_core_VTO_calcNCC(JNIEnv *env, jobject,
                                                                jbyteArray oFrameJava,
                                                                jintArray bbJava,
                                                                jfloatArray mapXYJava,
                                                                jbyteArray frameJava,
                                                                jbyteArray maskJava) {

    uint8_t *oFrameUint8 = (uint8_t *) env->GetByteArrayElements(oFrameJava, NULL);
    int *bb = env->GetIntArrayElements(bbJava, NULL);
    float *mapXYFloat = env->GetFloatArrayElements(mapXYJava, NULL);

    uint8_t *frameUint8 = (uint8_t *) env->GetByteArrayElements(frameJava, NULL);
    uint8_t *maskArray = (uint8_t *) env->GetByteArrayElements(maskJava, NULL);

    cv::Mat frameMat = cv::Mat(960, 540, CV_8UC3, frameUint8);
    cv::Mat oFrameMat = cv::Mat((bb[7] - bb[6]), (bb[5] - bb[4]), CV_8UC3, oFrameUint8);

    cv::Mat maskMat = cv::Mat((bb[7] - bb[6]), (bb[5] - bb[4]), CV_8U, maskArray);

    cv::Mat cropFrame;
    frameMat(cv::Range(bb[6], bb[7]), cv::Range(bb[4], bb[5])).copyTo(cropFrame);

    cv::Mat mapX = cv::Mat((bb[7] - bb[6]), (bb[5] - bb[4]), CV_32F, mapXYFloat);
    cv::Mat mapY = cv::Mat((bb[7] - bb[6]), (bb[5] - bb[4]), CV_32F, mapXYFloat + (bb[5] - bb[4]) * (bb[7] - bb[6]));

    cv::Mat remapOrgFrame;
    cv::remap(oFrameMat, remapOrgFrame, mapX, mapY, cv::INTER_LINEAR);

    cv::Mat remap_mask;
    cv::remap(maskMat, remap_mask, mapX, mapY, cv::INTER_NEAREST);

    uint8_t *remapOrgFrameWithBack = new uint8_t[3 * (bb[7] - bb[6]) * (bb[5] - bb[4])];
    addBackground(remapOrgFrame.data, cropFrame.data, 3, (bb[7] - bb[6]), (bb[5] - bb[4]), remap_mask.data, remapOrgFrameWithBack);
    cv::Mat remapOrgFrameWithBackMat = cv::Mat((bb[7] - bb[6]), (bb[5] - bb[4]), CV_8UC3, remapOrgFrameWithBack);

    jfloat result;

    cv::Mat ncc = cv::Mat(1, 1, CV_32FC1);
    cv::matchTemplate(cropFrame, remapOrgFrameWithBackMat, ncc, cv::TM_CCORR_NORMED);
    result = (jfloat) ncc.at<float>(0, 0);

    env->ReleaseByteArrayElements(oFrameJava, (jbyte *) oFrameUint8, 0);
    env->ReleaseIntArrayElements(bbJava, bb, 0);
    env->ReleaseFloatArrayElements(mapXYJava, mapXYFloat, 0);
    env->ReleaseByteArrayElements(frameJava, (jbyte *) frameUint8, 0);
    env->ReleaseByteArrayElements(maskJava, (jbyte *) maskArray, 0);

    delete[] remapOrgFrameWithBack;

    return result;
}

extern "C" JNIEXPORT jbyteArray JNICALL
Java_ai_onnxruntime_example_virtualtryon_core_VTO_convertToNV21(JNIEnv *env, jobject,
                                                                      jbyteArray rgbFrameJava) {

    uint8_t *rgbFrameArray = (uint8_t *) env->GetByteArrayElements(rgbFrameJava, NULL);


    cv::Mat rgbFrame = cv::Mat(960, 540, CV_8UC3, rgbFrameArray);
    cv::Mat yuvFrame;
    cv::cvtColor(rgbFrame, yuvFrame, cv::COLOR_BGR2YUV_YV12);

    jbyteArray result;
    result = env->NewByteArray(540 * 960 * 3 / 2);

    env->SetByteArrayRegion(result, 0, 540 * 960 * 3 / 2, (jbyte *) yuvFrame.data);

    env->ReleaseByteArrayElements(rgbFrameJava, (jbyte *) rgbFrameArray, 0);

    return result;
}



extern "C" JNIEXPORT jbyteArray JNICALL
Java_ai_onnxruntime_example_virtualtryon_core_VTO_getMaskFromRGB(JNIEnv *env, jobject,
                                                                jbyteArray inputImageJava, jint height, jint width) {

    uint8_t *inputImage = (uint8_t *) env->GetByteArrayElements(inputImageJava, NULL);

    cv::Mat image(height, width, CV_8UC3, inputImage);

    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Thresholding
    cv::Mat thresh;
    cv::threshold(gray, thresh, 0, 255, cv::THRESH_BINARY_INV + cv::THRESH_OTSU);

    // Morphological closing to remove noise
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::Mat closing;
    cv::morphologyEx(thresh, closing, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 2);

    // Dilate to find sure background
    cv::Mat sure_bg;
    cv::dilate(closing, sure_bg, kernel, cv::Point(-1, -1), 3);

    // Finding sure foreground area using Distance Transformation
    cv::Mat dist_transform;
    cv::distanceTransform(closing, dist_transform, cv::DIST_L2, 5);
    cv::Mat sure_fg;
    double minVal, maxVal;
    cv::minMaxLoc(dist_transform, &minVal, &maxVal);
    cv::threshold(dist_transform, sure_fg, 0.7*maxVal, 255, 0);

    // Finding unknown region
    sure_fg.convertTo(sure_fg, CV_8U);
    cv::Mat unknown = sure_bg - sure_fg;

    // Marker labeling
    cv::Mat markers;
    cv::connectedComponents(sure_fg, markers);

    // Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1;

    // Mark the region of unknown with zero
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            if (unknown.at<uint8_t>(i, j) == 255) {
                markers.at<int>(i, j) = 0;
            }
        }
    }

    // Apply the Watershed algorithm
    cv::watershed(image, markers);

    // Create the mask
    cv::Mat mask = cv::Mat::zeros(gray.size(), CV_8UC1);
    for (int i = 0; i < markers.rows; i++) {
        for (int j = 0; j < markers.cols; j++) {
            if (markers.at<int>(i, j) > 1) { // Assuming the cloth region is not labeled as 1 (background)
                mask.at<uint8_t>(i, j) = 255;
            }
        }
    }

    jbyteArray result;
    result = env->NewByteArray(width*height);

    env->SetByteArrayRegion(result, 0, width*height, (jbyte *) mask.data);
    env->ReleaseByteArrayElements(inputImageJava, (jbyte *) inputImage, 0);

    return result;
}

