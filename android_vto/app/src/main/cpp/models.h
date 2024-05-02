//
// Created by silky on 2/2/2022.
//

#ifndef ORT_VIRTUALTRYON_MODELS_H
#define ORT_VIRTUALTRYON_MODELS_H


#include <stdio.h>
#include <chrono>
#include <inttypes.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include "modelUtils.h"
#include "onnxruntime_cxx_api.h"
#include "onnxruntime_c_api.h"
#include "nnapi_provider_factory.h"


class TotalModel {

private:

    std::unique_ptr<Ort::Env> env;

    std::unique_ptr<Ort::Session> warpSession;
    std::unique_ptr<Ort::Session> genSession;
    std::unique_ptr<Ort::Session> parsingSession;

    int parsingSize;

    int parsingImgHeight;
    int parsingImgWidth;
    int vtoImgHeight;
    int vtoImgWidth;

    size_t parsingInputNums;
    size_t parsingOutputNums;
    size_t warpInputNums;
    size_t warpOutputNums;
    size_t genInputNums;
    size_t genOutputNums;

    std::vector<size_t> parsingInputTensorSizeArr;
    std::vector<size_t> parsingOutputTensorSizeArr;
    std::vector<size_t> warpInputTensorSizeArr;
    std::vector<size_t> genInputTensorSizeArr;
    std::vector<size_t> warpOutputTensorSizeArr;
    std::vector<size_t> genOutputTensorSizeArr;

    std::vector<std::vector<int64_t>> parsingInputNodeDimsArr;
    std::vector<std::vector<int64_t>> parsingOutputNodeDimsArr;
    std::vector<std::vector<int64_t>> warpInputNodeDimsArr;
    std::vector<std::vector<int64_t>> genInputNodeDimsArr;
    std::vector<std::vector<int64_t>> warpOutputNodeDimsArr;
    std::vector<std::vector<int64_t>> genOutputNodeDimsArr;

    Ort::Value parsingInputTensor{nullptr};
    std::vector<Ort::Value> warpInputTensors;
    Ort::Value genInputTensor{nullptr};

    std::vector<const char *> parsingInputNodeNames;
    std::vector<const char *> parsingOutputNodeNames;
    std::vector<const char *> warpInputNodeNames;
    std::vector<const char *> genInputNodeNames;
    std::vector<const char *> warpOutputNodeNames;
    std::vector<const char *> genOutputNodeNames;

    std::unique_ptr<float[]> frameCHW512;
    std::unique_ptr<float[]> frameCHW;
    std::unique_ptr<float[]> parsingCHW;
    std::unique_ptr<float[]> clothCHW;
    std::unique_ptr<float[]> edgeCHW;
    std::unique_ptr<float[]> genInput;

    void createParsingInputBuffer();

    void createVtoInputBuffer();

public:
    TotalModel();

    TotalModel(const TotalModel &) = delete;

    TotalModel &operator=(const TotalModel &) = delete;

    void parsingModelInit(std::unique_ptr<Ort::Env> &env_, AAssetManager *mgr);

    void vtoModelInit(std::unique_ptr<Ort::Env> &env_, AAssetManager *mgr);

    void runParsing(uint8_t *pixels);

    void runVto(uint8_t *image, uint8_t *parsing, uint8_t *cloth, uint8_t *edge);

    std::unique_ptr<int[]> outputImage;
    std::unique_ptr<uint8_t[]> uint8ParsingOutput;

    ~TotalModel();

    std::unique_ptr<float[]> modelOutput;
    std::unique_ptr<float[]> maskOutput;
};


#endif //ORT_VIRTUALTRYON_MODELS_H
