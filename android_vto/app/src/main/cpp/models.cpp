#include "models.h"


TotalModel::TotalModel() {}

// Definition of Virtual try on model functions
void TotalModel::parsingModelInit(std::unique_ptr<Ort::Env> &env_, AAssetManager *mgr) {
    env = std::move(env_);

    //Parsing Model initialization Start
    Ort::SessionOptions parsingSessionOptions;
    parsingSessionOptions.AddConfigEntry("session.load_model_format", "ORT");
    parsingSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    AAsset *parsingAsset = AAssetManager_open(mgr, "parsing.ort", 0);
    parsingSession = std::make_unique<Ort::Session>(*env.get(), AAsset_getBuffer(parsingAsset), static_cast<size_t>(AAsset_getLength(parsingAsset)),
                                                    parsingSessionOptions);

    createParsingInputBuffer();
}

// Definition of Virtual try on model functions
void TotalModel::vtoModelInit(std::unique_ptr<Ort::Env> &env_, AAssetManager *mgr) {
    env = std::move(env_);

    // Warp Model initialization Start
    Ort::SessionOptions warpSessionOptions;
    warpSessionOptions.AddConfigEntry("session.load_model_format", "ORT");

    AAsset *warpAsset = AAssetManager_open(mgr, "warp.ort", 0);
    warpSession = std::make_unique<Ort::Session>(*env.get(), AAsset_getBuffer(warpAsset), static_cast<size_t>(AAsset_getLength(warpAsset)),
                                                 warpSessionOptions);

    //Gen Model initialization Start
    Ort::SessionOptions genSessionOptions;
    genSessionOptions.AddConfigEntry("session.load_model_format", "ORT");
    genSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    AAsset *genAsset = AAssetManager_open(mgr, "gen.ort", 0);
    genSession = std::make_unique<Ort::Session>(*env.get(), AAsset_getBuffer(genAsset), static_cast<size_t>(AAsset_getLength(genAsset)),
                                                genSessionOptions);

    createVtoInputBuffer();
}


void TotalModel::createParsingInputBuffer() {
    parsingSize = 7;

    parsingImgHeight = 512;
    parsingImgWidth = 512;

    Ort::AllocatorWithDefaultOptions parsingAllocator;
    Ort::MemoryInfo parsingMemoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // input node initialize
    parsingInputNums = parsingSession->GetInputCount();
    for (int idx = 0; idx < parsingInputNums; idx++) {
        char *parsingInput_name = parsingSession->GetInputName(idx, parsingAllocator);
        std::vector<int64_t> parsingInputNodeDims = parsingSession->GetInputTypeInfo(idx).GetTensorTypeAndShapeInfo().GetShape();
        size_t parsingInputTensorSize = calculateProduct(parsingInputNodeDims);

        parsingInputNodeNames.push_back(parsingInput_name);
        parsingInputNodeDimsArr.push_back(parsingInputNodeDims);
        parsingInputTensorSizeArr.push_back(parsingInputTensorSize);
    }

    // output node initialize
    parsingOutputNums = parsingSession->GetOutputCount();
    for (int idx = 0; idx < parsingOutputNums; idx++) {
        char *parsingOutput_name = parsingSession->GetOutputName(idx, parsingAllocator);
        std::vector<int64_t> parsingOutputNodeDims = parsingSession->GetOutputTypeInfo(idx).GetTensorTypeAndShapeInfo().GetShape();
        size_t parsingOutputTensorSize = calculateProduct(parsingOutputNodeDims);

        parsingOutputNodeNames.push_back(parsingOutput_name);
        parsingOutputNodeDimsArr.push_back(parsingOutputNodeDims);
        parsingOutputTensorSizeArr.push_back(parsingOutputTensorSize);
    }

    frameCHW512 = std::make_unique<float[]>(parsingInputTensorSizeArr[0]);
    uint8ParsingOutput = std::make_unique<uint8_t[]>(parsingOutputTensorSizeArr[0]);

    parsingInputTensor = Ort::Value::CreateTensor<float>(parsingMemoryInfo, frameCHW512.get(), parsingInputTensorSizeArr[0],
                                                         parsingInputNodeDimsArr[0].data(), parsingInputNodeDimsArr[0].size());
}

void TotalModel::createVtoInputBuffer() {
    parsingSize = 7;

    vtoImgHeight = 256;
    vtoImgWidth = 192;

    Ort::MemoryInfo warpMemoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::AllocatorWithDefaultOptions warpAllocator;

    // input node initialize
    warpInputNums = warpSession->GetInputCount();
    for (int idx = 0; idx < warpInputNums; idx++) {
        char *warpInput_name = warpSession->GetInputName(idx, warpAllocator);
        std::vector<int64_t> warpInputNodeDims = warpSession->GetInputTypeInfo(idx).GetTensorTypeAndShapeInfo().GetShape();
        size_t warpInputTensorSize = calculateProduct(warpInputNodeDims);

        warpInputNodeNames.push_back(warpInput_name);
        warpInputNodeDimsArr.push_back(warpInputNodeDims);
        warpInputTensorSizeArr.push_back(warpInputTensorSize);
    }

    frameCHW = std::make_unique<float[]>(warpInputTensorSizeArr[0]);
    clothCHW = std::make_unique<float[]>(warpInputTensorSizeArr[1]);
    edgeCHW = std::make_unique<float[]>(warpInputTensorSizeArr[2]);
    parsingCHW = std::make_unique<float[]>(warpInputTensorSizeArr[3]);

    warpInputTensors.reserve(warpInputNums);
    warpInputTensors.push_back(
            Ort::Value::CreateTensor<float>(warpMemoryInfo, frameCHW.get(), warpInputTensorSizeArr[0], warpInputNodeDimsArr[0].data(),
                                            warpInputNodeDimsArr[0].size()));
    warpInputTensors.push_back(
            Ort::Value::CreateTensor<float>(warpMemoryInfo, clothCHW.get(), warpInputTensorSizeArr[1], warpInputNodeDimsArr[1].data(),
                                            warpInputNodeDimsArr[1].size()));
    warpInputTensors.push_back(
            Ort::Value::CreateTensor<float>(warpMemoryInfo, edgeCHW.get(), warpInputTensorSizeArr[2], warpInputNodeDimsArr[2].data(),
                                            warpInputNodeDimsArr[2].size()));
    warpInputTensors.push_back(
            Ort::Value::CreateTensor<float>(warpMemoryInfo, parsingCHW.get(), warpInputTensorSizeArr[3], warpInputNodeDimsArr[3].data(),
                                            warpInputNodeDimsArr[3].size()));

    // output node initialize
    warpOutputNums = warpSession->GetOutputCount();
    for (int idx = 0; idx < warpOutputNums; idx++) {
        char *warpOutput_name = warpSession->GetOutputName(idx, warpAllocator);
        std::vector<int64_t> warpOutputNodeDims = warpSession->GetOutputTypeInfo(idx).GetTensorTypeAndShapeInfo().GetShape();
        size_t warpOutputTensorSize = calculateProduct(warpOutputNodeDims);

        warpOutputNodeNames.push_back(warpOutput_name);
        warpOutputNodeDimsArr.push_back(warpOutputNodeDims);
        warpOutputTensorSizeArr.push_back(warpOutputTensorSize);
    }

    Ort::AllocatorWithDefaultOptions genAllocator;
    Ort::MemoryInfo genMemoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // input node initialize
    genInputNums = genSession->GetInputCount();
    for (int idx = 0; idx < genInputNums; idx++) {
        char *genInput_name = genSession->GetInputName(idx, genAllocator);
        std::vector<int64_t> genInputNodeDims = genSession->GetInputTypeInfo(idx).GetTensorTypeAndShapeInfo().GetShape();
        size_t genInputTensorSize = calculateProduct(genInputNodeDims);

        genInputNodeNames.push_back(genInput_name);
        genInputNodeDimsArr.push_back(genInputNodeDims);
        genInputTensorSizeArr.push_back(genInputTensorSize);
    }

    // output node initialize
    genOutputNums = genSession->GetOutputCount();
    for (int idx = 0; idx < genOutputNums; idx++) {
        char *genOutput_name = genSession->GetOutputName(idx, genAllocator);
        std::vector<int64_t> genOutputNodeDims = genSession->GetOutputTypeInfo(idx).GetTensorTypeAndShapeInfo().GetShape();
        size_t genOutputTensorSize = calculateProduct(genOutputNodeDims);

        genOutputNodeNames.push_back(genOutput_name);
        genOutputNodeDimsArr.push_back(genOutputNodeDims);
        genOutputTensorSizeArr.push_back(genOutputTensorSize);
    }

    genInput = std::make_unique<float[]>(genInputTensorSizeArr[0]);
    modelOutput = std::make_unique<float[]>(vtoImgHeight * vtoImgWidth * 3);
    maskOutput = std::make_unique<float[]>(vtoImgHeight * vtoImgWidth);
    outputImage = std::make_unique<int[]>(vtoImgHeight * vtoImgWidth * 3);

    genInputTensor = Ort::Value::CreateTensor<float>(genMemoryInfo, genInput.get(), genInputTensorSizeArr[0], genInputNodeDimsArr[0].data(),
                                                     genInputNodeDimsArr[0].size());
}

void TotalModel::runParsing(uint8_t *pixels) {
    transform(pixels, 3, parsingImgHeight, parsingImgWidth, frameCHW512.get());

    auto outputTensors = parsingSession->Run(Ort::RunOptions{nullptr}, parsingInputNodeNames.data(), &parsingInputTensor, 1,
                                             parsingOutputNodeNames.data(), 1);
    assert(outputTensors.size() == 1 && outputTensors.front().IsTensor());

    int64_t *floatarr = outputTensors.front().GetTensorMutableData<int64_t>();
    int64ToUint8(floatarr, 512 * 288, uint8ParsingOutput.get());
}

void TotalModel::runVto(uint8_t *image, uint8_t *parsing, uint8_t *cloth, uint8_t *edge) {
    transform(image, 3, vtoImgHeight, vtoImgWidth, frameCHW.get());
    transformP(parsing, parsingSize, vtoImgHeight, vtoImgWidth, parsingCHW.get());
    transform(cloth, 3, vtoImgHeight, vtoImgWidth, clothCHW.get());
    transformL(edge, vtoImgHeight, vtoImgWidth, edgeCHW.get());

    auto warpOutputTensors = warpSession->Run(Ort::RunOptions{nullptr}, warpInputNodeNames.data(), warpInputTensors.data(), warpInputNums,
                                              warpOutputNodeNames.data(), warpOutputNums);

    pushValue(warpOutputTensors.front().GetTensorMutableData<float>(), (parsingSize + 7) * vtoImgHeight * vtoImgWidth, genInput.get());

    auto genOutputTensors = genSession->Run(Ort::RunOptions{nullptr}, genInputNodeNames.data(), &genInputTensor, genInputNums,
                                            genOutputNodeNames.data(), genOutputNums);
    float *genOutput = genOutputTensors.front().GetTensorMutableData<float>();

    postProcessingMask(genInput.get(), genOutput, modelOutput.get(), maskOutput.get());
}

TotalModel::~TotalModel() {}
