
#include <opencv2/opencv.hpp>

void transform(uint8_t *input, int channel, int height, int width,
               float *output) {   /// image[i,j,c] = img[(width*i + j )*channels +c] - > HWC : image[c,i,j] = img[(height*c +i)*width + j] - > CHW
    for (int c = 0; c < channel; c++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                float inputValue = (float) input[(width * h + w) * channel + c];
                output[(height * c + h) * width + w] = (inputValue / 255.0f - 0.5f) / 0.5f;
            }
        }
    }
    return;
}

void transformL(uint8_t *input, int height, int width,
                float *output) {   /// image[i,j,c] = img[(width*i + j )*channels +c] - > HWC : image[c,i,j] = img[(height*c +i)*width + j] - > CHW
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            float inputValue = (float) input[width * h + w];
            output[h * width + w] = inputValue / 255.0f;
        }
    }
    return;
}

void transformP(uint8_t *input, int channel, int height, int width,
                float *output) {   /// image[i,j,c] = img[(width*i + j )*channels +c] - > HWC : image[c,i,j] = img[(height*c +i)*width + j] - > CHW
    for (int c = 0; c < channel; c++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                int outC = (int) input[h * width + w];
                if (c == outC) output[c * height * width + h * width + w] = 1.0f;
                else output[c * height * width + h * width + w] = 0.0f;
            }
        }
    }
    return;
}

void postProcessingMask(float *genInput, float *genOutput, float *output, float *mask) {
    int height = 256;
    int width = 192;
    int channel = 3;

    for (int c = 0; c < channel; c++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                float pRendered = tanh(genOutput[c * height * width + h * width + w]);
                float mComposite = genOutput[3 * height * width + h * width + w];
                mComposite = 1 / (1 + exp(-mComposite)) * genInput[6 * height * width + h * width + w];
                float warpedCloth = genInput[(c + 3) * height * width + h * width + w];

                output[c * height * width + h * width + w] = warpedCloth * mComposite + pRendered * (1 - mComposite);
            }
        }
    }

    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            float maskVal = genOutput[4 * height * width + h * width + w];
            mask[h * width + w] = 1 / (1 + exp(-maskVal));
        }
    }


    return;
}

void pushValue(float *input, int64_t size, float *output) {
    for (int64_t i = 0; i < size; i++) {
        output[i] = input[i];
    }
    return;
}


void int64ToUint8(int64_t *input, int64_t size, uint8_t *output) {
    for (int64_t i = 0; i < size; i++) {
        uint8_t inputValue = (uint8_t) input[i];
        output[i] = inputValue;
    }
    return;
}

int calculateProduct(const std::vector<int64_t> &v) {
    int total = 1;
    for (auto &i : v) total *= i;
    return total;
}

void cropTensor(float *input, int channel, int height, int width, int *cropValues, float *output) {
    for (int c = 0; c < channel; c++) {
        int hIdx = 0;
        for (int h = cropValues[0]; h < cropValues[1]; h++) {
            int wIdx = 0;
            for (int w = cropValues[2]; w < cropValues[3]; w++) {
                float inputValue = input[c * height * width + h * width + w];
                output[c * (cropValues[1] - cropValues[0]) * (cropValues[3] - cropValues[2]) +
                       hIdx * (cropValues[3] - cropValues[2]) + wIdx] = inputValue;
                wIdx++;
            }
            hIdx++;
        }
    }
    return;
}
