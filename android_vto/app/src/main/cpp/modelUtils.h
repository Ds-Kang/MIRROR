
#ifndef ORT_VIRTUALTRYON_MODELUTILS_H
#define ORT_VIRTUALTRYON_MODELUTILS_H

#include <opencv2/opencv.hpp>

void transform(uint8_t *input, int channel, int height, int width, float *output);

void transformL(uint8_t *input, int height, int width, float *output);

void transformP(uint8_t *input, int channel, int height, int width, float *output);

void postProcessingMask(float *gen_input, float *gen_output, float *output, float *mask);

void pushValue(float *input, int64_t size, float *output);

void int64ToUint8(int64_t *input, int64_t size, uint8_t *output);

int calculateProduct(const std::vector<int64_t> &v);

void cropTensor(float *input, int channel, int height, int width, int *crop_values, float *output);

#endif //ORT_VIRTUALTRYON_MODELUTILS_H
