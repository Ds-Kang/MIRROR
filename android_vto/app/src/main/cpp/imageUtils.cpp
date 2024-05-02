
#include "imageUtils.h"

void makeBoundingBox(uint8_t *input, int height, int width, const int *paddings, int *bb, int parsingSize) {
    int allXMin = 540;
    int allXMax = 0;
    int allYMin = 960;
    int allYMax = 0;

    int xMin = 540;
    int xMax = 0;
    int yMin = 960;
    int yMax = 0;

    std::vector<uint8_t> upperHeadInt, upperInt;

    upperHeadInt = {1, 2, 3, 4};
    upperInt = {2, 3, 4};

    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            int inputValue = (int) input[h * width + w];

            if (std::find(upperHeadInt.begin(), upperHeadInt.end(), inputValue) != upperHeadInt.end()) {
                if (w - paddings[0] < allXMin && w - paddings[0] >= 0) allXMin = w - paddings[0];
                if (w + paddings[1] > allXMax && w + paddings[1] < 540) allXMax = w + paddings[1];
                if (h - paddings[2] < allYMin && h - paddings[2] >= 0) allYMin = h - paddings[2];
                if (h + paddings[3] > allYMax && h + paddings[3] < 960) allYMax = h + paddings[3];
            }
        }
    }

    if (allXMax <= allXMin) {
        bb[0] = 0;
        bb[1] = 540;
    } else {
        bb[0] = allXMin;
        bb[1] = allXMax;
    }
    if (allYMax <= allYMin) {
        bb[2] = 0;
        bb[3] = 960;
    } else {
        bb[2] = allYMin;
        bb[3] = allYMax;
    }

    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            int inputValue = (int) input[h * width + w];
            if (std::find(upperInt.begin(), upperInt.end(), inputValue) != upperInt.end()) {
                if (w - paddings[4] < xMin && w - paddings[4] >= 0 && w - paddings[4] > allXMin) xMin = w - paddings[4];
                if (w + paddings[5] > xMax && w + paddings[5] < 540 && w + paddings[5] < allXMax) xMax = w + paddings[5];
                if (h - paddings[6] < yMin && h - paddings[6] >= 0 && h - paddings[6] > allYMin) yMin = h - paddings[6];
                if (h + paddings[7] > yMax && h + paddings[7] < 960 && h + paddings[7] < allYMax) yMax = h + paddings[7];
            }

        }
    }

    if (xMax <= xMin) {
        bb[4] = bb[0];
        bb[5] = bb[1];
    } else {
        bb[4] = xMin;
        bb[5] = xMax;
    }
    if (yMax <= yMin) {
        bb[6] = bb[2];
        bb[7] = bb[3];
    } else {
        bb[6] = yMin;
        bb[7] = yMax;
    }

    return;
}

void cropImage(uint8_t *input, int channel, int height, int width, const int *cropValues, uint8_t *output) {
    int hIdx = 0;
    for (int h = cropValues[2]; h < cropValues[3]; h++) {
        int wIdx = 0;
        for (int w = cropValues[0]; w < cropValues[1]; w++) {
            for (int c = 0; c < channel; c++) {
                int idx = h * width * channel + w * channel + c;
                uint8_t inputValue = input[h * width * channel + w * channel + c];
                output[hIdx * (cropValues[1] - cropValues[0]) * channel + wIdx * channel + c] = inputValue;
            }
            wIdx++;
        }
        hIdx++;
    }
    return;
}

void rmBackground(uint8_t *input, int channel, int height, int width, uint8_t *parse, uint8_t *output) {
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            uint8_t parseValue = parse[h * width + w];
            for (int c = 0; c < channel; c++) {
                int inputValue = input[h * width * channel + w * channel + c];
                if (parseValue != 0) {
                    output[h * width * channel + w * channel + c] = inputValue;
                } else {
                    output[h * width * channel + w * channel + c] = 255;
                }

            }
        }
    }
}


void addBackground(uint8_t *input, uint8_t *back, int channel, int height, int width, uint8_t *parse, uint8_t *output) {
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            uint8_t parseValue = parse[h * width + w];
            for (int c = 0; c < channel; c++) {
                int inputValue = input[h * width * channel + w * channel + c];
                int backValue = back[h * width * channel + w * channel + c];
                if (parseValue != 0) {
                    output[h * width * channel + w * channel + c] = inputValue;
                } else {
                    output[h * width * channel + w * channel + c] = backValue;
                }

            }
        }
    }
}

void makeOrbSource(uint8_t *input, int channel, int height, int width, int *bb, uint8_t *output) {
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            for (int c = 0; c < channel; c++) {
                if ((w >= bb[4]) && (w < bb[5]) && (h >= bb[6]) && (h < bb[7])) {
                    output[h * width * channel + w * channel + c] = input[h * width * channel + w * channel + c];
                } else {
                    output[h * width * channel + w * channel + c] = 255;
                }
            }
        }
    }

    return;
}

bool compare(cv::KeyPoint a, cv::KeyPoint b) {
    return a.response > b.response;
}

double calcDist(float ptFirstPosX, float ptFirstPosY, cv::Point2f ptSecondPos) {
    return sqrt(((ptFirstPosX - ptSecondPos.x) * (ptFirstPosX - ptSecondPos.x)) + ((ptFirstPosY - ptSecondPos.y) * (ptFirstPosY - ptSecondPos.y)));
}


void computePartialRepr(float *inputPoints, float *controlPoints, int N, int M, float *output) {
    for (int n = 0; n < N; n++) {
        for (int m = 0; m < M; m++) {
            float xDiff = inputPoints[2 * n] - controlPoints[2 * m];
            float yDiff = inputPoints[2 * n + 1] - controlPoints[2 * m + 1];
            float diffSquare = xDiff * xDiff + yDiff * yDiff;
            if (diffSquare == 0) output[n * M + m] = 0;
            else output[n * M + m] = 0.5 * diffSquare * log(diffSquare);
        }
    }
}

void makeForwardKernel(float *targetControlPartialRepr, float *targetControlPoints, int N, float *output) {
    for (int n = 0; n < N + 3; n++) {
        for (int m = 0; m < N + 3; m++) {
            if ((n < N) && (m < N)) output[n * (N + 3) + m] = targetControlPartialRepr[n * N + m];
            else if ((n < N) && (m == N)) output[n * (N + 3) + m] = 1;
            else if ((n == N) && (m < N)) output[n * (N + 3) + m] = 1;
            else if ((n < N) && (m > N)) output[n * (N + 3) + m] = targetControlPoints[n * 2 + (m - N - 1)];
            else if ((n > N) && (m < N)) output[n * (N + 3) + m] = targetControlPoints[m * 2 + (n - N - 1)];
            else output[n * (N + 3) + m] = 0;
        }
    }
}

void makeTargetCoordinate(float *targetCoordinate, int targetHeight, int targetWidth) {
    for (int h = 0; h < targetHeight; h++) {
        for (int w = 0; w < targetWidth; w++) {
            float x = 2.0 * (float) w / (targetWidth - 1) - 1;
            float y = 2.0 * (float) h / (targetHeight - 1) - 1;

            targetCoordinate[(h * targetWidth + w) * 2] = x;
            targetCoordinate[(h * targetWidth + w) * 2 + 1] = y;
        }
    }
}


void makeTargetCoordinateRepr(float *targetCoordinatePartialRepr, float *targetCoordinate, int HW, int N, float *targetCoordinateRepr) {
    for (int i = 0; i < HW; i++) {
        for (int j = 0; j < N + 3; j++) {
            if (j < N) targetCoordinateRepr[i * (N + 3) + j] = targetCoordinatePartialRepr[i * N + j];
            else if (j > N) targetCoordinateRepr[i * (N + 3) + j] = targetCoordinate[i * 2 + (j - N - 1)];
            else targetCoordinateRepr[i * (N + 3) + j] = 1;
        }
    }
}


void normalize(float *input, int size, int *bb, int *diff, float *norm_out) {
    for (int i = 0; i < size; i += 2) {
        float normX = 2 * (input[i] - (float) bb[4] - (float) diff[0]) / ((float) bb[5] - (float) bb[4]) - 1;
        float normY = 2 * (input[i + 1] - (float) bb[6] - (float) diff[1]) / ((float) bb[7] - (float) bb[6]) - 1;
        norm_out[i] = normX;
        norm_out[i + 1] = normY;
    }
}

void normalize(float *input, int size, int *bb, float *norm_out) {
    for (int i = 0; i < size; i += 2) {
        float normX = 2 * (input[i] - (float) bb[4]) / ((float) bb[5] - (float) bb[4]) - 1;
        float normY = 2 * (input[i + 1] - (float) bb[6]) / ((float) bb[7] - (float) bb[6]) - 1;
        norm_out[i] = normX;
        norm_out[i + 1] = normY;
    }
}

void getMap(float *grid, int sizeX, int sizeY, float *mapXy) {
    for (int i = 0; i < sizeX * sizeY; i++) {
        mapXy[i] = (grid[2 * i] + 1) * 0.5 * sizeX;
        mapXy[i + sizeX * sizeY] = (grid[2 * i + 1] + 1) * 0.5 * sizeY;
    }
}

// put generated frame to input frame with partial frame
void putOutputToFrame(uint8_t *frame, int channel, int height, int width, uint8_t *genImage, uint8_t *parsing, int *bb, uint8_t *outputFrame) {
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            for (int c = 0; c < channel; c++) {
                if ((w >= bb[4]) && (w < bb[5]) && (h >= bb[6]) && (h < bb[7])) {
                    if (parsing[(h - bb[6]) * (bb[5] - bb[4]) + w - bb[4]] != 0)
                        outputFrame[h * width * channel + w * channel + c] = genImage[(h - bb[6]) * (bb[5] - bb[4]) * channel +
                                                                                      (w - bb[4]) * channel + c];
                    else outputFrame[h * width * channel + w * channel + c] = frame[h * width * channel + w * channel + c];
                } else {
                    outputFrame[h * width * channel + w * channel + c] = frame[h * width * channel + w * channel + c];
                }
            }
        }
    }

    return;
}


void makeMask(uint8_t *mask1, uint8_t *mask2, int height, int width, uint8_t *outputMask) {
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            if (mask1[h * width + w] != 0 || mask2[h * width + w] != 0) {
                outputMask[h * width + w] = 1;
            } else {
                outputMask[h * width + w] = 0;
            }
        }
    }

    return;
}


void makeMask(uint8_t *mask1, int height, int width, uint8_t *outputMask) {
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            if (mask1[h * width + w] != 0) {
                outputMask[h * width + w] = 1;
            } else {
                outputMask[h * width + w] = 0;
            }
        }
    }

    return;
}


void uint8ToInt(uint8_t *input, int64_t size, int *output) {
    for (int64_t i = 0; i < size; i++) {
        int inputValue = (int) input[i];
        output[i] = inputValue;
    }

    return;
}

void intToUint8(int *input, int64_t size, uint8_t *output) {
    for (int64_t i = 0; i < size; i++) {
        uint8_t inputValue = (uint8_t) input[i];
        output[i] = inputValue;
    }
    return;
}

void floatToUint8(float *input, int64_t size, uint8_t *output) {
    for (int64_t i = 0; i < size; i++) {
        float floatIn = input[i];

        uint8_t inputValue = floatIn > 0.5 ? 1 : 0;
        output[i] = inputValue;

    }
    return;
}


void toImage(float *input, int *output) {
    int height = 256;
    int width = 192;
    int channel = 3;

    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            for (int c = 0; c < channel; c++) {
                float out = (input[c * height * width + h * width + w] + 1) * 0.5 * 255;
                if (out > 255) out = 255;
                else if (out < 0) out = 0;
                output[h * width * channel + w * channel + c] = (int) out;
            }
        }
    }

    return;
}

void toImage(float *input, int channel, int height, int width, uint8_t *output) {
    for (int h = 0; h < height; h++) {
        for (int w = 0; w < width; w++) {
            for (int c = 0; c < channel; c++) {
                float out = (input[c * height * width + h * width + w] + 1) * 0.5 * 255;
                if (out > 255) out = 255;
                else if (out < 0) out = 0;
                output[h * width * channel + w * channel + c] = (uint8_t) out;
            }
        }
    }

    return;
}


void argmaxChannel(float *input, int channel, int height, int width, uint8_t *output) {
    float *maxValues = new float[height * width];
    std::fill_n(maxValues, height * width, -1.0F);

    for (int c = 0; c < channel; c++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                float val = input[c * height * width + h * width + w];

                if (val > maxValues[h * width + w]) {
                    maxValues[h * width + w] = val;
                    output[h * width + w] = (uint8_t) c;
                }
            }
        }
    }

    delete[] maxValues;

    return;
}


void calcErr(float *points1, float *points2, int size, float *err, int targetHeight, int targetWidth) {
    err[0] = 0.0;
    if (size != 0) {
        for (int i = 0; i < size; i += 2) {
            err[0] += sqrt(
                    (points1[i] - points2[i]) * (points1[i] - points2[i]) + (points1[i + 1] - points2[i + 1]) * (points1[i + 1] - points2[i + 1]));
        }
        err[0] /= (size / 2);
    } else {
        err[0] = 1000.0;
    }
}