
#ifndef ORT_VIRTUALTRYON_IMAGEUTILS_H
#define ORT_VIRTUALTRYON_IMAGEUTILS_H

#include <algorithm>
#include <random>
#include <cmath>
#include <opencv2/opencv.hpp>

void cropImage(uint8_t *input, int channel, int height, int width, const int *crop_values, uint8_t *output);

void rmBackground(uint8_t *input, int channel, int height, int width, uint8_t *parse, uint8_t *output);

void addBackground(uint8_t *input, uint8_t *back, int channel, int height, int width, uint8_t *parse, uint8_t *output);

void makeBoundingBox(uint8_t *input, int height, int width, const int *paddings, int *bb, int parsing_size);

void makeOrbSource(uint8_t *input, int channel, int height, int width, int *bb, uint8_t *output);

bool compare(cv::KeyPoint a, cv::KeyPoint b);

double calcDist(float _ptFirstPos_x, float _ptFirstPos_y, cv::Point2f _ptSecondPos);

void calcErr(float *points1, float *points2, int size, float *err, int target_height, int target_width);

void computePartialRepr(float *input_points, float *control_points, int N, int M, float *output);

void makeForwardKernel(float *target_control_partial_repr, float *target_control_points, int N, float *output);

void makeTargetCoordinate(float *target_coordinate, int target_height, int target_width);

void makeTargetCoordinateRepr(float *target_coordinate_partial_repr, float *target_coordinate, int HW, int N, float *target_coordinate_repr);

void normalize(float *input, int size, int *bb, int *diff, float *norm_out);

void normalize(float *input, int size, int *bb, float *norm_out);

void getMap(float *grid, int size_x, int size_y, float *map_xy);

void putOutputToFrame(uint8_t *frame, int channel, int height, int width, uint8_t *gen_image, uint8_t *parsing, int *bb, uint8_t *output_frame);

void makeMask(uint8_t *mask1, uint8_t *mask2, int height, int width, uint8_t *output_mask);

void makeMask(uint8_t *mask1, int height, int width, uint8_t *output_mask);

void uint8ToInt(uint8_t *input, int64_t size, int *output);

void intToUint8(int *input, int64_t size, uint8_t *output);

void floatToUint8(float *input, int64_t size, uint8_t *output);

void toImage(float *input, int *output);

void toImage(float *input, int channel, int height, int width, uint8_t *output);

void argmaxChannel(float *input, int channel, int height, int width, uint8_t *output);

#endif //ORT_VIRTUALTRYON_IMAGEUTILS_H
