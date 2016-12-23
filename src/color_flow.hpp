/*
 * Copyright (C) 2016 An Tran.
 * This code is for research, please do not distribute it.
 *
 */

#ifndef COLOR_FLOW_HPP
#define COLOR_FLOW_HPP

#include <opencv2/core.hpp>

#define MAXCOLS 60

class ColorFlow {
public:
    ColorFlow(bool verbose = false);

    // convert flows into color image.
    void MotionToColor(cv::Mat flow_x, cv::Mat flow_y, cv::Mat& flow_image, const float maxmotion = 0);

private:
    int ncols;
    int colorwheel[MAXCOLS][3];
    bool verbose_;

    // some private functions
    void setcols(int r, int g, int b, int k);
    void makecolorwheel();
    void computeColor(float fx, float fy, cv::Vec3b &pix);

};

#endif // COLOR_FLOW_HPP

