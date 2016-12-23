/*
 * Copyright (C) 2016 An Tran.
 * This code is for research, please do not distribute it.
 *
 */

#include "color_flow.hpp"

#include <math.h>
#include <iostream>

// the "official" threshold - if the absolute value of either
// flow component is greater, it's considered unknown
#define UNKNOWN_FLOW_THRESH 1e3

using namespace std;

// return whether flow vector is unknown
bool unknown_flow(float u, float v) {
    return (fabs(u) >  UNKNOWN_FLOW_THRESH)
        || (fabs(v) >  UNKNOWN_FLOW_THRESH)
        || isnan(u) || isnan(v);
}

bool unknown_flow(float *f) {
    return unknown_flow(f[0], f[1]);
}

ColorFlow::ColorFlow(bool verbose)
{
    verbose_ = verbose;
    makecolorwheel();
}

void ColorFlow::setcols(int r, int g, int b, int k)
{
    colorwheel[k][0] = r;
    colorwheel[k][1] = g;
    colorwheel[k][2] = b;
}

void ColorFlow::makecolorwheel()
{
    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow
    //  than between yellow and green)
    int RY = 15;
    int YG = 6;
    int GC = 4;
    int CB = 11;
    int BM = 13;
    int MR = 6;
    ncols = RY + YG + GC + CB + BM + MR;
    //printf("ncols = %d\n", ncols);
    if (ncols > MAXCOLS) {
        cout << "Number of colors exceeds MAXCOLS" << endl;
        exit(1);
    }
    int i;
    int k = 0;
    for (i = 0; i < RY; i++) setcols(255,	   255*i/RY,	 0,	       k++);
    for (i = 0; i < YG; i++) setcols(255-255*i/YG, 255,		 0,	       k++);
    for (i = 0; i < GC; i++) setcols(0,		   255,		 255*i/GC,     k++);
    for (i = 0; i < CB; i++) setcols(0,		   255-255*i/CB, 255,	       k++);
    for (i = 0; i < BM; i++) setcols(255*i/BM,	   0,		 255,	       k++);
    for (i = 0; i < MR; i++) setcols(255,	   0,		 255-255*i/MR, k++);
}

void ColorFlow::computeColor(float fx, float fy, cv::Vec3b& pix)
{
    float rad = sqrt(fx * fx + fy * fy);
    float a = atan2(-fy, -fx) / M_PI;           // map atan to [-1, 1]
    float fk = (a + 1.0) / 2.0 * (ncols-1);     // map a in [-1, 1] to [0, ncols-1]
    int k0 = (int)fk;
    int k1 = (k0 + 1) % ncols;
    float f = fk - k0;                          // fractal portion
    //f = 0; // uncomment to see original color wheel
    for (int b = 0; b < 3; b++) {
        float col0 = colorwheel[k0][b] / 255.0;
        float col1 = colorwheel[k1][b] / 255.0;
        float col = (1 - f) * col0 + f * col1;     // average color between two bands
        if (rad <= 1)
            col = 1 - rad * (1 - col); // increase saturation with radius
        else
            col *= .75; // out of range
        pix[2 - b] = (int)(255.0 * col);   // rgb --> bgr
    }
}

void ColorFlow::MotionToColor(cv::Mat flow_x, cv::Mat flow_y, cv::Mat& flow_image, const float maxmotion)
{
    int height = flow_x.rows;
    int width = flow_x.cols;
    int x, y;
    // determine motion range:
    double maxx, maxy;
    double minx, miny;
    double minrad, maxrad;
    cv::minMaxLoc(flow_x, &minx, &maxx);
    cv::minMaxLoc(flow_y, &miny, &maxy);
    cv::Mat rad;
    cv::magnitude(flow_x, flow_y, rad);
    cv::minMaxLoc(rad, NULL, &maxrad);

    if (verbose_)
        printf("max motion: %.4f  motion range: u = %.3f .. %.3f;  v = %.3f .. %.3f\n",
            maxrad, minx, maxx, miny, maxy);


    if (maxmotion > 0) // i.e., specified on commandline
        maxrad = maxmotion;

    if (maxrad == 0) // if flow == 0 everywhere
        maxrad = 1;

    if (verbose_)
        printf("normalizing by %g\n", maxrad);

    flow_image.create(height, width, CV_8UC3);
    cv::Vec3b pix(3);

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            float fx = flow_x.at<float>(y, x);
            float fy = flow_y.at<float>(y, x);
            if (unknown_flow(fx, fy)) {
                        pix[0] = pix[1] = pix[2] = 0;
            } else {
                        computeColor(fx/maxrad, fy/maxrad, pix);
            }
            flow_image.at<cv::Vec3b>(y, x) = pix;
        }
    }
}
