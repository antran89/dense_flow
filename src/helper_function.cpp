/*
 * Copyright (C) 2016 An Tran.
 * This code is for research, please do not distribute it.
 *
 */

#include "helper_function.h"
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv;
using namespace std;

template <typename T>
inline T mapVal(T x, T a, T b, T c, T d)
{
    x = ::max(::min(x, b), a);
    return c + (d-c) * (x-a) / (b-a);
}

/**
 * @brief To visual optical flow in an image.
 * @param u horizontal flow u
 * @param v vertical flow v
 * @param dst
 */
void colorizeFlow(const Mat u, const Mat v, Mat &dst)
{
    double uMin, uMax;
    cv::minMaxLoc(u, &uMin, &uMax, 0, 0);
    double vMin, vMax;
    cv::minMaxLoc(v, &vMin, &vMax, 0, 0);
    uMin = ::abs(uMin); uMax = ::abs(uMax);
    vMin = ::abs(vMin); vMax = ::abs(vMax);
    float dMax = static_cast<float>(::max(::max(uMin, uMax), ::max(vMin, vMax)));

    dst.create(u.size(), CV_8UC3);
    for (int y = 0; y < u.rows; ++y)
    {
        for (int x = 0; x < u.cols; ++x)
        {
            dst.at<uchar>(y,3*x) = 0;
            dst.at<uchar>(y,3*x+1) = (uchar)mapVal(-v.at<float>(y,x), -dMax, dMax, 0.f, 255.f);
            dst.at<uchar>(y,3*x+2) = (uchar)mapVal(u.at<float>(y,x), -dMax, dMax, 0.f, 255.f);
        }
    }
}

/**
 * @brief drawOpticalFlow2
 *
 * @param flow
 * @param dst
 */
void drawOpticalFlow2(const Mat flow, Mat& dst)
{
    Mat flows[2];
    cv::split(flow, flows);
    colorizeFlow(flows[0], flows[1], dst);
}

inline bool isFlowCorrect(Point2f u)
{
    return !cvIsNaN(u.x) && !cvIsNaN(u.y) && fabs(u.x) < 1e9 && fabs(u.y) < 1e9;
}

static Vec3b computeColor(float fx, float fy)
{
    static bool first = true;

    // relative lengths of color transitions:
    // these are chosen based on perceptual similarity
    // (e.g. one can distinguish more shades between red and yellow
    //  than between yellow and green)
    const int RY = 15;
    const int YG = 6;
    const int GC = 4;
    const int CB = 11;
    const int BM = 13;
    const int MR = 6;
    const int NCOLS = RY + YG + GC + CB + BM + MR;
    static Vec3i colorWheel[NCOLS];

    if (first)
    {
        int k = 0;

        for (int i = 0; i < RY; ++i, ++k)
            colorWheel[k] = Vec3i(255, 255 * i / RY, 0);

        for (int i = 0; i < YG; ++i, ++k)
            colorWheel[k] = Vec3i(255 - 255 * i / YG, 255, 0);

        for (int i = 0; i < GC; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255, 255 * i / GC);

        for (int i = 0; i < CB; ++i, ++k)
            colorWheel[k] = Vec3i(0, 255 - 255 * i / CB, 255);

        for (int i = 0; i < BM; ++i, ++k)
            colorWheel[k] = Vec3i(255 * i / BM, 0, 255);

        for (int i = 0; i < MR; ++i, ++k)
            colorWheel[k] = Vec3i(255, 0, 255 - 255 * i / MR);

        first = false;
    }

    const float rad = sqrt(fx * fx + fy * fy);
    const float a = atan2(-fy, -fx) / (float)CV_PI;

    const float fk = (a + 1.0f) / 2.0f * (NCOLS - 1);
    const int k0 = static_cast<int>(fk);
    const int k1 = (k0 + 1) % NCOLS;
    const float f = fk - k0;

    Vec3b pix;

    for (int b = 0; b < 3; b++)
    {
        const float col0 = colorWheel[k0][b] / 255.f;
        const float col1 = colorWheel[k1][b] / 255.f;

        float col = (1 - f) * col0 + f * col1;

        if (rad <= 1)
            col = 1 - rad * (1 - col); // increase saturation with radius
        else
            col *= .75; // out of range

        pix[2 - b] = static_cast<uchar>(255.f * col);
    }

    return pix;
}

/**
 * @brief To draw optical flow on color images
 *
 * @param flow
 * @param dst
 * @param maxmotion
 */
void drawOpticalFlow(const Mat_<Point2f>& flow, Mat& dst, float maxmotion)
{
    dst.create(flow.size(), CV_8UC3);
    dst.setTo(Scalar::all(0));

    // determine motion range:
    float maxrad = maxmotion;

    if (maxmotion <= 0)
    {
        maxrad = 1;
        for (int y = 0; y < flow.rows; ++y)
        {
            for (int x = 0; x < flow.cols; ++x)
            {
                Point2f u = flow(y, x);

                if (!isFlowCorrect(u))
                    continue;

                maxrad = max(maxrad, sqrt(u.x * u.x + u.y * u.y));
            }
        }
    }

    for (int y = 0; y < flow.rows; ++y)
    {
        for (int x = 0; x < flow.cols; ++x)
        {
            Point2f u = flow(y, x);

            if (isFlowCorrect(u))
                dst.at<Vec3b>(y, x) = computeColor(u.x / maxrad, u.y / maxrad);
        }
    }
}

#ifdef USE_LDOF
/**
 * @brief To compute optical flow using LDOF from Thomas Brox group.
 * The most accurate optical flow, this function only use cpu version
 * of the library ldof.so. This function does not use gpu to run multi-thread
 * on CPUs.
 *
 * @param prev_img previous 3-channel frame
 * @param img current 3-channel frame
 * @param flow the result flow
 */
void myCpuCalcOpticalFlowLDOF(const Mat prev_img, const Mat img, Mat &flow)
{
    CTensorOpencv<float> prevFrame;
    CTensorOpencv<float> nextFrame;

    prevFrame.copyFromMat(prev_img);
    nextFrame.copyFromMat(img);

    NFilter::recursiveSmoothX( prevFrame, 0.8f );
    NFilter::recursiveSmoothY( prevFrame, 0.8f );

    NFilter::recursiveSmoothX( nextFrame, 0.8f );
    NFilter::recursiveSmoothY( nextFrame, 0.8f );

    CTensorOpencv<float> fflow;

    //ldof( prevFrame, nextFrame, fflow, bflow );

    runFlow(prevFrame, nextFrame, fflow);

    fflow.copyToMat( flow );
}

/**
 * @brief myCalcOpticalFlowLDOF
 *
 * @param prev_img
 * @param img
 * @param flow
 * @param useGPU
 */
void myCalcOpticalFlowLDOF(const Mat prev_img, const Mat img, Mat &flow, bool useGPU)
{
    CTensorOpencv<float> prevFrame;
    CTensorOpencv<float> nextFrame;

    prevFrame.copyFromMat(prev_img);
    nextFrame.copyFromMat(img);

    NFilter::recursiveSmoothX( prevFrame, 0.8f );
    NFilter::recursiveSmoothY( prevFrame, 0.8f );

    NFilter::recursiveSmoothX( nextFrame, 0.8f );
    NFilter::recursiveSmoothY( nextFrame, 0.8f );

    CTensorOpencv<float> fflow;

    //ldof( prevFrame, nextFrame, fflow, bflow );

    if (useGPU)
        runFlowGPU( prevFrame, nextFrame, fflow, 0.8f, 30.f, 300.f, 5.f, 0.95f, 5, 25);
    else runFlow(prevFrame, nextFrame, fflow);

    fflow.copyToMat( flow );
}
#endif // USE_LDOF

void ComputeMatch(const std::vector<KeyPoint>& prev_kpts, const std::vector<KeyPoint>& kpts,
                                  const Mat& prev_desc, const Mat& desc, std::vector<Point2f>& prev_pts, std::vector<Point2f>& pts)
{
        prev_pts.clear();
        pts.clear();

        if(prev_kpts.size() == 0 || kpts.size() == 0)
                return;

        //Mat mask = windowedMatchingMask(kpts, prev_kpts, 25, 25);

        BFMatcher desc_matcher(NORM_L2);
        std::vector<DMatch> matches;

        desc_matcher.match(desc, prev_desc, matches);

        prev_pts.reserve(matches.size());
        pts.reserve(matches.size());

        for(size_t i = 0; i < matches.size(); i++) {
                const DMatch& dmatch = matches[i];
                // get the point pairs that are successfully matched
                prev_pts.push_back(prev_kpts[dmatch.trainIdx].pt);
                pts.push_back(kpts[dmatch.queryIdx].pt);
        }

        return;
}

void MergeMatch(const std::vector<Point2f>& prev_pts1, const std::vector<Point2f>& pts1,
                                const std::vector<Point2f>& prev_pts2, const std::vector<Point2f>& pts2,
                                std::vector<Point2f>& prev_pts_all, std::vector<Point2f>& pts_all)
{
        prev_pts_all.clear();
        prev_pts_all.reserve(prev_pts1.size() + prev_pts2.size());

        pts_all.clear();
        pts_all.reserve(pts1.size() + pts2.size());

        for(size_t i = 0; i < prev_pts1.size(); i++) {
                prev_pts_all.push_back(prev_pts1[i]);
                pts_all.push_back(pts1[i]);
        }

        for(size_t i = 0; i < prev_pts2.size(); i++) {
                prev_pts_all.push_back(prev_pts2[i]);
                pts_all.push_back(pts2[i]);
        }

        return;
}

void MatchFromFlow(const Mat& prev_grey, const Mat& flow, std::vector<Point2f>& prev_pts, std::vector<Point2f>& pts)
{
        int width = prev_grey.cols;
        int height = prev_grey.rows;
        prev_pts.clear();
        pts.clear();

        const int MAX_COUNT = 1000;
        goodFeaturesToTrack(prev_grey, prev_pts, MAX_COUNT, 0.001, 3);

        if(prev_pts.size() == 0)
                return;

        for(int i = 0; i < prev_pts.size(); i++) {
                int x = std::min<int>(std::max<int>(cvRound(prev_pts[i].x), 0), width-1);
                int y = std::min<int>(std::max<int>(cvRound(prev_pts[i].y), 0), height-1);

                const float* f = flow.ptr<float>(y);
                pts.push_back(Point2f(x+f[2*x], y+f[2*x+1]));
        }
}
