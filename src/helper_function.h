/*
 * Copyright (C) 2016 An Tran.
 * This code is for research, please do not distribute it.
 *
 */

#ifndef HELPER_FUNCTION_H
#define HELPER_FUNCTION_H

#ifdef USE_LDOF
#include "CTensorOpencv.h"
#include "CFilter.h"
#include "ldof.h"
#endif // USE_LDOF

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

/**
 * @brief To draw optical flow on color images
 *
 * @param flow
 * @param dst
 * @param maxmotion
 */
void drawOpticalFlow(const cv::Mat_<cv::Point2f>& flow, cv::Mat& dst, float maxmotion = -1);

/**
 * @brief To visual optical flow in an image.
 * @param u horizontal flow u
 * @param v vertical flow v
 * @param dst
 */
void drawOpticalFlow2(const cv::Mat flow, cv::Mat& dst);

#ifdef USE_LDOF
/**
 * @brief myCalcOpticalFlowLDOF
 *        Calculate LDOF optical flow using CPU or GPU.
 * @param prev_img
 * @param img
 * @param flow
 * @param useGPU
 */
void myCalcOpticalFlowLDOF(const cv::Mat prev_img, const cv::Mat img, cv::Mat &flow, bool useGPU = false);

/**
 * @brief myCpuCalcOpticalFlowLDOF
 * This function is only running on CPU to avoid memory allocation on GPU.
 * @param prev_img
 * @param img
 * @param flow
 */
void myCpuCalcOpticalFlowLDOF(const cv::Mat prev_img, const cv::Mat img, cv::Mat &flow);
#endif // USE_LDOF


void ComputeMatch(const std::vector<cv::KeyPoint>& prev_kpts, const std::vector<cv::KeyPoint>& kpts,
                                  const cv::Mat& prev_desc, const cv::Mat& desc, std::vector<cv::Point2f>& prev_pts, std::vector<cv::Point2f>& pts);

void MergeMatch(const std::vector<cv::Point2f>& prev_pts1, const std::vector<cv::Point2f>& pts1,
                                const std::vector<cv::Point2f>& prev_pts2, const std::vector<cv::Point2f>& pts2,
                                std::vector<cv::Point2f>& prev_pts_all, std::vector<cv::Point2f>& pts_all);

void MatchFromFlow(const cv::Mat& prev_grey, const cv::Mat& flow, std::vector<cv::Point2f>& prev_pts, std::vector<cv::Point2f>& pts);

#endif // HELPER_FUNCTION_H

