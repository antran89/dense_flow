/*
 * Copyright (C) 2017 An Tran.
 *
 */

#include "helper_function.h"
#include <vector>
#include <string>
#include <utility>
#include <iostream>

#include <opencv2/xfeatures2d.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/cudaoptflow.hpp>

#define FARNEBACK_FLOW 0
#define TVL1_FLOW 1
#define BROX_FLOW 2
#define LDOF_FLOW 3

#define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255*((v) - (L))/((H)-(L))))

using namespace cv;
using namespace std;

static void convertFlowToImage(const Mat &flow_x, const Mat &flow_y, Mat &img_x, Mat &img_y,
                               double lowerBound, double higherBound) {
    for (int i = 0; i < flow_x.rows; ++i) {
        for (int j = 0; j < flow_y.cols; ++j) {
            float x = flow_x.at<float>(i,j);
            float y = flow_y.at<float>(i,j);
            img_x.at<uchar>(i,j) = CAST(x, lowerBound, higherBound);
            img_y.at<uchar>(i,j) = CAST(y, lowerBound, higherBound);
        }
    }
}

/**
 * type 0 - Farnerback optical flow
 * type 1 - TVL1 optical flow
 * type 2 - Brox optical flow
 * type 3 - LDOF optical flow
 */
int main(int argc, char** argv){
    // IO operation
    const char* keys =
    {
        "{ f  vidFile      | ex2.avi | filename of video }"
        "{ x  xFlowFile    | flow_x  | filename of flow x component }"
        "{ y  yFlowFile    | flow_y  | filename of flow x component }"
        "{ i  imgFile      |<none>| filename of flow image}"
        "{ b  bound        | 15 | specify the maximum of optical flow}"
        "{ t  type         | 1  | specify the optical flow algorithm }"
        "{ d  device_id    | 0  | set gpu id}"
        "{ s  step         | 1  | specify the step for frame sampling}"
        "{ h  new_height       | 0  | new height of images and flows}"
        "{ w  new_width        | 0  | new width of images and flows}"
    };

    CommandLineParser cmd(argc, argv, keys);
    string vidFile = cmd.get<string>("vidFile");
    string xFlowFile = cmd.get<string>("xFlowFile");
    string yFlowFile = cmd.get<string>("yFlowFile");
    string imgFile = cmd.get<string>("imgFile");
    int bound = cmd.get<int>("bound");
    int type  = cmd.get<int>("type");
    int device_id = cmd.get<int>("device_id");
    int step = cmd.get<int>("step");
    int new_height = cmd.get<int>("new_height");
    int new_width = cmd.get<int>("new_width");

    bool noImageOutput = imgFile.empty();

    VideoCapture capture(vidFile);
    if(!capture.isOpened()) {
        printf("Could not initialize capturing...%s\n", vidFile.c_str());
        return -1;
    }

    new_height = (new_height > 0) ? new_height : capture.get(CV_CAP_PROP_FRAME_HEIGHT);
    new_width = (new_width > 0) ? new_width : capture.get(CV_CAP_PROP_FRAME_WIDTH);
    cv::Size new_size(new_width, new_height);

    int frame_index = -1, result_index = 0;
    Mat image, prev_image, prev_grey, grey, org_frame, frame, flow, flow_x, flow_y, flows[2];
    cuda::GpuMat frame_0, frame_1, d_flow, d_frame_warp;
    cuda::GpuMat d_frame0f, d_frame1f, d_frame1f_warp;

    cuda::setDevice(device_id);
    Ptr<cuda::FarnebackOpticalFlow> alg_farn = cuda::FarnebackOpticalFlow::create();
    Ptr<cuda::OpticalFlowDual_TVL1> alg_tvl1 = cuda::OpticalFlowDual_TVL1::create();
    Ptr<cuda::BroxOpticalFlow> alg_brox = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);

    Ptr<cv::xfeatures2d::SURF> detector_surf = cv::xfeatures2d::SurfFeatureDetector::create(200);
    Ptr<cv::xfeatures2d::SURF> extractor_surf = cv::xfeatures2d::SurfDescriptorExtractor::create(200, 4, 3, true, true);

    // matching points in two frames
    std::vector<Point2f> prev_pts_flow, pts_flow;
    std::vector<Point2f> prev_pts_surf, pts_surf;
    std::vector<Point2f> prev_pts_all, pts_all;

    std::vector<KeyPoint> prev_kpts_surf, kpts_surf;
    Mat prev_desc_surf, desc_surf;

    while(true) {
        capture >> org_frame;
        frame_index++;
        if(org_frame.empty())
            break;

        // resize frame
        resize(org_frame, frame, new_size);

        if(frame_index == 0) {
            image.create(frame.size(), CV_8UC3);
            grey.create(frame.size(), CV_8UC1);
            prev_image.create(frame.size(), CV_8UC3);
            prev_grey.create(frame.size(), CV_8UC1);

            frame.copyTo(prev_image);
            cvtColor(prev_image, prev_grey, CV_BGR2GRAY);

            // extract surf features
            detector_surf->detect(prev_grey, prev_kpts_surf);
            extractor_surf->compute(prev_grey, prev_kpts_surf, prev_desc_surf);

            // upload prev_grey into gpu
            frame_0.upload(prev_grey);
            if (type == BROX_FLOW)
                frame_0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);

            int step_t = step;
            while (step_t > 1){
                capture >> frame;
                frame_index++;
                step_t--;
            }
            continue;
        }

        frame.copyTo(image);
        cvtColor(image, grey, CV_BGR2GRAY);

        // match surf features
        detector_surf->detect(grey, kpts_surf);
        extractor_surf->compute(grey, kpts_surf, desc_surf);
        ComputeMatch(prev_kpts_surf, kpts_surf, prev_desc_surf, desc_surf, prev_pts_surf, pts_surf);

        if (type != LDOF_FLOW) {    // ldof processing rgb frames
            frame_1.upload(grey);
        }

        // GPU optical flow
        switch (type) {
        case FARNEBACK_FLOW:
            alg_farn->calc(frame_0, frame_1, d_flow);
            break;
        case TVL1_FLOW:
            alg_tvl1->calc(frame_0, frame_1, d_flow);
            break;
        case BROX_FLOW:
        {
            frame_1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);
            alg_brox->calc(d_frame0f, d_frame1f, d_flow);
            break;
        }
        case LDOF_FLOW:     // gpu ldof computations (not in opencv flow)
#ifdef USE_LDOF
            myCalcOpticalFlowLDOF(prev_image, image, flow, true);
#else
            cout << "LDOF is not built" << endl;
#endif // USE_LDOF
            break;
        default:
            cout << "unknown type of optical flows algorithm" << endl;
            exit(-1);
            break;
        }

        if (type != LDOF_FLOW)
            d_flow.download(flow);

        // matching features based on Flow
        MatchFromFlow(prev_grey, flow, prev_pts_flow, pts_flow);
        MergeMatch(prev_pts_flow, pts_flow, prev_pts_surf, pts_surf, prev_pts_all, pts_all);
        // estimating homography
        Mat H = Mat::eye(3, 3, CV_64FC1);
        if(pts_all.size() > 50) {
            std::vector<unsigned char> match_mask;
            Mat temp = findHomography(prev_pts_all, pts_all, RANSAC, 1, match_mask);
            if(countNonZero(Mat(match_mask)) > 25)
                H = temp;
        }

        Mat H_inv = H.inv();            // invert because we warp second frame to first frame
        Mat frame_warp;                 // warp of second frame
        if (type != LDOF_FLOW)
            warpPerspective(grey, frame_warp, H_inv, frame_warp.size(), INTER_LINEAR, BORDER_REPLICATE);
        else // LDOF flow
            warpPerspective(frame, frame_warp, H_inv, frame_warp.size(), INTER_LINEAR, BORDER_REPLICATE);

        // re-estimate clean flow from frame0 and frame1_warp
        if (type != LDOF_FLOW)      // upload frame1 to gpuMat
            d_frame_warp.upload(frame_warp);

        switch (type) {
        case FARNEBACK_FLOW:
            alg_farn->calc(frame_0, d_frame_warp, d_flow);
            break;
        case TVL1_FLOW:
            alg_tvl1->calc(frame_0, d_frame_warp, d_flow);
            break;
        case BROX_FLOW:
        {
            d_frame_warp.convertTo(d_frame1f_warp, CV_32F, 1.0 / 255.0);
            alg_brox->calc(d_frame0f, d_frame1f_warp, d_flow);
            break;
        }
        case LDOF_FLOW:     // gpu ldof computations (not in opencv flow)
#ifdef USE_LDOF
            myCalcOpticalFlowLDOF(prev_image, image, flow, true);
#else
            cout << "LDOF is not built" << endl;
#endif // USE_LDOF
            break;
        default:
            cout << "unknown type of optical flows algorithm" << endl;
            exit(-1);
            break;
        }

        if (type != LDOF_FLOW)
            d_flow.download(flow);
        cv::split(flow, flows);
        flow_x = flows[0];
        flow_y = flows[1];

        // converting optical flow to images
        Mat imgX(flow_x.size(), CV_8UC1);
        Mat imgY(flow_y.size(), CV_8UC1);
        convertFlowToImage(flow_x,flow_y, imgX, imgY, -bound, bound);
        char tmp[20];
        sprintf(tmp,"_%04d.jpg",int(++result_index));

        imwrite(xFlowFile + tmp, imgX);
        imwrite(yFlowFile + tmp, imgY);

        if (!noImageOutput) {
            imwrite(imgFile + tmp, image);
        }

        // swapping
        prev_image = std::move(image);
        prev_grey = std::move(grey);
        prev_kpts_surf = std::move(kpts_surf);
        prev_desc_surf = std::move(desc_surf);
        if (type != LDOF_FLOW)
            frame_1.copyTo(frame_0);
        if (type == BROX_FLOW)
            d_frame1f.copyTo(d_frame0f);

        int step_t = step;
        while (step_t > 1){
            capture >> org_frame;
            frame_index++;
            step_t--;
        }
    }

    return 0;
}
