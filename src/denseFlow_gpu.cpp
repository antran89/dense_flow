#include "opencv2/video/video.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/cudaoptflow.hpp"

#include <stdio.h>
#include <iostream>
#include <string>

#ifdef USE_LDOF
#include "CTensorOpencv.h"
#include "CFilter.h"
#include "ldof.h"
#endif // USE_LDOF

using namespace std;
using namespace cv;

#ifdef USE_LDOF
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

static void convertFlowToImage(const Mat &flow_x, const Mat &flow_y, Mat &img_x, Mat &img_y,
                               double lowerBound, double higherBound) {
#define CAST(v, L, H) ((v) > (H) ? 255 : (v) < (L) ? 0 : cvRound(255*((v) - (L))/((H)-(L))))
    for (int i = 0; i < flow_x.rows; ++i) {
        for (int j = 0; j < flow_y.cols; ++j) {
            float x = flow_x.at<float>(i,j);
            float y = flow_y.at<float>(i,j);
            img_x.at<uchar>(i,j) = CAST(x, lowerBound, higherBound);
            img_y.at<uchar>(i,j) = CAST(y, lowerBound, higherBound);
        }
    }
#undef CAST
}

static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,double, const Scalar& color){
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at<Point2f>(y, x);
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            circle(cflowmap, Point(x,y), 2, color, -1);
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
    cuda::GpuMat frame_0, frame_1, d_flow;

    cuda::setDevice(device_id);
    Ptr<cuda::FarnebackOpticalFlow> alg_farn = cuda::FarnebackOpticalFlow::create();
    Ptr<cuda::OpticalFlowDual_TVL1> alg_tvl1 = cuda::OpticalFlowDual_TVL1::create();
    Ptr<cuda::BroxOpticalFlow> alg_brox = cuda::BroxOpticalFlow::create(0.197f, 50.0f, 0.8f, 10, 77, 10);

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

            int step_t = step;
            while (step_t > 1){
                capture >> frame;
                frame_index++;
                step_t--;
            }
            continue;
        }

        frame.copyTo(image);

        if (type != 3) {    // ldof processing rgb frames
            cvtColor(image, grey, CV_BGR2GRAY);
            frame_0.upload(prev_grey);
            frame_1.upload(grey);
        }


        // GPU optical flow
        switch(type){
        case 0:
            alg_farn->calc(frame_0, frame_1, d_flow);
            break;
        case 1:
            alg_tvl1->calc(frame_0, frame_1, d_flow);
            break;
        case 2:
        {
            cuda::GpuMat d_frame0f, d_frame1f;
            frame_0.convertTo(d_frame0f, CV_32F, 1.0 / 255.0);
            frame_1.convertTo(d_frame1f, CV_32F, 1.0 / 255.0);
            alg_brox->calc(d_frame0f, d_frame1f, d_flow);
            break;
        }
        case 3:     // gpu ldof computations (not in opencv flow)
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

        if (type != 3)
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

        if (type != 3)
            std::swap(prev_grey, grey);
        std::swap(prev_image, image);

        int step_t = step;
        while (step_t > 1){
            capture >> org_frame;
            frame_index++;
            step_t--;
        }
    }
    return 0;
}
