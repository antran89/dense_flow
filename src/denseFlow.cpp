#include "opencv2/video/video.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <stdio.h>
#include <iostream>

#ifdef USE_LDOF
#include "CTensorOpencv.h"
#include "CFilter.h"
#include "ldof.h"
#endif // USE_LDOF

using namespace std;
using namespace cv;

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
#ifdef USE_LDOF
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
#endif // USE_LDOF

static void convertFlowToImage(const Mat &flow_x, const Mat &flow_y, Mat &img_x, Mat &img_y, double lowerBound, double higherBound) {
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

static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step,double, const Scalar& color)
{
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
 * type 1 - LDOF optical flow [default]
 */
int main(int argc, char** argv)
{
	// IO operation

    const char* keys =
    {
        "{ f  vidFile      | ex2.avi | filename of video }"
        "{ x  xFlowFile    | flow_x  | filename of flow x component }"
        "{ y  yFlowFile    | flow_y  | filename of flow x component }"
        "{ i  imgFile      || filename of flow image}"
        "{ b  bound        | 15 | specify the maximum of optical flow}"
        "{ t  type         | 1  | specify the optical flow algorithm }"
        "{ d  device_id    | 0  | set gpu id}"
        "{ s  step         | 1  | specify the step for frame sampling}"
    };

    CommandLineParser cmd(argc, argv, keys);
    string vidFile = cmd.get<string>("vidFile");
    string xFlowFile = cmd.get<string>("xFlowFile");
    string yFlowFile = cmd.get<string>("yFlowFile");
    string imgFile = cmd.get<string>("imgFile");
    int bound = cmd.get<int>("bound");
    int type  = cmd.get<int>("type");
    int step = cmd.get<int>("step");

    bool noImageOutput = imgFile.empty();

	VideoCapture capture(vidFile);
	if(!capture.isOpened()) {
		printf("Could not initialize capturing..\n");
		return -1;
	}

	int frame_num = 0;
	Mat image, prev_image, prev_grey, grey, frame, flow, cflow;

	while(true) {
		capture >> frame;
		if(frame.empty())
			break;
      
		if(frame_num == 0) {
			image.create(frame.size(), CV_8UC3);
			grey.create(frame.size(), CV_8UC1);
			prev_image.create(frame.size(), CV_8UC3);
			prev_grey.create(frame.size(), CV_8UC1);

			frame.copyTo(prev_image);
			cvtColor(prev_image, prev_grey, CV_BGR2GRAY);

			frame_num++;

            int step_t = step;
            while (step_t > 1){
                capture >> frame;
                step_t--;
            }
			continue;
		}

		frame.copyTo(image);
        if (type == 0)
            cvtColor(image, grey, CV_BGR2GRAY);

        switch (type) {
        case 0:
            // calcOpticalFlowFarneback(prev_grey,grey,flow,0.5, 3, 15, 3, 5, 1.2, 0 );
            calcOpticalFlowFarneback(prev_grey, grey, flow, 0.702, 5, 10, 2, 7, 1.5, cv::OPTFLOW_FARNEBACK_GAUSSIAN );
            break;
        case 1:
#ifdef USE_LDOF
            myCpuCalcOpticalFlowLDOF(prev_image, image, flow);
#else
            cout << "LDOF is not built" << endl;
#endif // USE_LDOF
            break;
        }
		
		// prev_image.copyTo(cflow);
		// drawOptFlowMap(flow, cflow, 12, 1.5, Scalar(0, 255, 0));

		Mat flows[2];
		split(flow,flows);
		Mat imgX(flows[0].size(),CV_8UC1);
		Mat imgY(flows[0].size(),CV_8UC1);
		convertFlowToImage(flows[0],flows[1], imgX, imgY, -bound, bound);
		char tmp[20];
		sprintf(tmp,"_%04d.jpg",int(frame_num));

        Mat imgX_, imgY_, image_;
        resize(imgX,imgX_, cv::Size(340,256));
        resize(imgY,imgY_, cv::Size(340,256));

        imwrite(xFlowFile + tmp, imgX_);
        imwrite(yFlowFile + tmp, imgY_);

        if (!noImageOutput) {
            resize(image,image_,cv::Size(340,256));
            imwrite(imgFile + tmp, image_);
        }

        if (type == 0)
            std::swap(prev_grey, grey);
		std::swap(prev_image, image);
		frame_num = frame_num + 1;

        int step_t = step;
        while (step_t > 1){
            capture >> frame;
            step_t--;
        }
	}
	return 0;
}
