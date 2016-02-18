//
//  dataset.cpp
//  Alphabets Project
//  
//  Use this to generate the dataset.
//  Modify fileName appropriately.
//
//  Created by Terna Kpamber on 2/9/16.
//  Copyright © 2016 Terna Kpamber. All rights reserved.
//

#include <iostream>
using namespace std;
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgcodecs.hpp>
#include <fstream>
using namespace cv;

int main(int argc, const char * argv[]) {
	
	string fileName = "/Users/Terna/Desktop/USCSpring16/EE586/data/Z/";
	string ext = ".jpg";
	
	Mat frame, roi;
	Point origin = Point(320-32, 240-32);
	Point end = Point(320+32, 240+32);

	// Initialize camera
	VideoCapture cap(0);
	cap.set(CAP_PROP_FRAME_HEIGHT, 480);
	cap.set(CAP_PROP_FRAME_WIDTH, 640);
	cap.set(CAP_PROP_FORMAT, CV_32F);
	cap.set(CAP_PROP_FPS, 60);

	namedWindow("frame");
	
	int numFrames = 512; //number of data samples

	int count = 0;
	int setup = 0;

	// Give you time to position object inside red rectangle.
	while(setup < 300){
		if(cap.grab()){
			setup++;
			cap.retrieve(frame);
			flip(frame, frame, 1);
			
			rectangle(frame, origin, end, Scalar(0, 0, 255));
			imshow("frame", frame);
		}
	}
	
	// Rectangle will turn green to indicate it's recrding.
	while(count < numFrames){
		if(cap.grab()){
			count++;
			cap.retrieve(frame);
			roi = frame(Rect(origin, end)).clone();
			flip(frame, frame, 1);
			rectangle(frame, origin, end, Scalar(0, 255, 0));
			
			cvtColor(roi, roi, COLOR_RGB2GRAY);
			imshow("frame", frame);

			// write frame to image
			imwrite(fileName+to_string(count)+ext, roi);
		}
	}
	
	cap.release();
	
}