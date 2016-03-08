//
//  main.cpp
//  DotTracker
//
//  Created by Terna Kpamber on 2/4/16.
//  Copyright © 2016 Terna Kpamber. All rights reserved.
//

#include <iostream>
using namespace std;
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <fstream>
using namespace cv;

#define MAX_SKEW 1
#define MAX_ROTATION 13.0
#define MAX_TRANSLATION 5.0
#define IMG_SIZE 28
#define THRESH 100
#define NOISE 30
#define NOISE_SD 10
RNG rng;

void thresh(Mat& input){
	adaptiveThreshold(input, input, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 7, -10);
	morphologyEx(input, input, MORPH_CLOSE, Mat::ones(3, 3, CV_8UC1));
}

void addTranslation(Mat& input){
	double tx = (rng.uniform(0., 1.) - 0.5) * 2 * MAX_TRANSLATION;
	double ty = (rng.uniform(0., 1.) - 0.5) * 2 * MAX_TRANSLATION;
	Mat trans_mat = (Mat_<double>(2,3) << 1, 0, tx, 0, 1, ty);
	warpAffine(input,input,trans_mat,input.size());
}

void addRotation(Mat& input){
	double angle = (rng.uniform(0., 1.) - 0.5) * 2 * MAX_ROTATION;
	Size size = input.size();
	Mat transform = getRotationMatrix2D(Point(size.width/2.0, size.height/2.0), angle, 1);
	warpAffine(input,input,transform,input.size());
}

void addSkew(Mat& input){
	double skew = (rng.uniform(0., 1.) - 0.5) * 2 * MAX_SKEW;
	Mat trans_mat = (Mat_<double>(2,3) << 1, skew/10, 0, 0, 1, 0);
	warpAffine(input,input,trans_mat,input.size());
}

void addNoise(Mat& input){
	
	Mat gaussian_noise = input.clone();
	randn(gaussian_noise,NOISE,NOISE_SD);
	input = input+gaussian_noise;
}

int main(int argc, const char * argv[]) {
	

	string root = "/Users/Terna/Desktop/USCSpring16/EE586/";
	string destRoot = "/Users/Terna/Desktop/MACHINELEARNING/Alphabets/AlphabetDataset/";
	string source = "data/";
	string dest = "thresh/";
	string alphabets = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";

	for(int l = 0 ; l < 26; l++){
		for(int i = 0; i < 512; i++){
			
			string sourcePath = root+source+alphabets[l]+"/"+to_string(i+1)+".jpg";
			string path1 = destRoot+dest+alphabets[l]+"/"+to_string(i+1)+".jpg";
			string path2 = destRoot+dest+alphabets[l]+"/"+to_string(i+1+512)+".jpg";
			string path3 = destRoot+dest+alphabets[l]+"/"+to_string(i+1+(2*512))+".jpg";
			string path4 = destRoot+dest+alphabets[l]+"/"+to_string(i+1+(3*512))+".jpg";
			string path5 = destRoot+dest+alphabets[l]+"/"+to_string(i+1+(4*512))+".jpg";
			string path6 = destRoot+dest+alphabets[l]+"/"+to_string(i+1+(5*512))+".jpg";
			string path7 = destRoot+dest+alphabets[l]+"/"+to_string(i+1+(6*512))+".jpg";
			string path8 = destRoot+dest+alphabets[l]+"/"+to_string(i+1+(7*512))+".jpg";
			string path9 = destRoot+dest+alphabets[l]+"/"+to_string(i+1+(8*512))+".jpg";
			
			Mat original = imread(sourcePath);
			cvtColor(original, original, COLOR_RGB2GRAY);
			resize(original, original, Size(IMG_SIZE, IMG_SIZE));
			
			Mat clone = original.clone();
			adaptiveThreshold(clone, clone, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 15, 20);
			
			Mat noisy = original.clone();
			Mat skewed = clone.clone();
			Mat translated = clone.clone();
			Mat rotated = clone.clone();
			
			addNoise(noisy);
			adaptiveThreshold(noisy, noisy, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 15, 20);
			
			addRotation(rotated);
			addTranslation(translated);
			addSkew(skewed);
			
			Mat rotated_translated = rotated.clone();
			Mat noisy_translated = noisy.clone();
			Mat noisy_rotated = noisy.clone();
			
			addTranslation(rotated_translated);
			addTranslation(noisy_translated);
			addRotation(noisy_rotated);
			
			
			imwrite(path1, clone);
			imwrite(path2, noisy);
			imwrite(path3, skewed);
			imwrite(path4, translated);
			imwrite(path5, rotated);
			imwrite(path6, rotated_translated);
			imwrite(path7, noisy_translated);
			imwrite(path8, noisy_rotated);
			
//			namedWindow("frame");
//			imshow("frame", noisy_rotated);
//			waitKey();

		}
	}

	
	
	cout << "Complete" << endl;

}


