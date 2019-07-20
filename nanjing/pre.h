#pragma once
#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include<stack>
#include<deque>
#include <io.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <math.h>
#include <assert.h>   
#include <time.h>
#include <cxcore.h>  
#include <highgui.h>  
#include <vector>
#define NUMSIZE 2  
#define GAUSSKERN 3.5   //调节高斯核的大小的dim = 2 * kern * xita + 1
#define PI 3.14159265358979323846  

//Sigma of base image -- See D.L.'s paper.  
#define INITSIGMA 0.5  
//Sigma of each octave -- See D.L.'s paper.  
#define SIGMA sqrt(3)//1.6//  

//Number of scales per octave.  See D.L.'s paper.  
#define SCALESPEROCTAVE 2       //每阶梯最后可计算出极值点的层数  
#define MAXOCTAVES 4  


#define CONTRAST_THRESHOLD   0.02  
#define CURVATURE_THRESHOLD  10.0  
#define DOUBLE_BASE_IMAGE_SIZE 1  
#define peakRelThresh 0.8  
#define LEN 128  

using namespace cv;
using namespace std;

struct ImageLevel {        /*金字塔阶内的每一层*/
	float levelsigma;
	int levelsigmalength;       //作用于前一张图上的高斯直径
	float absolute_sigma;
	Mat Level;
};

struct ImageOctave {      /*金字塔每一阶*/
	int row, col;          //Dimensions of image.   
	float subsample;
	ImageLevel* Octave;
};

struct Keypoint
{
	float row, col; /* 反馈回原图像大小，特征点的位置 */
	float sx, sy;    /* 金字塔中特征点的位置*/
	int octave, level;/*金字塔中，特征点所在的阶梯、层次*/

	float scale, ori, mag; /*所在层的绝对尺度absolute_sigma,主方向orientation ( range [-PI,PI) )，以及幅值*/
	float* descrip;       /*特征描述字指针：128维或32维等*/
	Keypoint* next;/* Pointer to next keypoint in list. */
};

struct matchPoint {
	Point2i p1, p2;
	matchPoint(Point2i pt1, Point2i pt2) {
		p1 = pt1;
		p2 = pt2;
	}
};
namespace cv_pre {
	void doubleSizeImageColor(Mat im, Mat imnew);
	vector<matchPoint> compute_macth(Keypoint* k1, Keypoint* k2, float maxloss = 10);


	Mat halfSizeImage(Mat im);     //缩小图像：下采样  
	Mat doubleSizeImage(Mat im);   //扩大图像：最近临方法  
	Mat  doubleSizeImage2(Mat im);  //扩大图像：线性插值  
	float getPixelBI(Mat im, float col, float row);//双线性插值函数  
	void normalizeVec(float* vec, int dim);//向量归一化    
	Mat GaussianKernel2D(float sigma);  //得到2维高斯核  
	void normalizeMat(Mat mat);        //矩阵归一化  
	float* GaussianKernel1D(float sigma, int dim); //得到1维高斯核  

	float GetVecNorm(float* vec, int dim);

	//在具体像素处宽度方向进行高斯卷积  
	float ConvolveLocWidth(float* kernel, int dim, Mat src, int x, int y);
	//在整个图像宽度方向进行1D高斯卷积  
	void Convolve1DWidth(float* kern, int dim, Mat src, Mat dst);
	//在具体像素处高度方向进行高斯卷积  
	float ConvolveLocHeight(float* kernel, int dim, Mat src, int x, int y);
	//在整个图像高度方向进行1D高斯卷积  
	void Convolve1DHeight(float* kern, int dim, Mat src, Mat dst);
	//用高斯函数模糊图像    
	int BlurImage(Mat src, Mat dst, float sigma);


}
#pragma once
