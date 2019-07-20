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
#define GAUSSKERN 3.5   //���ڸ�˹�˵Ĵ�С��dim = 2 * kern * xita + 1
#define PI 3.14159265358979323846  

//Sigma of base image -- See D.L.'s paper.  
#define INITSIGMA 0.5  
//Sigma of each octave -- See D.L.'s paper.  
#define SIGMA sqrt(3)//1.6//  

//Number of scales per octave.  See D.L.'s paper.  
#define SCALESPEROCTAVE 2       //ÿ�������ɼ������ֵ��Ĳ���  
#define MAXOCTAVES 4  


#define CONTRAST_THRESHOLD   0.02  
#define CURVATURE_THRESHOLD  10.0  
#define DOUBLE_BASE_IMAGE_SIZE 1  
#define peakRelThresh 0.8  
#define LEN 128  

using namespace cv;
using namespace std;

struct ImageLevel {        /*���������ڵ�ÿһ��*/
	float levelsigma;
	int levelsigmalength;       //������ǰһ��ͼ�ϵĸ�˹ֱ��
	float absolute_sigma;
	Mat Level;
};

struct ImageOctave {      /*������ÿһ��*/
	int row, col;          //Dimensions of image.   
	float subsample;
	ImageLevel* Octave;
};

struct Keypoint
{
	float row, col; /* ������ԭͼ���С���������λ�� */
	float sx, sy;    /* ���������������λ��*/
	int octave, level;/*�������У����������ڵĽ��ݡ����*/

	float scale, ori, mag; /*���ڲ�ľ��Գ߶�absolute_sigma,������orientation ( range [-PI,PI) )���Լ���ֵ*/
	float* descrip;       /*����������ָ�룺128ά��32ά��*/
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


	Mat halfSizeImage(Mat im);     //��Сͼ���²���  
	Mat doubleSizeImage(Mat im);   //����ͼ������ٷ���  
	Mat  doubleSizeImage2(Mat im);  //����ͼ�����Բ�ֵ  
	float getPixelBI(Mat im, float col, float row);//˫���Բ�ֵ����  
	void normalizeVec(float* vec, int dim);//������һ��    
	Mat GaussianKernel2D(float sigma);  //�õ�2ά��˹��  
	void normalizeMat(Mat mat);        //�����һ��  
	float* GaussianKernel1D(float sigma, int dim); //�õ�1ά��˹��  

	float GetVecNorm(float* vec, int dim);

	//�ھ������ش���ȷ�����и�˹���  
	float ConvolveLocWidth(float* kernel, int dim, Mat src, int x, int y);
	//������ͼ���ȷ������1D��˹���  
	void Convolve1DWidth(float* kern, int dim, Mat src, Mat dst);
	//�ھ������ش��߶ȷ�����и�˹���  
	float ConvolveLocHeight(float* kernel, int dim, Mat src, int x, int y);
	//������ͼ��߶ȷ������1D��˹���  
	void Convolve1DHeight(float* kern, int dim, Mat src, Mat dst);
	//�ø�˹����ģ��ͼ��    
	int BlurImage(Mat src, Mat dst, float sigma);


}
#pragma once
