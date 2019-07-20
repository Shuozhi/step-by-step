//#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include"pre.h"
#include <algorithm> 
#include <direct.h>
#include <io.h>
#include <iostream>
#include <fstream>
using namespace cv;
using namespace std;
using namespace cv_pre;
class sift {
public:
	int descrip_lenth = 0;
	int     numoctaves = 0;
	ImageOctave* DOGoctaves;
	//DOG pyr，DOG算子计算简单，是尺度归一化的LoG算子的近似。
	ImageOctave* mag_pyr; //梯度塔
	ImageOctave* grad_pyr;//梯度方向塔
	//定义特征点具体变量  
	Keypoint* keypoints = NULL;      //用于临时存储特征点的位置等  
	Keypoint* keyDescriptors = NULL; //用于存储最终特征点（同点不同向属于不同点）
	//4*4 
#define GridSpacing 4  


//第一步
	Mat ScaleInitImage(Mat im)  //输入im为灰度图像32F
	{
		double preblur_sigma;
		Mat imMat = im.clone();
		int gaussdim = (int)max(3.0, 2.0 * GAUSSKERN * INITSIGMA + 1.0);//高斯核的尺寸
		gaussdim = 2 * (gaussdim / 2) + 1;
		//预滤波除噪声
		GaussianBlur(imMat, imMat, Size(gaussdim, gaussdim), INITSIGMA);

		//针对两种情况分别进行处理：初始化放大原始图像或者在原图像基础上进行后续操作
		if (DOUBLE_BASE_IMAGE_SIZE)
		{
			Mat bottom_Mat = doubleSizeImage2(imMat);//线性插值两倍上采样
			preblur_sigma = 1.0;
			gaussdim = (int)max(3.0, 2.0 * GAUSSKERN * preblur_sigma + 1.0);//高斯核的尺寸
			gaussdim = 2 * (gaussdim / 2) + 1;
			GaussianBlur(bottom_Mat, bottom_Mat, Size(gaussdim, gaussdim), preblur_sigma);
			return bottom_Mat;
		}
		else
		{
			preblur_sigma = 1.0;
			gaussdim = (int)max(3.0, 2.0 * GAUSSKERN * preblur_sigma + 1.0);//高斯核的尺寸
			BlurImage(imMat, imMat, preblur_sigma); //得到金字塔的最底层：原始图像大小  
			return imMat;
		}
	}

	//SIFT算法第二步  
	ImageOctave* BuildGaussianOctaves(Mat img)
	{
		//分配内存 给octaves , DOGoctaves
		ImageOctave* octaves;
		octaves = new ImageOctave[numoctaves];
		DOGoctaves = new ImageOctave[numoctaves];
		//计算阶梯内的层数
		int num_peroc_levels = SCALESPEROCTAVE + 3;
		int num_perdog_levels = num_peroc_levels - 1;
		//
				//其他参数
		Mat tempMat = img.clone(), dst, temp;
		float init_sigma = pow(2, 1.0 / 2);


		double k = pow(2, 1.0 / ((float)SCALESPEROCTAVE));  //方差倍数   根号2 
//
		//在每一阶金字塔图像中建立不同的尺度图像  
		for (int i = 0; i < numoctaves; i++)
		{
			{
				//分配内存
				octaves[i].Octave = new ImageLevel[num_peroc_levels];
				DOGoctaves[i].Octave = new ImageLevel[num_perdog_levels];
				//首先建立金字塔每一阶梯的最底层，其中0阶梯的最底层已经建立好  
				(octaves[i].Octave)[0].Level = tempMat;
				octaves[i].col = tempMat.cols;
				octaves[i].row = tempMat.rows;
				DOGoctaves[i].col = tempMat.cols;
				DOGoctaves[i].row = tempMat.rows;
				if (DOUBLE_BASE_IMAGE_SIZE)
					octaves[i].subsample = pow(2, i) * 0.5;
				else
					octaves[i].subsample = pow(2, i);
			}

			if (i == 0)
			{
				(octaves[0].Octave)[0].levelsigma = init_sigma;
				(octaves[0].Octave)[0].absolute_sigma = init_sigma / 2;
			}
			else
			{
				(octaves[i].Octave)[0].levelsigma = init_sigma;
				(octaves[i].Octave)[0].absolute_sigma = (octaves[i - 1].Octave)[num_peroc_levels - 3].absolute_sigma;
			}

			float sigma = init_sigma;
			float sigma_act, absolute_sigma;    //每次直接作用于前图像上的blur值    ；   尺度空间中的绝对值

			//建立本阶梯其他层的图像  
			for (int j = 1; j < SCALESPEROCTAVE + 3; j++)
			{
				dst = Mat(tempMat.rows, tempMat.cols, CV_32FC1);//用于存储高斯层  
				temp = Mat(tempMat.rows, tempMat.cols, CV_32FC1);//用于存储DOG层  

				sigma_act = sqrt(k * k - 1) * sigma;
				sigma = k * sigma;

				(octaves[i].Octave)[j].levelsigma = sigma;
				(octaves[i].Octave)[j].absolute_sigma = sigma * (octaves[i].subsample);
				// (octaves[i].Octave)[j].absolute_sigma = k *((octaves[i].Octave)[j-1].absolute_sigma);

				//产生高斯层  
				int gaussdim = (int)max(3.0, 2.0 * GAUSSKERN * sigma_act + 1.0);//高斯核的尺寸
				gaussdim = 2 * (gaussdim / 2) + 1;
				GaussianBlur((octaves[i].Octave)[j - 1].Level, dst, Size(gaussdim, gaussdim), sigma_act);
				//BlurImage((octaves[i].Octave)[j - 1].Level, dst, sigma_act);
				(octaves[i].Octave)[j].levelsigmalength = gaussdim;
				(octaves[i].Octave)[j].Level = dst;

				//产生DOG层  
				temp = ((octaves[i].Octave)[j]).Level - ((octaves[i].Octave)[j - 1]).Level;
				//subtract(((octaves[i].Octave)[j]).Level, ((octaves[i].Octave)[j - 1]).Level, temp, 0);
				((DOGoctaves[i].Octave)[j - 1]).Level = temp;
			}

			tempMat = halfSizeImage(((octaves[i].Octave)[SCALESPEROCTAVE].Level));
		}
		return octaves;
	}

	//SIFT算法第三步，特征点位置检测，  
	int DetectKeypoint(int numoctaves, ImageOctave* GaussianPyr)
	{
		//计算用于DOG极值点检测的主曲率比的阈值  
		double curvature_threshold = ((CURVATURE_THRESHOLD + 1) * (CURVATURE_THRESHOLD + 1)) / CURVATURE_THRESHOLD;
		curvature_threshold = 10;
#define ImLevels(OCTAVE,LEVEL,ROW,COL)  ((float)(DOGoctaves[(OCTAVE)].Octave[(LEVEL)].Level).ptr<float>(ROW)[COL])
		int   keypoint_count = 0;
		for (int i = 0; i < numoctaves; i++)
		{
			for (int j = 1; j < SCALESPEROCTAVE + 1; j++)//取中间的scaleperoctave个层  
			{
				//在图像的有效区域内寻找具有显著性特征的局部最大值   
				int dim = (int)(0.5 * ((GaussianPyr[i].Octave)[j].levelsigmalength) + 0.5);
				for (int m = dim; m < ((DOGoctaves[i].row) - dim); m++)
					for (int n = dim; n < ((DOGoctaves[i].col) - dim); n++)
					{
						if (fabs(ImLevels(i, j, m, n)) >= CONTRAST_THRESHOLD)
						{
							if (ImLevels(i, j, m, n) != 0.0)  //1、首先是非零  
							{
								float inf_val = ImLevels(i, j, m, n);
								if (((inf_val <= ImLevels(i, j - 1, m - 1, n - 1)) &&
									(inf_val <= ImLevels(i, j - 1, m, n - 1)) &&
									(inf_val <= ImLevels(i, j - 1, m + 1, n - 1)) &&
									(inf_val <= ImLevels(i, j - 1, m - 1, n)) &&
									(inf_val <= ImLevels(i, j - 1, m, n)) &&
									(inf_val <= ImLevels(i, j - 1, m + 1, n)) &&
									(inf_val <= ImLevels(i, j - 1, m - 1, n + 1)) &&
									(inf_val <= ImLevels(i, j - 1, m, n + 1)) &&
									(inf_val <= ImLevels(i, j - 1, m + 1, n + 1)) &&    //底层的小尺度9  

									(inf_val <= ImLevels(i, j, m - 1, n - 1)) &&
									(inf_val <= ImLevels(i, j, m, n - 1)) &&
									(inf_val <= ImLevels(i, j, m + 1, n - 1)) &&
									(inf_val <= ImLevels(i, j, m - 1, n)) &&
									(inf_val <= ImLevels(i, j, m + 1, n)) &&
									(inf_val <= ImLevels(i, j, m - 1, n + 1)) &&
									(inf_val <= ImLevels(i, j, m, n + 1)) &&
									(inf_val <= ImLevels(i, j, m + 1, n + 1)) &&     //当前层8  

									(inf_val <= ImLevels(i, j + 1, m - 1, n - 1)) &&
									(inf_val <= ImLevels(i, j + 1, m, n - 1)) &&
									(inf_val <= ImLevels(i, j + 1, m + 1, n - 1)) &&
									(inf_val <= ImLevels(i, j + 1, m - 1, n)) &&
									(inf_val <= ImLevels(i, j + 1, m, n)) &&
									(inf_val <= ImLevels(i, j + 1, m + 1, n)) &&
									(inf_val <= ImLevels(i, j + 1, m - 1, n + 1)) &&
									(inf_val <= ImLevels(i, j + 1, m, n + 1)) &&
									(inf_val <= ImLevels(i, j + 1, m + 1, n + 1))     //下一层大尺度9          
									) ||
									((inf_val >= ImLevels(i, j - 1, m - 1, n - 1)) &&
									(inf_val >= ImLevels(i, j - 1, m, n - 1)) &&
										(inf_val >= ImLevels(i, j - 1, m + 1, n - 1)) &&
										(inf_val >= ImLevels(i, j - 1, m - 1, n)) &&
										(inf_val >= ImLevels(i, j - 1, m, n)) &&
										(inf_val >= ImLevels(i, j - 1, m + 1, n)) &&
										(inf_val >= ImLevels(i, j - 1, m - 1, n + 1)) &&
										(inf_val >= ImLevels(i, j - 1, m, n + 1)) &&
										(inf_val >= ImLevels(i, j - 1, m + 1, n + 1)) &&

										(inf_val >= ImLevels(i, j, m - 1, n - 1)) &&
										(inf_val >= ImLevels(i, j, m, n - 1)) &&
										(inf_val >= ImLevels(i, j, m + 1, n - 1)) &&
										(inf_val >= ImLevels(i, j, m - 1, n)) &&
										(inf_val >= ImLevels(i, j, m + 1, n)) &&
										(inf_val >= ImLevels(i, j, m - 1, n + 1)) &&
										(inf_val >= ImLevels(i, j, m, n + 1)) &&
										(inf_val >= ImLevels(i, j, m + 1, n + 1)) &&

										(inf_val >= ImLevels(i, j + 1, m - 1, n - 1)) &&
										(inf_val >= ImLevels(i, j + 1, m, n - 1)) &&
										(inf_val >= ImLevels(i, j + 1, m + 1, n - 1)) &&
										(inf_val >= ImLevels(i, j + 1, m - 1, n)) &&
										(inf_val >= ImLevels(i, j + 1, m, n)) &&
										(inf_val >= ImLevels(i, j + 1, m + 1, n)) &&
										(inf_val >= ImLevels(i, j + 1, m - 1, n + 1)) &&
										(inf_val >= ImLevels(i, j + 1, m, n + 1)) &&
										(inf_val >= ImLevels(i, j + 1, m + 1, n + 1))
										))      //2、满足26个中极值点  
								{
									//此处可存储  
									//解释：有效极值点的尺度空间构形应该是平坦区夹着 剧变区 ，而不是剧变区夹着平坦区，即不应该是0点。
									if (fabs(ImLevels(i, j, m, n)) >= CONTRAST_THRESHOLD)
									{
										//最后显著处的特征点必须具有足够的曲率比，CURVATURE_THRESHOLD=10.0，首先计算Hessian矩阵  
										// Compute the entries of the Hessian matrix at the extrema location.  
										/*
										1   0   -1
										0   0   0
										-1   0   1         *0.25
										*/
										// Compute the trace and the determinant of the Hessian.  
										//Tr_H = Dxx + Dyy;  
										//Det_H = Dxx*Dyy - Dxy^2;  
										float Dxx, Dyy, Dxy, Tr_H, Det_H, curvature_ratio;
										Dxx = ImLevels(i, j, m, n - 1) + ImLevels(i, j, m, n + 1) - 2.0 * ImLevels(i, j, m, n);
										Dyy = ImLevels(i, j, m - 1, n) + ImLevels(i, j, m + 1, n) - 2.0 * ImLevels(i, j, m, n);
										Dxy = ImLevels(i, j, m - 1, n - 1) + ImLevels(i, j, m + 1, n + 1) - ImLevels(i, j, m + 1, n - 1) - ImLevels(i, j, m - 1, n + 1);
										Tr_H = Dxx + Dyy;
										Det_H = Dxx * Dyy - Dxy * Dxy;
										// Compute the ratio of the principal curvatures.  
										curvature_ratio = (1.0 * Tr_H * Tr_H) / Det_H;
										if ((Det_H >= 0.0) && (curvature_ratio <= curvature_threshold))  //最后得到最具有显著性特征的特征点  
										{
											//将其存储起来，以计算后面的特征描述字  
											keypoint_count++;
											Keypoint* k;
											k = (Keypoint*)malloc(sizeof(struct Keypoint));
											k->next = keypoints;
											keypoints = k;
											k->row = m * (GaussianPyr[i].subsample);
											k->col = n * (GaussianPyr[i].subsample);
											k->sy = m;    //行  
											k->sx = n;    //列  
											k->octave = i;
											k->level = j;
											k->scale = (GaussianPyr[i].Octave)[j].absolute_sigma;
										}//if >curvature_thresh  
									}//if >contrast  
								}//if inf value  
							}//if non zero  
						}//if >contrast  
					}  //for concrete image level col  
			}//for levels  
		}//for octaves  
		return keypoint_count;
	}

	//在图像中，显示SIFT特征点的位置  
	void DisplayKeypointLocation(Mat image, ImageOctave* GaussianPyr)
	{

		Keypoint* p = keypoints; // p指向第一个结点  
		while (p) // 没到表尾  
		{
			line(image, Point2i((int)((p->col) - 3), (int)(p->row)), Point2i((int)((p->col) + 3), (int)(p->row)), Scalar(0, 0, 255), 1, 8, 0);
			line(image, Point2i((int)(p->col), (int)((p->row) - 3)), Point2i((int)(p->col), (int)((p->row) + 3)), Scalar(0, 0, 255), 1, 8, 0);
			p = p->next;
		}
	}



	//Sift第四步准备工作

	// 计算高斯金字塔的梯度方向和量值。此时keypoints已经计算好
	void ComputeGrad_DirecandMag(ImageOctave* GaussianPyr)
	{
		mag_pyr = new ImageOctave[numoctaves];
		grad_pyr = new ImageOctave[numoctaves];
#define ImLevels(OCTAVE,LEVEL,ROW,COL)  ((float)(GaussianPyr[(OCTAVE)].Octave[(LEVEL)].Level).ptr<float>(ROW)[COL])
		for (int i = 0; i < numoctaves; i++)
		{
			mag_pyr[i].Octave = new ImageLevel[SCALESPEROCTAVE];
			grad_pyr[i].Octave = new ImageLevel[SCALESPEROCTAVE];
			cout << GaussianPyr[i].row << endl;
			for (int j = 1; j < SCALESPEROCTAVE + 1; j++)//取中间的scaleperoctave个层  
			{
				Mat Mag(GaussianPyr[i].row, GaussianPyr[i].col, CV_32FC1);
				Mat Ori(GaussianPyr[i].row, GaussianPyr[i].col, CV_32FC1);
				Mat gradx(GaussianPyr[i].row, GaussianPyr[i].col, CV_32FC1);
				Mat grady(GaussianPyr[i].row, GaussianPyr[i].col, CV_32FC1);
#define MAG(ROW,COL) ((float)((Mag).ptr<float>(ROW)[COL]) )   
#define ORI(ROW,COL) ((float)((Ori).ptr<float>(ROW)[COL]) )  
#define GRADX(ROW,COL) ((float)((gradx).ptr<float>(ROW)[COL])  ) 
#define GRADY(ROW,COL) ((float)((grady).ptr<float>(ROW)[COL])  )
				for (int m = 1; m < (GaussianPyr[i].row - 1); m++)
					for (int n = 1; n < (GaussianPyr[i].col - 1); n++)
					{
						//计算幅值  
						(gradx).ptr<float>(m)[n] = 0.5 * (ImLevels(i, j, m, n + 1) - ImLevels(i, j, m, n - 1));  //dx  
						(grady).ptr<float>(m)[n] = 0.5 * (ImLevels(i, j, m + 1, n) - ImLevels(i, j, m - 1, n));  //dy  
						(Mag).ptr<float>(m)[n] = sqrt(GRADX(m, n) * GRADX(m, n) + GRADY(m, n) * GRADY(m, n));  //mag  

																										   //atan的范围是 （-PI/2，PI/2 ）
						(Ori).ptr<float>(m)[n] = atan(GRADY(m, n) / GRADX(m, n)); //+ ((gradx).ptr<float>(m)[n] < 0 ? CV_PI : 0);
						if ((gradx).ptr<float>(m)[n] < 0) {
							(Ori).ptr<float>(m)[n] = (Ori).ptr<float>(m)[n] + CV_PI;
						}
						if (ORI(m, n) >= CV_PI)
							(Ori).ptr<float>(m)[n] = (Ori).ptr<float>(m)[n] - 2 * CV_PI;
					}
				((mag_pyr[i].Octave)[j - 1]).Level = Mag;
				((grad_pyr[i].Octave)[j - 1]).Level = Ori;
			}//for levels  
		}//for octaves  
	}

	//寻找与方向直方图最近的柱，确定其index   
	int FindClosestRotationBin(int binCount, float angle)
	{
		angle += CV_PI;
		angle /= 2.0 * CV_PI;
		// calculate the aligned bin  
		angle *= binCount;
		int idx = (int)angle;
		if (idx == binCount)
			idx = 0;
		return (idx);
	}

	// 对梯度分布直方图进行中值滤波  
	void AverageWeakBins(double* hist, int binCount)
	{
		for (int sn = 0; sn < 2; ++sn)
		{
			double firstE = hist[0];
			double last = hist[binCount - 1];
			for (int sw = 0; sw < binCount; ++sw)
			{
				double cur = hist[sw];
				double next = (sw == (binCount - 1)) ? firstE : hist[(sw + 1) % binCount];
				hist[sw] = (last + cur + next) / 3.0;
				last = cur;
			}
		}
	}

	//二次曲线拟合精确角度
	bool InterpolateOrientation(double left, double middle, double right, double* degreeCorrection, double* peakValue)
	{
		double a = ((left + right) - 2.0 * middle) / 2.0;   //抛物线捏合系数a  
															// degreeCorrection = peakValue = Double.NaN;  

															// Not a parabol  
		if (a == 0.0)
			return false;
		double c = (((left - middle) / a) - 1.0) / 2.0;
		double b = middle - c * c * a;
		if (c < -0.5 || c > 0.5)
			return false;
		*degreeCorrection = c;
		*peakValue = b;
		return true;
	}

	//SIFT算法第四步：计算各个特征点的主方向，确定主方向  
	void AssignTheMainOrientation(int numoctaves, ImageOctave* GaussianPyr, ImageOctave* mag_pyr, ImageOctave* grad_pyr)
	{
		int num_bins = 36;
		float hist_step = (2.0 * CV_PI) / (num_bins + 0.0);   //角度离散化的步长
		float hist_orient[36];          //角度离散化的起始 角度。
		for (int i = 0; i < 36; i++) {
			hist_orient[i] = -CV_PI + i * hist_step;
		}

		//边缘zeropad之外为有效区域
		float sigma1 = (((GaussianPyr[0].Octave)[SCALESPEROCTAVE].absolute_sigma)) / (GaussianPyr[0].subsample);
		int zero_pad = (int)((int(std::max(3.0, 2 * GAUSSKERN * sigma1 + 1.0))) / 2 + 1);
#define ImLevels(OCTAVE,LEVEL,ROW,COL)  ((float)((GaussianPyr[(OCTAVE)].Octave[(LEVEL)].Level).ptr<float>(ROW)[COL]))
		int keypoint_count = 0;
		Keypoint* p = keypoints;

		//声明方向直方图变量  
		double* orienthist = new double[36];
		//double* orienthist = (double *)malloc(36 * sizeof(double));

		while (p)
		{
			int i = p->octave;
			int j = p->level;   //极值点在dog的第几层，就是在Gauss的第几层.  都是在1 、2层
			int m = p->sy;   //行  
			int n = p->sx;   //列  
			if ((m >= zero_pad) && (m < GaussianPyr[i].row - zero_pad) &&
				(n >= zero_pad) && (n < GaussianPyr[i].col - zero_pad))       //有效区域
			{
				float sigma = (((GaussianPyr[i].Octave)[j].absolute_sigma)) / (GaussianPyr[i].subsample);
				//产生二维高斯模板  
				Mat mat = GaussianKernel2D(sigma);
				int dim = (int)max(3.0, 2.0 * GAUSSKERN * sigma + 1.0);   dim = 2 * (dim / 2) + 1;
				dim = dim / 2;
				//分配用于存储Patch幅值和方向的空间  
#define MAT(ROW,COL) ((float)(mat).ptr<float>(ROW)[COL])
				for (int sw = 0; sw < 36; ++sw)
				{
					orienthist[sw] = 0.0;
				}
				//在特征点的周围统计梯度方向  
				for (int x = m - dim, mm = 0; x <= (m + dim); x++, mm++)
					for (int y = n - dim, nn = 0; y <= (n + dim); y++, nn++)
					{
						//计算特征点处的幅值  
						double mag = ((mag_pyr[i].Octave)[j - 1]).Level.ptr<float>(x)[y];
						double Ori = ((grad_pyr[i].Octave)[j - 1]).Level.ptr<float>(x)[y];
						int binIdx = FindClosestRotationBin(36, Ori);                   //得到离现有方向最近的直方块  
						orienthist[binIdx] = orienthist[binIdx] + 1.0 * mag * MAT(mm, nn);//利用高斯加权累加进直方图相应的块  
					}

				// Find peaks in the orientation histogram using nonmax suppression.  

				//对orienthist进行中值滤波。
				AverageWeakBins(orienthist, 36);

				// 排序寻找最大方向 
				double maxGrad = 0.0;   //最大方向的权值
				int maxBin = 0;
				for (int b = 0; b < 36; ++b)
				{
					if (orienthist[b] > maxGrad)
					{
						maxGrad = orienthist[b];
						maxBin = b;
					}
				}

				double maxPeakValue = 0.0;
				double maxDegreeCorrection = 0.0;
				//二次曲线精确拟合maxDegree
				if ((InterpolateOrientation(orienthist[maxBin == 0 ? (36 - 1) : (maxBin - 1)],
					orienthist[maxBin],
					orienthist[(maxBin + 1) % 36],
					&maxDegreeCorrection,
					&maxPeakValue)) == false)
				{
					printf("BUG: Parabola fitting broken");
				}


				//次最大值方向的筛选。0.8*maxPeakValue以上的都是
				bool binIsKeypoint[36];
				for (int b = 0; b < 36; ++b)
				{
					binIsKeypoint[b] = false;

					if (b == maxBin)
					{
						binIsKeypoint[b] = true;
						continue;
					}
					if (orienthist[b] < (peakRelThresh * maxPeakValue))
						continue;

					int leftI = (b == 0) ? (36 - 1) : (b - 1);
					int rightI = (b + 1) % 36;
					if (orienthist[b] <= orienthist[leftI] || orienthist[b] <= orienthist[rightI])
						continue; // 虽然满足0.8但是不是局部最大值  
					binIsKeypoint[b] = true;
				}

				// find other possible locations  
				double oneBinRad = (2.0 * CV_PI) / 36;
				for (int b = 0; b < 36; ++b)
				{
					if (binIsKeypoint[b] == true) {
						int bLeft = (b == 0) ? (36 - 1) : (b - 1);
						int bRight = (b + 1) % 36;

						double peakValue;
						double degreeCorrection;

						if (InterpolateOrientation(orienthist[maxBin == 0 ? (36 - 1) : (maxBin - 1)],
							orienthist[maxBin], orienthist[(maxBin + 1) % 36],
							&degreeCorrection, &peakValue) == false)        //degreeCorrection?????
						{
							printf("BUG: Parabola fitting broken");
						}

						double degree = (b + degreeCorrection) * oneBinRad - CV_PI;
						if (degree < -CV_PI)
							degree += 2.0 * CV_PI;
						else if (degree > CV_PI)
							degree -= 2.0 * CV_PI;

						//分配内存重新存储特征点  
						Keypoint* k = new Keypoint();
						k->next = keyDescriptors;
						keyDescriptors = k;
						k->row = p->row;
						k->col = p->col;
						k->sy = p->sy;    //行  
						k->sx = p->sx;    //列  
						k->octave = p->octave;
						k->level = p->level;
						k->scale = p->scale;

						k->ori = degree;
						k->mag = peakValue;
					}
				}
			}
			p = p->next;
		}
		//free(orienthist);
	}

	//显示特征点处的主方向  
	void DisplayOrientation(Mat image, ImageOctave* GaussianPyr)
	{
		Keypoint* p = keyDescriptors; // p指向第一个结点  
		while (p) // 没到表尾  
		{
			float scale = (GaussianPyr[p->octave].Octave)[p->level].absolute_sigma;
			//scale = p->scale;
			float autoscale = 3.0;
			float uu = autoscale * scale * cos(p->ori);
			float vv = autoscale * scale * sin(p->ori);
			float x = (p->col) + uu;
			float y = (p->row) + vv;

			line(image, cvPoint((int)(p->col), (int)(p->row)),
				cvPoint((int)x, (int)y), CV_RGB(255, 255, 0),
				1, 8, 0);

			// 画箭头 
			float alpha = 0.33; // Size of arrow head relative to the length of the vector  
			float beta = 0.6;  // Width of the base of the arrow head relative to the length  

			float xx0 = (p->col) + uu - alpha * (uu + beta * vv);
			float yy0 = (p->row) + vv - alpha * (vv - beta * uu);
			float xx1 = (p->col) + uu - alpha * (uu - beta * vv);
			float yy1 = (p->row) + vv - alpha * (vv + beta * uu);
			line(image, cvPoint((int)xx0, (int)yy0),
				cvPoint((int)x, (int)y), CV_RGB(255, 255, 0),
				1, 8, 0);
			line(image, cvPoint((int)xx1, (int)yy1),
				cvPoint((int)x, (int)y), CV_RGB(255, 255, 0),
				1, 8, 0);
			p = p->next;
		}
	}


	//void ExtractFeatureDescriptors(ImageOctave *GaussianPyr)
	//{
	//  float feat_window = 2 * GridSpacing;
	//  float orient_bin_spacing = CV_PI / 4;//bin区间间隔 四分之PI
	//                                       //8个方向的角度值
	//  float orient_angles[8] = { -CV_PI,-CV_PI + orient_bin_spacing,-CV_PI*0.5, -orient_bin_spacing,
	//      0.0, orient_bin_spacing, CV_PI*0.5,  CV_PI + orient_bin_spacing };
	//  //产生描述字中心各点坐标  
	//  float *feat_grid = (float *)malloc(2 * 16 * sizeof(float));
	//  for (int i = 0; i<GridSpacing; i++)
	//  {
	//      for (int j = 0; j<2 * GridSpacing; j += 2)
	//      {
	//          feat_grid[i * 2 * GridSpacing + j] = -6.0 + i*GridSpacing;
	//          feat_grid[i * 2 * GridSpacing + j + 1] = -6.0 + 0.5*j*GridSpacing;
	//      }
	//  }
	//  //产生网格  16个大格子。每个大格子里有16个小格子。。。 
	//  float *feat_samples = (float *)malloc(2 * 256 * sizeof(float));
	//  for (int i = 0; i<4 * GridSpacing; i++)
	//  {
	//      for (int j = 0; j<8 * GridSpacing; j += 2)
	//      {
	//          feat_samples[i * 8 * GridSpacing + j] = -(2 * GridSpacing - 0.5) + i;
	//          feat_samples[i * 8 * GridSpacing + j + 1] = -(2 * GridSpacing - 0.5) + 0.5*j;
	//      }
	//  }
	//  // p指向第一个结点 
	//  Keypoint *p = keyDescriptors;
	//  while (p)
	//  {
	//      float scale = (GaussianPyr[p->octave].Octave)[p->level].absolute_sigma;
	//      float sine = sin(p->ori);
	//      float cosine = cos(p->ori);
	//      //计算大格中心点旋转之后的位置        featcenter[]数组 
	//      float *featcenter = (float *)malloc(2 * 16 * sizeof(float));
	//      for (int i = 0; i<GridSpacing; i++)
	//      {
	//          for (int j = 0; j<2 * GridSpacing; j += 2)
	//          {
	//              float x = feat_grid[i * 2 * GridSpacing + j];
	//              float y = feat_grid[i * 2 * GridSpacing + j + 1];
	//              featcenter[i * 2 * GridSpacing + j] = ((cosine * x + sine * y) + p->sx);
	//              featcenter[i * 2 * GridSpacing + j + 1] = ((-sine * x + cosine * y) + p->sy);
	//          }
	//      }
	//      //网格中心点旋转后的位置       feat[]数组
	//      float *feat = (float *)malloc(2 * 256 * sizeof(float));
	//      for (int i = 0; i<64 * GridSpacing; i++, i++)
	//      {
	//          float x = feat_samples[i];
	//          float y = feat_samples[i + 1];
	//          feat[i] = ((cosine * x + sine * y) + p->sx);
	//          feat[i + 1] = ((-sine * x + cosine * y) + p->sy);
	//      }
	//      //初始化特征描述子  feature_descriptors
	//      float *feat_desc = (float *)malloc(128 * sizeof(float));
	//      for (int i = 0; i<128; i++)
	//      {
	//          feat_desc[i] = 0.0;
	//      }
	//      //256个网格采样点 *2 
	//      for (int i = 0; i<512; i += 2)
	//      {
	//          //插值计算网格中心点梯度方向
	//          /*
	//          0   12   0
	//          21  22   23
	//          0   32   0   具体插值策略如图示
	//          */
	//          float mag_sample = 0, grad_sample = 0, x_sample = 0, y_sample = 0; {
	//              x_sample = feat[i];
	//              y_sample = feat[i + 1];
	//              float sample12 = getPixelBI(((GaussianPyr[p->octave].Octave)[p->level]).Level, x_sample, y_sample - 1);
	//              float sample21 = getPixelBI(((GaussianPyr[p->octave].Octave)[p->level]).Level, x_sample - 1, y_sample);
	//              float sample22 = getPixelBI(((GaussianPyr[p->octave].Octave)[p->level]).Level, x_sample, y_sample);
	//              float sample23 = getPixelBI(((GaussianPyr[p->octave].Octave)[p->level]).Level, x_sample + 1, y_sample);
	//              float sample32 = getPixelBI(((GaussianPyr[p->octave].Octave)[p->level]).Level, x_sample, y_sample + 1);
	//              float diff_x = sample23 - sample21;
	//              float diff_y = sample32 - sample12;
	//              mag_sample = sqrt(diff_x*diff_x + diff_y*diff_y);
	//              grad_sample = atan(diff_y / diff_x);
	//              if (diff_x < 0)             grad_sample += CV_PI;
	//              if (grad_sample == CV_PI)   grad_sample = -CV_PI;
	//          }
	//          //663-710行代码，表示不透彻
	//          //需要改进调优的地方！！！！！！！！！！
	//          // 计算采样点对于4*4个种子点的权重，其实只有邻近的种子点会有权重
	//          // 这类似 hog算子的 block 归一化。
	//          // float[128]，对每个种子点内的8方向权值是一样的。
	//          float *pos_wght = (float *)malloc(8 * GridSpacing * GridSpacing * sizeof(float)); {
	//              float *x_wght = (float *)malloc(GridSpacing * GridSpacing * sizeof(float)); // float[16]
	//              float *y_wght = (float *)malloc(GridSpacing * GridSpacing * sizeof(float)); // float[16]
	//              for (int m = 0; m<32; ++m, ++m)
	//              {
	//                  //(x,y)是16个种子点的位置
	//                  float x = featcenter[m];
	//                  float y = featcenter[m + 1];
	//                  //GridSpacing是不是少乘了一个0.5？？？？
	//                  x_wght[m / 2] = max(1 - (fabs(x - x_sample)*1.0 / GridSpacing), 0.0);   //只有采样点附近的种子点 的x_wt值才是非0；
	//                  y_wght[m / 2] = max(1 - (fabs(y - y_sample)*1.0 / GridSpacing), 0.0);
	//              }
	//              for (int m = 0; m<16; ++m)
	//                  for (int n = 0; n<8; ++n)
	//                      pos_wght[m * 8 + n] = x_wght[m] * y_wght[m];
	//              free(x_wght);
	//              free(y_wght);
	//          }
	//          //首先旋转梯度场到主方向，然后计算幅值到各个方向的差值
	//          float diff[8];
	//          for (int m = 0; m<8; ++m)
	//          {
	//              float angle = grad_sample - (p->ori) - orient_angles[m] + CV_PI; //差值+pi
	//              float temp = angle / (2.0 * CV_PI);
	//              angle -= (int)(temp) * (2.0 * CV_PI);
	//              diff[m] = angle - CV_PI;
	//          }
	//          // 计算高斯权重  
	//          float x = p->sx, y = p->sy;                                                                 //feat_window=2*GridSpacing=8  .原代码是不是少写了sqrt
	//          float g = exp(-((x_sample - x)*(x_sample - x) + (y_sample - y)*(y_sample - y)) / (2 * feat_window*feat_window)) / sqrt(2 * CV_PI*feat_window*feat_window);
	//          float orient_wght[128];
	//          for (int m = 0; m<128; ++m) //累加上 采样点[i]  对128维度的贡献值
	//          {
	//              //orient_wt是幅值方向在8个选定方向上的影响值。例如PI/3只对PI/4和PI/2有影响，权值视(0,PI/4)远近从（1，0）渐变
	//              orient_wght[m] = max((1.0 - (1.0*fabs(diff[m % 8])) / orient_bin_spacing), 0.0);
	//              feat_desc[m] = feat_desc[m] + orient_wght[m] * pos_wght[m] * g*mag_sample;
	//          }
	//          free(pos_wght);
	//      }
	//      free(feat);
	//      free(featcenter);
	//      //归一化、抑制、再归一化
	//      float norm = GetVecNorm(feat_desc, 128);
	//      for (int m = 0; m<128; m++)
	//      {
	//          feat_desc[m] /= norm;
	//          if (feat_desc[m]>0.2)
	//              feat_desc[m] = 0.2;
	//      }
	//      norm = GetVecNorm(feat_desc, 128);
	//      for (int m = 0; m<128; m++)
	//      {
	//          feat_desc[m] /= norm;
	//      }
	//      p->descrip = feat_desc;
	//      p = p->next;
	//  }
	//  free(feat_grid);
	//  free(feat_samples);
	//}


	//第五步
	void ExtractFeatureDescriptors(ImageOctave* GaussianPyr)
	{
		float feat_window = 2 * GridSpacing;
		float orient_bin_spacing = CV_PI / 4;//bin区间间隔 四分之PI

		//8个方向的角度值
		float orient_angles[8] = { -CV_PI,-CV_PI + orient_bin_spacing,-CV_PI * 0.5, -orient_bin_spacing,
			0.0, orient_bin_spacing, CV_PI * 0.5,  CV_PI + orient_bin_spacing };

		//产生描述字中心各点坐标  
		float* feat_grid = (float*)malloc(2 * 16 * sizeof(float));
		for (int i = 0; i < GridSpacing; i++)
		{
			for (int j = 0; j < 2 * GridSpacing; j += 2)
			{
				feat_grid[i * 2 * GridSpacing + j] = -6.0 + i * GridSpacing;
				feat_grid[i * 2 * GridSpacing + j + 1] = -6.0 + 0.5 * j * GridSpacing;
			}
		}

		//产生网格  16个大格子。每个大格子里有16个小格子。。。 
		float* feat_samples = (float*)malloc(2 * 256 * sizeof(float));
		for (int i = 0; i < 4 * GridSpacing; i++)
		{
			for (int j = 0; j < 8 * GridSpacing; j += 2)
			{
				feat_samples[i * 8 * GridSpacing + j] = -(2 * GridSpacing - 0.5) + i;
				feat_samples[i * 8 * GridSpacing + j + 1] = -(2 * GridSpacing - 0.5) + 0.5 * j;
			}
		}

		// p指向第一个结点 
		Keypoint* p = keyDescriptors;
		while (p)
		{
			float scale = (GaussianPyr[p->octave].Octave)[p->level].absolute_sigma;
			float sine = -sin(p->ori);
			float cosine = cos(p->ori);

			//计算大格中心点旋转之后的位置        featcenter[]数组 
			float* featcenter = (float*)malloc(2 * 16 * sizeof(float));
			for (int i = 0; i < GridSpacing; i++)
			{
				for (int j = 0; j < 2 * GridSpacing; j += 2)
				{
					float x = feat_grid[i * 2 * GridSpacing + j];
					float y = feat_grid[i * 2 * GridSpacing + j + 1];
					featcenter[i * 2 * GridSpacing + j] = ((cosine * x + sine * y) + p->sx);
					featcenter[i * 2 * GridSpacing + j + 1] = ((-sine * x + cosine * y) + p->sy);
				}
			}

			//网格中心点旋转后的位置       feat[]数组
			float* feat = (float*)malloc(2 * 256 * sizeof(float));
			for (int i = 0; i < 64 * GridSpacing; i++, i++)
			{
				float x = feat_samples[i];
				float y = feat_samples[i + 1];
				feat[i] = ((cosine * x + sine * y) + p->sx);
				feat[i + 1] = ((-sine * x + cosine * y) + p->sy);
			}


			//初始化特征描述子  feature_descriptors
			float* feat_desc = (float*)malloc(128 * sizeof(float));
			for (int i = 0; i < 128; i++)
			{
				feat_desc[i] = 0.0;
			}


			//256个网格采样点 *2 
			for (int i = 0; i < 512; i += 2)
			{
				//插值计算网格中心点梯度方向
				/*
				0   12   0
				21  22   23
				0   32   0   具体插值策略如图示
				*/
				float mag_sample = 0, grad_sample = 0, x_sample = 0, y_sample = 0; {
					x_sample = feat[i];
					y_sample = feat[i + 1];
					float sample12 = getPixelBI(((GaussianPyr[p->octave].Octave)[p->level]).Level, x_sample, y_sample - 1);
					float sample21 = getPixelBI(((GaussianPyr[p->octave].Octave)[p->level]).Level, x_sample - 1, y_sample);
					float sample22 = getPixelBI(((GaussianPyr[p->octave].Octave)[p->level]).Level, x_sample, y_sample);
					float sample23 = getPixelBI(((GaussianPyr[p->octave].Octave)[p->level]).Level, x_sample + 1, y_sample);
					float sample32 = getPixelBI(((GaussianPyr[p->octave].Octave)[p->level]).Level, x_sample, y_sample + 1);
					float diff_x = sample23 - sample21;
					float diff_y = sample32 - sample12;
					mag_sample = sqrt(diff_x * diff_x + diff_y * diff_y);
					grad_sample = atan(diff_y / diff_x);
					if (diff_x < 0)
						grad_sample += CV_PI;
					if (grad_sample >= CV_PI)   grad_sample -= 2 * CV_PI;
				}

				//663-710行代码，表示不透彻

								//需要改进调优的地方！！！！！！！！！！
								// 计算采样点对于4*4个种子点的权重，其实只有邻近的种子点会有权重
								// 这类似 hog算子的 block 归一化。
								// float[128]，对每个种子点内的8方向权值是一样的。
				float* pos_wght = (float*)malloc(8 * GridSpacing * GridSpacing * sizeof(float)); {
					float* x_wght = (float*)malloc(GridSpacing * GridSpacing * sizeof(float)); // float[16]
					float* y_wght = (float*)malloc(GridSpacing * GridSpacing * sizeof(float)); // float[16]
					for (int m = 0; m < 32; ++m, ++m)
					{
						//(x,y)是16个种子点的位置
						float x = featcenter[m];
						float y = featcenter[m + 1];
						//GridSpacing是不是少乘了一个0.5？？？？
						x_wght[m / 2] = max(1 - (fabs(x - x_sample) * 1.0 / GridSpacing), 0.0);   //只有采样点附近的种子点 的x_wt值才是非0；
						y_wght[m / 2] = max(1 - (fabs(y - y_sample) * 1.0 / GridSpacing), 0.0);
					}
					for (int m = 0; m < 16; ++m)
						for (int n = 0; n < 8; ++n)
							pos_wght[m * 8 + n] = x_wght[m] * y_wght[m];
					free(x_wght);
					free(y_wght);
				}

				//首先旋转梯度场到主方向，然后计算幅值到各个方向的差值
				float diff[8];
				for (int m = 0; m < 8; ++m)
				{
					float angle = grad_sample - (p->ori) - orient_angles[m] + CV_PI; //差值+pi
					float temp = angle / (2.0 * CV_PI);
					angle -= (int)(temp) * (2.0 * CV_PI);
					diff[m] = angle - CV_PI;
				}


				// 计算高斯权重  
				float x = p->sx, y = p->sy;                                                                    //feat_window=2*GridSpacing=8  .原代码是不是少写了sqrt
				float g = exp(-((x_sample - x) * (x_sample - x) + (y_sample - y) * (y_sample - y)) / (2 * feat_window * feat_window)) / sqrt(2 * CV_PI * feat_window * feat_window);


				float orient_wght[128];
				for (int m = 0; m < 128; ++m) //累加上 采样点[i]  对128维度的贡献值
				{
					//orient_wt是幅值方向在8个选定方向上的影响值。例如PI/3只对PI/4和PI/2有影响，权值视(0,PI/4)远近从（1，0）渐变
					orient_wght[m] = max((1.0 - (1.0 * fabs(diff[m % 8])) / orient_bin_spacing), 0.0);
					feat_desc[m] = feat_desc[m] + orient_wght[m] * pos_wght[m] * g * mag_sample;
				}
				free(pos_wght);
			}
			free(feat);
			free(featcenter);

			//归一化、抑制、再归一化
			float norm = GetVecNorm(feat_desc, 128);
			for (int m = 0; m < 128; m++)
			{
				feat_desc[m] /= norm;
				if (feat_desc[m] > 0.2)
					feat_desc[m] = 0.2;
			}
			norm = GetVecNorm(feat_desc, 128);
			for (int m = 0; m < 128; m++)
			{
				feat_desc[m] /= norm;
			}
			p->descrip = feat_desc;
			p = p->next;
			descrip_lenth++;
		}
		free(feat_grid);
		free(feat_samples);
	}


	//释放内存
	void release() {
		free(DOGoctaves);
		free(mag_pyr);
		free(grad_pyr);
		free(keypoints);
		free(keyDescriptors);
	}

};



//------------------------------------------------------------------------
int main(void)
{
	sift* sf = new sift();
	sift* sf2 = new sift();
	string img_path1 = "262A2643.tif", img_path2 = "262A2644.tif";
	Mat img1 = imread(img_path1), img2 = imread(img_path2);
	//cvtColor(img1, img1, CV_RGB2GRAY);

	{
		Mat src, grey, image1, DoubleSizeImage;     //彩色原图； 灰度原图 ； 用于show点的图 ；  
		Mat init_Mat, bottom_Mat;//灰度浮点原图； 金字塔底层图；
		int rows, cols;
		//读取图片、做图片赋值准备
		{
			src = imread(img_path1, 1);
			if (src.rows == 0) {
				return -1;
			}
			cvtColor(src, grey, CV_BGR2GRAY);

			image1 = src.clone();       //此处image1是彩色原图的复原
			DoubleSizeImage = Mat(2 * src.rows, 2 * src.cols, CV_32FC3);
			init_Mat = Mat(src.rows, src.cols, CV_32FC1);
			grey.convertTo(init_Mat, CV_32FC1);
			cv::normalize(init_Mat, init_Mat, 1.0, 0.0, NORM_MINMAX);       //init_Mat是归一化的灰度图
		}

		//求金字塔阶数
		int numoctaves = 4;

		{
			int dim = min(init_Mat.rows, init_Mat.cols);
			numoctaves = (int)(log((double)dim) / log(2.0)) - 2;    //金字塔阶数  
			numoctaves = min(numoctaves, MAXOCTAVES);
			sf->numoctaves = numoctaves;
		}

		//SIFT算法第一步，预滤波除噪声，建立金字塔底层  
		bottom_Mat = sf->ScaleInitImage(init_Mat);
		//SIFT算法第二步，建立Guassian金字塔和DOG金字塔  
		ImageOctave* Gaussianpyr;
		Gaussianpyr = sf->BuildGaussianOctaves(bottom_Mat);
		//SIFT算法第三步：特征点位置检测，最后确定特征点的位置  
		int keycount = sf->DetectKeypoint(numoctaves, Gaussianpyr);
		printf("the keypoints number are %d ;/n", keycount);
		sf->DisplayKeypointLocation(image1, Gaussianpyr);
		img1 = image1.clone();
		image1.convertTo(image1, CV_32FC3);   //自己写的
		cv::normalize(image1, image1, 1.0, 0.0, NORM_MINMAX);
		doubleSizeImageColor(image1, DoubleSizeImage);

		//SIFT算法第四步：计算高斯图像的梯度方向和幅值，计算各个特征点的主方向  
		sf->ComputeGrad_DirecandMag(Gaussianpyr);
		sf->AssignTheMainOrientation(numoctaves, Gaussianpyr, sf->mag_pyr, sf->grad_pyr);
		image1 = src.clone();
		sf->DisplayOrientation(image1, Gaussianpyr);

		//SIFT算法第五步：抽取各个特征点处的特征描述字  
		sf->ExtractFeatureDescriptors(Gaussianpyr);

		//释放内存
		src.release(), grey.release(), image1.release(), DoubleSizeImage.release(), init_Mat.release(), bottom_Mat.release();
		//free(Gaussianpyr);
	}

	{
		Mat src, grey, image1, DoubleSizeImage;     //彩色原图； 灰度原图 ； 用于show点的图 ；  
		Mat init_Mat, bottom_Mat;//灰度浮点原图； 金字塔底层图；
		int rows, cols;

		//读取图片、做图片赋值准备
		{
			src = imread(img_path2, 1);
			if (src.rows == 0) {
				return -1;
			}
			cvtColor(src, grey, CV_BGR2GRAY);

			image1 = src.clone();       //此处image1是彩色原图的复原
			DoubleSizeImage = Mat(2 * src.rows, 2 * src.cols, CV_32FC3);
			init_Mat = Mat(src.rows, src.cols, CV_32FC1);
			grey.convertTo(init_Mat, CV_32FC1);
			cv::normalize(init_Mat, init_Mat, 1.0, 0.0, NORM_MINMAX);       //init_Mat是归一化的灰度图
		}

		//求金字塔阶数
		int numoctaves = 4;

		{
			int dim = min(init_Mat.rows, init_Mat.cols);
			numoctaves = (int)(log((double)dim) / log(2.0)) - 2;    //金字塔阶数  
			numoctaves = min(numoctaves, MAXOCTAVES);
			sf2->numoctaves = numoctaves;
		}

		//SIFT算法第一步，预滤波除噪声，建立金字塔底层  
		bottom_Mat = sf2->ScaleInitImage(init_Mat);
		//SIFT算法第二步，建立Guassian金字塔和DOG金字塔  
		ImageOctave* Gaussianpyr;
		Gaussianpyr = sf2->BuildGaussianOctaves(bottom_Mat);
		//SIFT算法第三步：特征点位置检测，最后确定特征点的位置  
		int keycount = sf2->DetectKeypoint(numoctaves, Gaussianpyr);
		printf("the keypoints number are %d ;/n", keycount);
		sf2->DisplayKeypointLocation(image1, Gaussianpyr);
		img2 = image1.clone();
		image1.convertTo(image1, CV_32FC3);   //自己写的
		cv::normalize(image1, image1, 1.0, 0.0, NORM_MINMAX);
		doubleSizeImageColor(image1, DoubleSizeImage);

		//SIFT算法第四步：计算高斯图像的梯度方向和幅值，计算各个特征点的主方向  
		sf2->ComputeGrad_DirecandMag(Gaussianpyr);
		sf2->AssignTheMainOrientation(numoctaves, Gaussianpyr, sf2->mag_pyr, sf2->grad_pyr);
		image1 = src.clone();
		sf2->DisplayOrientation(image1, Gaussianpyr);

		//SIFT算法第五步：抽取各个特征点处的特征描述字  
		sf2->ExtractFeatureDescriptors(Gaussianpyr);

		//释放内存
		src.release(), grey.release(), image1.release(), DoubleSizeImage.release(), init_Mat.release(), bottom_Mat.release();
		//free(Gaussianpyr);
	}


	//合成图

	int row1 = img1.rows, col1 = img1.cols, row2 = img2.rows, col2 = img2.cols;
	int row = max(row1, row2), col = col1 + col2 + 100;
	Mat img = Mat(row, col, CV_8UC3);
	for (int r = 0; r < row1; r++) {
		for (int c = 0; c < col1; c++) {
			img.ptr<uchar>(r)[c * 3 + 0] = img1.ptr<uchar>(r)[c * 3 + 0];
			img.ptr<uchar>(r)[c * 3 + 1] = img1.ptr<uchar>(r)[c * 3 + 1];
			img.ptr<uchar>(r)[c * 3 + 2] = img1.ptr<uchar>(r)[c * 3 + 2];
		}
	}
	for (int r = 0; r < row2; r++) {
		for (int c = 0; c < col2; c++) {
			img.ptr<uchar>(r)[3 * col1 + 300 + c * 3 + 0] = img2.ptr<uchar>(r)[c * 3 + 0];
			img.ptr<uchar>(r)[3 * col1 + 300 + c * 3 + 1] = img2.ptr<uchar>(r)[c * 3 + 1];
			img.ptr<uchar>(r)[3 * col1 + 300 + c * 3 + 2] = img2.ptr<uchar>(r)[c * 3 + 2];
		}
	}


	//绘制匹配线
	cout << "sf1描述子长度：  " << sf->descrip_lenth << endl;
	cout << "sf2描述子长度：  " << sf2->descrip_lenth << endl;
	vector<matchPoint> match = compute_macth(sf->keyDescriptors, sf2->keyDescriptors, 0.3);
	for (matchPoint mp : match) {
		line(img, mp.p1, Point2i(mp.p2.x + col1 + 100, mp.p2.y), Scalar(255, 255, 0), 3, 8, 0);
	}


	//图像拼接
	vector<Point2f> imagePoints1, imagePoints2;
	for (matchPoint mp : match) {
		imagePoints1.push_back(mp.p1);
		imagePoints2.push_back(mp.p2);
	}
	Mat homo = findHomography(imagePoints2, imagePoints1, CV_RANSAC);
	////也可以使用getPerspectiveTransform方法获得透视变换矩阵，不过要求只能有4个点，效果稍差  
	//Mat   homo=getPerspectiveTransform(imagePoints1,imagePoints2);  
	cout << "变换矩阵为：\n" << homo << endl << endl; //输出映射矩阵     

												//图像配准  
	Mat imageTransform1, imageTransform2;
	warpPerspective(img2, imageTransform2, homo, Size(img.cols, img1.rows));

	double x11 = 0, y11 = 0, x21 = 0, y21 = 0; {
		int x1 = 0, y1 = 0, x2 = 0, y2 = img2.rows;
		x11 = x1 * homo.ptr<double>(0)[0] + y1 * homo.ptr<double>(0)[1] + homo.ptr<double>(0)[2];
		y11 = x1 * homo.ptr<double>(1)[0] + y1 * homo.ptr<double>(1)[1] + homo.ptr<double>(1)[2];
		double w11 = x1 * homo.ptr<double>(2)[0] + y1 * homo.ptr<double>(2)[1] + homo.ptr<double>(2)[2];
		x21 = x2 * homo.ptr<double>(0)[0] + y2 * homo.ptr<double>(0)[1] + homo.ptr<double>(0)[2];
		y21 = x2 * homo.ptr<double>(1)[0] + y2 * homo.ptr<double>(1)[1] + homo.ptr<double>(1)[2];
		double w21 = x2 * homo.ptr<double>(2)[0] + y2 * homo.ptr<double>(2)[1] + homo.ptr<double>(2)[2];
		x11 = x11 / w11; y11 = y11 / w11;
		x21 = x21 / w21; y21 = y21 / w21;
	}
	int start = min(x11, x21), end = img1.cols;
	double processWidth = end - start;
	//warpPerspective(image01, imageTransform2, adjustMat*homo, Size(image02.cols*1.3, image02.rows*1.8));
	imageTransform1 = imageTransform2.clone();
	for (int r = 0; r < img1.rows; r++) {
		for (int c = 0; c < end; c++) {
			imageTransform1.ptr(r)[3 * c] = img1.ptr(r)[3 * c];
			imageTransform1.ptr(r)[3 * c + 1] = img1.ptr(r)[3 * c + 1];
			imageTransform1.ptr(r)[3 * c + 2] = img1.ptr(r)[3 * c + 2];
			if ((imageTransform2.ptr(r)[3 * c] + imageTransform2.ptr(r)[3 * c + 1] + imageTransform2.ptr(r)[3 * c + 2]) == 0) {
				imageTransform2.ptr(r)[3 * c] = img1.ptr(r)[3 * c];
				imageTransform2.ptr(r)[3 * c + 1] = img1.ptr(r)[3 * c + 1];
				imageTransform2.ptr(r)[3 * c + 2] = img1.ptr(r)[3 * c + 2];
			}
			else {
				double alpha = (processWidth - c + start) / processWidth;
				imageTransform2.ptr(r)[3 * c] = alpha * img1.ptr(r)[3 * c] + (1.0 - alpha) * imageTransform2.ptr(r)[3 * c];
				imageTransform2.ptr(r)[3 * c + 1] = alpha * img1.ptr(r)[3 * c + 1] + (1.0 - alpha) * imageTransform2.ptr(r)[3 * c + 1];
				imageTransform2.ptr(r)[3 * c + 2] = alpha * img1.ptr(r)[3 * c + 2] + (1.0 - alpha) * imageTransform2.ptr(r)[3 * c + 2];
			}
		}
	}

	imshow("2", imageTransform2);
	imshow("1", imageTransform1);
	waitKey(0);
	img.release();
	homo.release();
	imageTransform1.release();
	imageTransform2.release();
	sf->release();
	sf2->release();

	return 0;
}

