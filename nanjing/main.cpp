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
	//DOG pyr��DOG���Ӽ���򵥣��ǳ߶ȹ�һ����LoG���ӵĽ��ơ�
	ImageOctave* mag_pyr; //�ݶ���
	ImageOctave* grad_pyr;//�ݶȷ�����
	//����������������  
	Keypoint* keypoints = NULL;      //������ʱ�洢�������λ�õ�  
	Keypoint* keyDescriptors = NULL; //���ڴ洢���������㣨ͬ�㲻ͬ�����ڲ�ͬ�㣩
	//4*4 
#define GridSpacing 4  


//��һ��
	Mat ScaleInitImage(Mat im)  //����imΪ�Ҷ�ͼ��32F
	{
		double preblur_sigma;
		Mat imMat = im.clone();
		int gaussdim = (int)max(3.0, 2.0 * GAUSSKERN * INITSIGMA + 1.0);//��˹�˵ĳߴ�
		gaussdim = 2 * (gaussdim / 2) + 1;
		//Ԥ�˲�������
		GaussianBlur(imMat, imMat, Size(gaussdim, gaussdim), INITSIGMA);

		//�����������ֱ���д�����ʼ���Ŵ�ԭʼͼ�������ԭͼ������Ͻ��к�������
		if (DOUBLE_BASE_IMAGE_SIZE)
		{
			Mat bottom_Mat = doubleSizeImage2(imMat);//���Բ�ֵ�����ϲ���
			preblur_sigma = 1.0;
			gaussdim = (int)max(3.0, 2.0 * GAUSSKERN * preblur_sigma + 1.0);//��˹�˵ĳߴ�
			gaussdim = 2 * (gaussdim / 2) + 1;
			GaussianBlur(bottom_Mat, bottom_Mat, Size(gaussdim, gaussdim), preblur_sigma);
			return bottom_Mat;
		}
		else
		{
			preblur_sigma = 1.0;
			gaussdim = (int)max(3.0, 2.0 * GAUSSKERN * preblur_sigma + 1.0);//��˹�˵ĳߴ�
			BlurImage(imMat, imMat, preblur_sigma); //�õ�����������ײ㣺ԭʼͼ���С  
			return imMat;
		}
	}

	//SIFT�㷨�ڶ���  
	ImageOctave* BuildGaussianOctaves(Mat img)
	{
		//�����ڴ� ��octaves , DOGoctaves
		ImageOctave* octaves;
		octaves = new ImageOctave[numoctaves];
		DOGoctaves = new ImageOctave[numoctaves];
		//��������ڵĲ���
		int num_peroc_levels = SCALESPEROCTAVE + 3;
		int num_perdog_levels = num_peroc_levels - 1;
		//
				//��������
		Mat tempMat = img.clone(), dst, temp;
		float init_sigma = pow(2, 1.0 / 2);


		double k = pow(2, 1.0 / ((float)SCALESPEROCTAVE));  //�����   ����2 
//
		//��ÿһ�׽�����ͼ���н�����ͬ�ĳ߶�ͼ��  
		for (int i = 0; i < numoctaves; i++)
		{
			{
				//�����ڴ�
				octaves[i].Octave = new ImageLevel[num_peroc_levels];
				DOGoctaves[i].Octave = new ImageLevel[num_perdog_levels];
				//���Ƚ���������ÿһ���ݵ���ײ㣬����0���ݵ���ײ��Ѿ�������  
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
			float sigma_act, absolute_sigma;    //ÿ��ֱ��������ǰͼ���ϵ�blurֵ    ��   �߶ȿռ��еľ���ֵ

			//�����������������ͼ��  
			for (int j = 1; j < SCALESPEROCTAVE + 3; j++)
			{
				dst = Mat(tempMat.rows, tempMat.cols, CV_32FC1);//���ڴ洢��˹��  
				temp = Mat(tempMat.rows, tempMat.cols, CV_32FC1);//���ڴ洢DOG��  

				sigma_act = sqrt(k * k - 1) * sigma;
				sigma = k * sigma;

				(octaves[i].Octave)[j].levelsigma = sigma;
				(octaves[i].Octave)[j].absolute_sigma = sigma * (octaves[i].subsample);
				// (octaves[i].Octave)[j].absolute_sigma = k *((octaves[i].Octave)[j-1].absolute_sigma);

				//������˹��  
				int gaussdim = (int)max(3.0, 2.0 * GAUSSKERN * sigma_act + 1.0);//��˹�˵ĳߴ�
				gaussdim = 2 * (gaussdim / 2) + 1;
				GaussianBlur((octaves[i].Octave)[j - 1].Level, dst, Size(gaussdim, gaussdim), sigma_act);
				//BlurImage((octaves[i].Octave)[j - 1].Level, dst, sigma_act);
				(octaves[i].Octave)[j].levelsigmalength = gaussdim;
				(octaves[i].Octave)[j].Level = dst;

				//����DOG��  
				temp = ((octaves[i].Octave)[j]).Level - ((octaves[i].Octave)[j - 1]).Level;
				//subtract(((octaves[i].Octave)[j]).Level, ((octaves[i].Octave)[j - 1]).Level, temp, 0);
				((DOGoctaves[i].Octave)[j - 1]).Level = temp;
			}

			tempMat = halfSizeImage(((octaves[i].Octave)[SCALESPEROCTAVE].Level));
		}
		return octaves;
	}

	//SIFT�㷨��������������λ�ü�⣬  
	int DetectKeypoint(int numoctaves, ImageOctave* GaussianPyr)
	{
		//��������DOG��ֵ����������ʱȵ���ֵ  
		double curvature_threshold = ((CURVATURE_THRESHOLD + 1) * (CURVATURE_THRESHOLD + 1)) / CURVATURE_THRESHOLD;
		curvature_threshold = 10;
#define ImLevels(OCTAVE,LEVEL,ROW,COL)  ((float)(DOGoctaves[(OCTAVE)].Octave[(LEVEL)].Level).ptr<float>(ROW)[COL])
		int   keypoint_count = 0;
		for (int i = 0; i < numoctaves; i++)
		{
			for (int j = 1; j < SCALESPEROCTAVE + 1; j++)//ȡ�м��scaleperoctave����  
			{
				//��ͼ�����Ч������Ѱ�Ҿ��������������ľֲ����ֵ   
				int dim = (int)(0.5 * ((GaussianPyr[i].Octave)[j].levelsigmalength) + 0.5);
				for (int m = dim; m < ((DOGoctaves[i].row) - dim); m++)
					for (int n = dim; n < ((DOGoctaves[i].col) - dim); n++)
					{
						if (fabs(ImLevels(i, j, m, n)) >= CONTRAST_THRESHOLD)
						{
							if (ImLevels(i, j, m, n) != 0.0)  //1�������Ƿ���  
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
									(inf_val <= ImLevels(i, j - 1, m + 1, n + 1)) &&    //�ײ��С�߶�9  

									(inf_val <= ImLevels(i, j, m - 1, n - 1)) &&
									(inf_val <= ImLevels(i, j, m, n - 1)) &&
									(inf_val <= ImLevels(i, j, m + 1, n - 1)) &&
									(inf_val <= ImLevels(i, j, m - 1, n)) &&
									(inf_val <= ImLevels(i, j, m + 1, n)) &&
									(inf_val <= ImLevels(i, j, m - 1, n + 1)) &&
									(inf_val <= ImLevels(i, j, m, n + 1)) &&
									(inf_val <= ImLevels(i, j, m + 1, n + 1)) &&     //��ǰ��8  

									(inf_val <= ImLevels(i, j + 1, m - 1, n - 1)) &&
									(inf_val <= ImLevels(i, j + 1, m, n - 1)) &&
									(inf_val <= ImLevels(i, j + 1, m + 1, n - 1)) &&
									(inf_val <= ImLevels(i, j + 1, m - 1, n)) &&
									(inf_val <= ImLevels(i, j + 1, m, n)) &&
									(inf_val <= ImLevels(i, j + 1, m + 1, n)) &&
									(inf_val <= ImLevels(i, j + 1, m - 1, n + 1)) &&
									(inf_val <= ImLevels(i, j + 1, m, n + 1)) &&
									(inf_val <= ImLevels(i, j + 1, m + 1, n + 1))     //��һ���߶�9          
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
										))      //2������26���м�ֵ��  
								{
									//�˴��ɴ洢  
									//���ͣ���Ч��ֵ��ĳ߶ȿռ乹��Ӧ����ƽ̹������ ����� �������Ǿ��������ƽ̹��������Ӧ����0�㡣
									if (fabs(ImLevels(i, j, m, n)) >= CONTRAST_THRESHOLD)
									{
										//������������������������㹻�����ʱȣ�CURVATURE_THRESHOLD=10.0�����ȼ���Hessian����  
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
										if ((Det_H >= 0.0) && (curvature_ratio <= curvature_threshold))  //���õ������������������������  
										{
											//����洢�������Լ�����������������  
											keypoint_count++;
											Keypoint* k;
											k = (Keypoint*)malloc(sizeof(struct Keypoint));
											k->next = keypoints;
											keypoints = k;
											k->row = m * (GaussianPyr[i].subsample);
											k->col = n * (GaussianPyr[i].subsample);
											k->sy = m;    //��  
											k->sx = n;    //��  
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

	//��ͼ���У���ʾSIFT�������λ��  
	void DisplayKeypointLocation(Mat image, ImageOctave* GaussianPyr)
	{

		Keypoint* p = keypoints; // pָ���һ�����  
		while (p) // û����β  
		{
			line(image, Point2i((int)((p->col) - 3), (int)(p->row)), Point2i((int)((p->col) + 3), (int)(p->row)), Scalar(0, 0, 255), 1, 8, 0);
			line(image, Point2i((int)(p->col), (int)((p->row) - 3)), Point2i((int)(p->col), (int)((p->row) + 3)), Scalar(0, 0, 255), 1, 8, 0);
			p = p->next;
		}
	}



	//Sift���Ĳ�׼������

	// �����˹���������ݶȷ������ֵ����ʱkeypoints�Ѿ������
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
			for (int j = 1; j < SCALESPEROCTAVE + 1; j++)//ȡ�м��scaleperoctave����  
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
						//�����ֵ  
						(gradx).ptr<float>(m)[n] = 0.5 * (ImLevels(i, j, m, n + 1) - ImLevels(i, j, m, n - 1));  //dx  
						(grady).ptr<float>(m)[n] = 0.5 * (ImLevels(i, j, m + 1, n) - ImLevels(i, j, m - 1, n));  //dy  
						(Mag).ptr<float>(m)[n] = sqrt(GRADX(m, n) * GRADX(m, n) + GRADY(m, n) * GRADY(m, n));  //mag  

																										   //atan�ķ�Χ�� ��-PI/2��PI/2 ��
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

	//Ѱ���뷽��ֱ��ͼ���������ȷ����index   
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

	// ���ݶȷֲ�ֱ��ͼ������ֵ�˲�  
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

	//����������Ͼ�ȷ�Ƕ�
	bool InterpolateOrientation(double left, double middle, double right, double* degreeCorrection, double* peakValue)
	{
		double a = ((left + right) - 2.0 * middle) / 2.0;   //���������ϵ��a  
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

	//SIFT�㷨���Ĳ�����������������������ȷ��������  
	void AssignTheMainOrientation(int numoctaves, ImageOctave* GaussianPyr, ImageOctave* mag_pyr, ImageOctave* grad_pyr)
	{
		int num_bins = 36;
		float hist_step = (2.0 * CV_PI) / (num_bins + 0.0);   //�Ƕ���ɢ���Ĳ���
		float hist_orient[36];          //�Ƕ���ɢ������ʼ �Ƕȡ�
		for (int i = 0; i < 36; i++) {
			hist_orient[i] = -CV_PI + i * hist_step;
		}

		//��Եzeropad֮��Ϊ��Ч����
		float sigma1 = (((GaussianPyr[0].Octave)[SCALESPEROCTAVE].absolute_sigma)) / (GaussianPyr[0].subsample);
		int zero_pad = (int)((int(std::max(3.0, 2 * GAUSSKERN * sigma1 + 1.0))) / 2 + 1);
#define ImLevels(OCTAVE,LEVEL,ROW,COL)  ((float)((GaussianPyr[(OCTAVE)].Octave[(LEVEL)].Level).ptr<float>(ROW)[COL]))
		int keypoint_count = 0;
		Keypoint* p = keypoints;

		//��������ֱ��ͼ����  
		double* orienthist = new double[36];
		//double* orienthist = (double *)malloc(36 * sizeof(double));

		while (p)
		{
			int i = p->octave;
			int j = p->level;   //��ֵ����dog�ĵڼ��㣬������Gauss�ĵڼ���.  ������1 ��2��
			int m = p->sy;   //��  
			int n = p->sx;   //��  
			if ((m >= zero_pad) && (m < GaussianPyr[i].row - zero_pad) &&
				(n >= zero_pad) && (n < GaussianPyr[i].col - zero_pad))       //��Ч����
			{
				float sigma = (((GaussianPyr[i].Octave)[j].absolute_sigma)) / (GaussianPyr[i].subsample);
				//������ά��˹ģ��  
				Mat mat = GaussianKernel2D(sigma);
				int dim = (int)max(3.0, 2.0 * GAUSSKERN * sigma + 1.0);   dim = 2 * (dim / 2) + 1;
				dim = dim / 2;
				//�������ڴ洢Patch��ֵ�ͷ���Ŀռ�  
#define MAT(ROW,COL) ((float)(mat).ptr<float>(ROW)[COL])
				for (int sw = 0; sw < 36; ++sw)
				{
					orienthist[sw] = 0.0;
				}
				//�����������Χͳ���ݶȷ���  
				for (int x = m - dim, mm = 0; x <= (m + dim); x++, mm++)
					for (int y = n - dim, nn = 0; y <= (n + dim); y++, nn++)
					{
						//���������㴦�ķ�ֵ  
						double mag = ((mag_pyr[i].Octave)[j - 1]).Level.ptr<float>(x)[y];
						double Ori = ((grad_pyr[i].Octave)[j - 1]).Level.ptr<float>(x)[y];
						int binIdx = FindClosestRotationBin(36, Ori);                   //�õ������з��������ֱ����  
						orienthist[binIdx] = orienthist[binIdx] + 1.0 * mag * MAT(mm, nn);//���ø�˹��Ȩ�ۼӽ�ֱ��ͼ��Ӧ�Ŀ�  
					}

				// Find peaks in the orientation histogram using nonmax suppression.  

				//��orienthist������ֵ�˲���
				AverageWeakBins(orienthist, 36);

				// ����Ѱ������� 
				double maxGrad = 0.0;   //������Ȩֵ
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
				//�������߾�ȷ���maxDegree
				if ((InterpolateOrientation(orienthist[maxBin == 0 ? (36 - 1) : (maxBin - 1)],
					orienthist[maxBin],
					orienthist[(maxBin + 1) % 36],
					&maxDegreeCorrection,
					&maxPeakValue)) == false)
				{
					printf("BUG: Parabola fitting broken");
				}


				//�����ֵ�����ɸѡ��0.8*maxPeakValue���ϵĶ���
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
						continue; // ��Ȼ����0.8���ǲ��Ǿֲ����ֵ  
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

						//�����ڴ����´洢������  
						Keypoint* k = new Keypoint();
						k->next = keyDescriptors;
						keyDescriptors = k;
						k->row = p->row;
						k->col = p->col;
						k->sy = p->sy;    //��  
						k->sx = p->sx;    //��  
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

	//��ʾ�����㴦��������  
	void DisplayOrientation(Mat image, ImageOctave* GaussianPyr)
	{
		Keypoint* p = keyDescriptors; // pָ���һ�����  
		while (p) // û����β  
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

			// ����ͷ 
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
	//  float orient_bin_spacing = CV_PI / 4;//bin������ �ķ�֮PI
	//                                       //8������ĽǶ�ֵ
	//  float orient_angles[8] = { -CV_PI,-CV_PI + orient_bin_spacing,-CV_PI*0.5, -orient_bin_spacing,
	//      0.0, orient_bin_spacing, CV_PI*0.5,  CV_PI + orient_bin_spacing };
	//  //�������������ĸ�������  
	//  float *feat_grid = (float *)malloc(2 * 16 * sizeof(float));
	//  for (int i = 0; i<GridSpacing; i++)
	//  {
	//      for (int j = 0; j<2 * GridSpacing; j += 2)
	//      {
	//          feat_grid[i * 2 * GridSpacing + j] = -6.0 + i*GridSpacing;
	//          feat_grid[i * 2 * GridSpacing + j + 1] = -6.0 + 0.5*j*GridSpacing;
	//      }
	//  }
	//  //��������  16������ӡ�ÿ�����������16��С���ӡ����� 
	//  float *feat_samples = (float *)malloc(2 * 256 * sizeof(float));
	//  for (int i = 0; i<4 * GridSpacing; i++)
	//  {
	//      for (int j = 0; j<8 * GridSpacing; j += 2)
	//      {
	//          feat_samples[i * 8 * GridSpacing + j] = -(2 * GridSpacing - 0.5) + i;
	//          feat_samples[i * 8 * GridSpacing + j + 1] = -(2 * GridSpacing - 0.5) + 0.5*j;
	//      }
	//  }
	//  // pָ���һ����� 
	//  Keypoint *p = keyDescriptors;
	//  while (p)
	//  {
	//      float scale = (GaussianPyr[p->octave].Octave)[p->level].absolute_sigma;
	//      float sine = sin(p->ori);
	//      float cosine = cos(p->ori);
	//      //���������ĵ���ת֮���λ��        featcenter[]���� 
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
	//      //�������ĵ���ת���λ��       feat[]����
	//      float *feat = (float *)malloc(2 * 256 * sizeof(float));
	//      for (int i = 0; i<64 * GridSpacing; i++, i++)
	//      {
	//          float x = feat_samples[i];
	//          float y = feat_samples[i + 1];
	//          feat[i] = ((cosine * x + sine * y) + p->sx);
	//          feat[i + 1] = ((-sine * x + cosine * y) + p->sy);
	//      }
	//      //��ʼ������������  feature_descriptors
	//      float *feat_desc = (float *)malloc(128 * sizeof(float));
	//      for (int i = 0; i<128; i++)
	//      {
	//          feat_desc[i] = 0.0;
	//      }
	//      //256����������� *2 
	//      for (int i = 0; i<512; i += 2)
	//      {
	//          //��ֵ�����������ĵ��ݶȷ���
	//          /*
	//          0   12   0
	//          21  22   23
	//          0   32   0   �����ֵ������ͼʾ
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
	//          //663-710�д��룬��ʾ��͸��
	//          //��Ҫ�Ľ����ŵĵط���������������������
	//          // ������������4*4�����ӵ��Ȩ�أ���ʵֻ���ڽ������ӵ����Ȩ��
	//          // ������ hog���ӵ� block ��һ����
	//          // float[128]����ÿ�����ӵ��ڵ�8����Ȩֵ��һ���ġ�
	//          float *pos_wght = (float *)malloc(8 * GridSpacing * GridSpacing * sizeof(float)); {
	//              float *x_wght = (float *)malloc(GridSpacing * GridSpacing * sizeof(float)); // float[16]
	//              float *y_wght = (float *)malloc(GridSpacing * GridSpacing * sizeof(float)); // float[16]
	//              for (int m = 0; m<32; ++m, ++m)
	//              {
	//                  //(x,y)��16�����ӵ��λ��
	//                  float x = featcenter[m];
	//                  float y = featcenter[m + 1];
	//                  //GridSpacing�ǲ����ٳ���һ��0.5��������
	//                  x_wght[m / 2] = max(1 - (fabs(x - x_sample)*1.0 / GridSpacing), 0.0);   //ֻ�в����㸽�������ӵ� ��x_wtֵ���Ƿ�0��
	//                  y_wght[m / 2] = max(1 - (fabs(y - y_sample)*1.0 / GridSpacing), 0.0);
	//              }
	//              for (int m = 0; m<16; ++m)
	//                  for (int n = 0; n<8; ++n)
	//                      pos_wght[m * 8 + n] = x_wght[m] * y_wght[m];
	//              free(x_wght);
	//              free(y_wght);
	//          }
	//          //������ת�ݶȳ���������Ȼ������ֵ����������Ĳ�ֵ
	//          float diff[8];
	//          for (int m = 0; m<8; ++m)
	//          {
	//              float angle = grad_sample - (p->ori) - orient_angles[m] + CV_PI; //��ֵ+pi
	//              float temp = angle / (2.0 * CV_PI);
	//              angle -= (int)(temp) * (2.0 * CV_PI);
	//              diff[m] = angle - CV_PI;
	//          }
	//          // �����˹Ȩ��  
	//          float x = p->sx, y = p->sy;                                                                 //feat_window=2*GridSpacing=8  .ԭ�����ǲ�����д��sqrt
	//          float g = exp(-((x_sample - x)*(x_sample - x) + (y_sample - y)*(y_sample - y)) / (2 * feat_window*feat_window)) / sqrt(2 * CV_PI*feat_window*feat_window);
	//          float orient_wght[128];
	//          for (int m = 0; m<128; ++m) //�ۼ��� ������[i]  ��128ά�ȵĹ���ֵ
	//          {
	//              //orient_wt�Ƿ�ֵ������8��ѡ�������ϵ�Ӱ��ֵ������PI/3ֻ��PI/4��PI/2��Ӱ�죬Ȩֵ��(0,PI/4)Զ���ӣ�1��0������
	//              orient_wght[m] = max((1.0 - (1.0*fabs(diff[m % 8])) / orient_bin_spacing), 0.0);
	//              feat_desc[m] = feat_desc[m] + orient_wght[m] * pos_wght[m] * g*mag_sample;
	//          }
	//          free(pos_wght);
	//      }
	//      free(feat);
	//      free(featcenter);
	//      //��һ�������ơ��ٹ�һ��
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


	//���岽
	void ExtractFeatureDescriptors(ImageOctave* GaussianPyr)
	{
		float feat_window = 2 * GridSpacing;
		float orient_bin_spacing = CV_PI / 4;//bin������ �ķ�֮PI

		//8������ĽǶ�ֵ
		float orient_angles[8] = { -CV_PI,-CV_PI + orient_bin_spacing,-CV_PI * 0.5, -orient_bin_spacing,
			0.0, orient_bin_spacing, CV_PI * 0.5,  CV_PI + orient_bin_spacing };

		//�������������ĸ�������  
		float* feat_grid = (float*)malloc(2 * 16 * sizeof(float));
		for (int i = 0; i < GridSpacing; i++)
		{
			for (int j = 0; j < 2 * GridSpacing; j += 2)
			{
				feat_grid[i * 2 * GridSpacing + j] = -6.0 + i * GridSpacing;
				feat_grid[i * 2 * GridSpacing + j + 1] = -6.0 + 0.5 * j * GridSpacing;
			}
		}

		//��������  16������ӡ�ÿ�����������16��С���ӡ����� 
		float* feat_samples = (float*)malloc(2 * 256 * sizeof(float));
		for (int i = 0; i < 4 * GridSpacing; i++)
		{
			for (int j = 0; j < 8 * GridSpacing; j += 2)
			{
				feat_samples[i * 8 * GridSpacing + j] = -(2 * GridSpacing - 0.5) + i;
				feat_samples[i * 8 * GridSpacing + j + 1] = -(2 * GridSpacing - 0.5) + 0.5 * j;
			}
		}

		// pָ���һ����� 
		Keypoint* p = keyDescriptors;
		while (p)
		{
			float scale = (GaussianPyr[p->octave].Octave)[p->level].absolute_sigma;
			float sine = -sin(p->ori);
			float cosine = cos(p->ori);

			//���������ĵ���ת֮���λ��        featcenter[]���� 
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

			//�������ĵ���ת���λ��       feat[]����
			float* feat = (float*)malloc(2 * 256 * sizeof(float));
			for (int i = 0; i < 64 * GridSpacing; i++, i++)
			{
				float x = feat_samples[i];
				float y = feat_samples[i + 1];
				feat[i] = ((cosine * x + sine * y) + p->sx);
				feat[i + 1] = ((-sine * x + cosine * y) + p->sy);
			}


			//��ʼ������������  feature_descriptors
			float* feat_desc = (float*)malloc(128 * sizeof(float));
			for (int i = 0; i < 128; i++)
			{
				feat_desc[i] = 0.0;
			}


			//256����������� *2 
			for (int i = 0; i < 512; i += 2)
			{
				//��ֵ�����������ĵ��ݶȷ���
				/*
				0   12   0
				21  22   23
				0   32   0   �����ֵ������ͼʾ
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

				//663-710�д��룬��ʾ��͸��

								//��Ҫ�Ľ����ŵĵط���������������������
								// ������������4*4�����ӵ��Ȩ�أ���ʵֻ���ڽ������ӵ����Ȩ��
								// ������ hog���ӵ� block ��һ����
								// float[128]����ÿ�����ӵ��ڵ�8����Ȩֵ��һ���ġ�
				float* pos_wght = (float*)malloc(8 * GridSpacing * GridSpacing * sizeof(float)); {
					float* x_wght = (float*)malloc(GridSpacing * GridSpacing * sizeof(float)); // float[16]
					float* y_wght = (float*)malloc(GridSpacing * GridSpacing * sizeof(float)); // float[16]
					for (int m = 0; m < 32; ++m, ++m)
					{
						//(x,y)��16�����ӵ��λ��
						float x = featcenter[m];
						float y = featcenter[m + 1];
						//GridSpacing�ǲ����ٳ���һ��0.5��������
						x_wght[m / 2] = max(1 - (fabs(x - x_sample) * 1.0 / GridSpacing), 0.0);   //ֻ�в����㸽�������ӵ� ��x_wtֵ���Ƿ�0��
						y_wght[m / 2] = max(1 - (fabs(y - y_sample) * 1.0 / GridSpacing), 0.0);
					}
					for (int m = 0; m < 16; ++m)
						for (int n = 0; n < 8; ++n)
							pos_wght[m * 8 + n] = x_wght[m] * y_wght[m];
					free(x_wght);
					free(y_wght);
				}

				//������ת�ݶȳ���������Ȼ������ֵ����������Ĳ�ֵ
				float diff[8];
				for (int m = 0; m < 8; ++m)
				{
					float angle = grad_sample - (p->ori) - orient_angles[m] + CV_PI; //��ֵ+pi
					float temp = angle / (2.0 * CV_PI);
					angle -= (int)(temp) * (2.0 * CV_PI);
					diff[m] = angle - CV_PI;
				}


				// �����˹Ȩ��  
				float x = p->sx, y = p->sy;                                                                    //feat_window=2*GridSpacing=8  .ԭ�����ǲ�����д��sqrt
				float g = exp(-((x_sample - x) * (x_sample - x) + (y_sample - y) * (y_sample - y)) / (2 * feat_window * feat_window)) / sqrt(2 * CV_PI * feat_window * feat_window);


				float orient_wght[128];
				for (int m = 0; m < 128; ++m) //�ۼ��� ������[i]  ��128ά�ȵĹ���ֵ
				{
					//orient_wt�Ƿ�ֵ������8��ѡ�������ϵ�Ӱ��ֵ������PI/3ֻ��PI/4��PI/2��Ӱ�죬Ȩֵ��(0,PI/4)Զ���ӣ�1��0������
					orient_wght[m] = max((1.0 - (1.0 * fabs(diff[m % 8])) / orient_bin_spacing), 0.0);
					feat_desc[m] = feat_desc[m] + orient_wght[m] * pos_wght[m] * g * mag_sample;
				}
				free(pos_wght);
			}
			free(feat);
			free(featcenter);

			//��һ�������ơ��ٹ�һ��
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


	//�ͷ��ڴ�
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
		Mat src, grey, image1, DoubleSizeImage;     //��ɫԭͼ�� �Ҷ�ԭͼ �� ����show���ͼ ��  
		Mat init_Mat, bottom_Mat;//�Ҷȸ���ԭͼ�� �������ײ�ͼ��
		int rows, cols;
		//��ȡͼƬ����ͼƬ��ֵ׼��
		{
			src = imread(img_path1, 1);
			if (src.rows == 0) {
				return -1;
			}
			cvtColor(src, grey, CV_BGR2GRAY);

			image1 = src.clone();       //�˴�image1�ǲ�ɫԭͼ�ĸ�ԭ
			DoubleSizeImage = Mat(2 * src.rows, 2 * src.cols, CV_32FC3);
			init_Mat = Mat(src.rows, src.cols, CV_32FC1);
			grey.convertTo(init_Mat, CV_32FC1);
			cv::normalize(init_Mat, init_Mat, 1.0, 0.0, NORM_MINMAX);       //init_Mat�ǹ�һ���ĻҶ�ͼ
		}

		//�����������
		int numoctaves = 4;

		{
			int dim = min(init_Mat.rows, init_Mat.cols);
			numoctaves = (int)(log((double)dim) / log(2.0)) - 2;    //����������  
			numoctaves = min(numoctaves, MAXOCTAVES);
			sf->numoctaves = numoctaves;
		}

		//SIFT�㷨��һ����Ԥ�˲��������������������ײ�  
		bottom_Mat = sf->ScaleInitImage(init_Mat);
		//SIFT�㷨�ڶ���������Guassian��������DOG������  
		ImageOctave* Gaussianpyr;
		Gaussianpyr = sf->BuildGaussianOctaves(bottom_Mat);
		//SIFT�㷨��������������λ�ü�⣬���ȷ���������λ��  
		int keycount = sf->DetectKeypoint(numoctaves, Gaussianpyr);
		printf("the keypoints number are %d ;/n", keycount);
		sf->DisplayKeypointLocation(image1, Gaussianpyr);
		img1 = image1.clone();
		image1.convertTo(image1, CV_32FC3);   //�Լ�д��
		cv::normalize(image1, image1, 1.0, 0.0, NORM_MINMAX);
		doubleSizeImageColor(image1, DoubleSizeImage);

		//SIFT�㷨���Ĳ��������˹ͼ����ݶȷ���ͷ�ֵ����������������������  
		sf->ComputeGrad_DirecandMag(Gaussianpyr);
		sf->AssignTheMainOrientation(numoctaves, Gaussianpyr, sf->mag_pyr, sf->grad_pyr);
		image1 = src.clone();
		sf->DisplayOrientation(image1, Gaussianpyr);

		//SIFT�㷨���岽����ȡ���������㴦������������  
		sf->ExtractFeatureDescriptors(Gaussianpyr);

		//�ͷ��ڴ�
		src.release(), grey.release(), image1.release(), DoubleSizeImage.release(), init_Mat.release(), bottom_Mat.release();
		//free(Gaussianpyr);
	}

	{
		Mat src, grey, image1, DoubleSizeImage;     //��ɫԭͼ�� �Ҷ�ԭͼ �� ����show���ͼ ��  
		Mat init_Mat, bottom_Mat;//�Ҷȸ���ԭͼ�� �������ײ�ͼ��
		int rows, cols;

		//��ȡͼƬ����ͼƬ��ֵ׼��
		{
			src = imread(img_path2, 1);
			if (src.rows == 0) {
				return -1;
			}
			cvtColor(src, grey, CV_BGR2GRAY);

			image1 = src.clone();       //�˴�image1�ǲ�ɫԭͼ�ĸ�ԭ
			DoubleSizeImage = Mat(2 * src.rows, 2 * src.cols, CV_32FC3);
			init_Mat = Mat(src.rows, src.cols, CV_32FC1);
			grey.convertTo(init_Mat, CV_32FC1);
			cv::normalize(init_Mat, init_Mat, 1.0, 0.0, NORM_MINMAX);       //init_Mat�ǹ�һ���ĻҶ�ͼ
		}

		//�����������
		int numoctaves = 4;

		{
			int dim = min(init_Mat.rows, init_Mat.cols);
			numoctaves = (int)(log((double)dim) / log(2.0)) - 2;    //����������  
			numoctaves = min(numoctaves, MAXOCTAVES);
			sf2->numoctaves = numoctaves;
		}

		//SIFT�㷨��һ����Ԥ�˲��������������������ײ�  
		bottom_Mat = sf2->ScaleInitImage(init_Mat);
		//SIFT�㷨�ڶ���������Guassian��������DOG������  
		ImageOctave* Gaussianpyr;
		Gaussianpyr = sf2->BuildGaussianOctaves(bottom_Mat);
		//SIFT�㷨��������������λ�ü�⣬���ȷ���������λ��  
		int keycount = sf2->DetectKeypoint(numoctaves, Gaussianpyr);
		printf("the keypoints number are %d ;/n", keycount);
		sf2->DisplayKeypointLocation(image1, Gaussianpyr);
		img2 = image1.clone();
		image1.convertTo(image1, CV_32FC3);   //�Լ�д��
		cv::normalize(image1, image1, 1.0, 0.0, NORM_MINMAX);
		doubleSizeImageColor(image1, DoubleSizeImage);

		//SIFT�㷨���Ĳ��������˹ͼ����ݶȷ���ͷ�ֵ����������������������  
		sf2->ComputeGrad_DirecandMag(Gaussianpyr);
		sf2->AssignTheMainOrientation(numoctaves, Gaussianpyr, sf2->mag_pyr, sf2->grad_pyr);
		image1 = src.clone();
		sf2->DisplayOrientation(image1, Gaussianpyr);

		//SIFT�㷨���岽����ȡ���������㴦������������  
		sf2->ExtractFeatureDescriptors(Gaussianpyr);

		//�ͷ��ڴ�
		src.release(), grey.release(), image1.release(), DoubleSizeImage.release(), init_Mat.release(), bottom_Mat.release();
		//free(Gaussianpyr);
	}


	//�ϳ�ͼ

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


	//����ƥ����
	cout << "sf1�����ӳ��ȣ�  " << sf->descrip_lenth << endl;
	cout << "sf2�����ӳ��ȣ�  " << sf2->descrip_lenth << endl;
	vector<matchPoint> match = compute_macth(sf->keyDescriptors, sf2->keyDescriptors, 0.3);
	for (matchPoint mp : match) {
		line(img, mp.p1, Point2i(mp.p2.x + col1 + 100, mp.p2.y), Scalar(255, 255, 0), 3, 8, 0);
	}


	//ͼ��ƴ��
	vector<Point2f> imagePoints1, imagePoints2;
	for (matchPoint mp : match) {
		imagePoints1.push_back(mp.p1);
		imagePoints2.push_back(mp.p2);
	}
	Mat homo = findHomography(imagePoints2, imagePoints1, CV_RANSAC);
	////Ҳ����ʹ��getPerspectiveTransform�������͸�ӱ任���󣬲���Ҫ��ֻ����4���㣬Ч���Բ�  
	//Mat   homo=getPerspectiveTransform(imagePoints1,imagePoints2);  
	cout << "�任����Ϊ��\n" << homo << endl << endl; //���ӳ�����     

												//ͼ����׼  
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

