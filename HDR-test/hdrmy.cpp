// hdrmy.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

void loadExposureSeq(String, vector<Mat>&, vector<float>&);

int main(int, char** argv)
{
	/*
	vector<Mat> images;
	vector<float> times;

	Mat img1 = imread("memorial0061.png");
	Mat img2 = imread("memorial0069.png");
	Mat img3 = imread("memorial0076.png");

	images.push_back(img1);
	images.push_back(img2);
	images.push_back(img3);

	times.push_back((float)1024);
	times.push_back((float)8);
	times.push_back((float)0.3125);
	*/

	vector<Mat> images;
	vector<float> times;
	loadExposureSeq(argv[1], images, times);
	cout << images.size() << endl;
	

	Mat response;
	Ptr<CalibrateDebevec> calibrate = createCalibrateDebevec();
	calibrate->process(images, response, times);

	Mat hdr;
	Ptr<MergeDebevec> merge_debevec = createMergeDebevec();
	merge_debevec->process(images, hdr, times, response);

	
	Mat ldr;
	Ptr<TonemapDurand> tonemap = createTonemapDurand(2.2f);
	tonemap->process(hdr, ldr);
	
	Mat fusion;
	Ptr<MergeMertens> merge_mertens = createMergeMertens();
	merge_mertens->process(images, fusion);

	fusion = fusion * 255;
	ldr = ldr * 255;

	
	imwrite("fusion.png", fusion);
	//imwrite("ldr.png", ldr);
//	imwrite("hdr.hdr", hdr);
	
	return 0;
}

void loadExposureSeq(String path, vector<Mat>& images, vector<float>& times)
{
	path = path + std::string("/");
	ifstream list_file((path + "list.txt").c_str());
	string name;
	float val;
	while (list_file >> name >> val) {
		Mat img = imread(path + name);
		images.push_back(img);
		times.push_back(1 / val);
	}
	list_file.close();
}