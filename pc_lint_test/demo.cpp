// GridMatch.cpp : Defines the entry point for the console application.

//#define USE_GPU 
#include "Header.h"
#include "gms_matcher.h"
#include <fstream>
using namespace std;

void GmsMatch(Mat &img1, Mat &img2);

void runImagePair(){
	/*Mat img1 = imread("E:\\Program\\pc_lint_test\\pc_lint_test\\data\\录像8 01.jpg");
	Mat img2 = imread("E:\\Program\\pc_lint_test\\pc_lint_test\\data\\录像8 02.jpg");*/

	Mat img1 = imread("H:\\406-王俊超文件夹\\学术研究\\毕业论文资料\\王俊超-研究生毕业设计\\LibaryData\\Blur.bikes\\img1.ppm");
	Mat img2 = imread("H:\\406-王俊超文件夹\\学术研究\\毕业论文资料\\王俊超-研究生毕业设计\\LibaryData\\Blur.bikes\\img2.ppm");

	imresize(img1, 480);
	imresize(img2, 480);

	GmsMatch(img1, img2);

}


int main()
{
#ifdef USE_GPU
	int flag = cuda::getCudaEnabledDeviceCount();
	if (flag != 0){ cuda::setDevice(0); }
#endif // USE_GPU

	runImagePair();

	return 0;
}


void GmsMatch(Mat &img1, Mat &img2){
	vector<KeyPoint> kp1, kp2;
	Mat d1, d2;
	vector<DMatch> matches_all, matches_all2, matches_gms;

	Ptr<ORB> orb = ORB::create(10000);
	orb->setFastThreshold(0);
	orb->detectAndCompute(img1, Mat(), kp1, d1);
	orb->detectAndCompute(img2, Mat(), kp2, d2);

#ifdef USE_GPU
	GpuMat gd1(d1), gd2(d2);
	Ptr<cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(NORM_HAMMING);
	matcher->match(gd1, gd2, matches_all);
#else
	BFMatcher matcher(NORM_HAMMING);
	matcher.match(d1, d2, matches_all);
	matcher.match(d2, d1, matches_all2);
#endif
	ofstream myfile;
	myfile.open("points.txt", ios::trunc);

	// print points
	for (int i = 0; i < matches_all.size(); ++i) {
		myfile << kp1[matches_all[i].queryIdx].pt << " " << kp2[matches_all[i].trainIdx].pt << " " << matches_all[i].distance << endl;
	}
	myfile.close();
	/*
	Mat show2 = DrawInlier(img1, img2, kp1, kp2, matches_all, 1);
	imshow("show2", show2);
	*/
	
	// GMS filter
	int num_inliers = 0, num_inliers2 = 0;
	std::vector<bool> vbInliers, vbInliers2;
	gms_matcher gms(kp1,img1.size(), kp2,img2.size(), matches_all);
	gms_matcher gms2(kp2, img2.size(), kp1, img1.size(), matches_all2);

	num_inliers = gms.GetInlierMask(vbInliers, false, false);
	num_inliers2 = gms2.GetInlierMask(vbInliers2, false, false);
	cout << "Get total " << num_inliers << " matches." << endl;
	cout << "Get total " << num_inliers2 << " matches." << endl;

	// draw matches
	for (size_t i = 0; i < vbInliers.size() && i < vbInliers2.size(); ++i)
	{
		if (vbInliers[i] == true && vbInliers2[i] == true)
		{
			matches_gms.push_back(matches_all[i]); 
		}
	}

	Mat show = DrawInlier(img1, img2, kp1, kp2, matches_gms, 0);
	imshow("show", show);
	waitKey();
}