
#include<stdio.h>
#include<iostream>
#include <time.h>
#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <cv.h>
#include <cvaux.hpp>
#include <cxcore.h>

#include "opencvx/cvxmat.h"
#include "opencvx/cvxrectangle.h"
#include "opencvx/cvrect32f.h"
#include "opencvx/cvcropimageroi.h"
#include "opencvx/cvdrawrectangle.h"
#include "opencvx/cvparticle.h"
// 
#include "state.h"
#include "observepca.h"
#include "image_add.h"
#include "particle.h"


using namespace cv::gpu;
using namespace cv;
using namespace std;


// void help()
// {
// 	cout << "\nThis program demonstrates using SURF_GPU features detector, descriptor extractor and BruteForceMatcher_GPU" << endl;
// }
void display_matrix(CvMat *H, int row ,int col)
{

	for(int i=0;i<row;i++)
	{
		for(int j=0;j<col;j++)
		{
			printf("%.4f ",cvGet2D(H,i,j).val[0]);
		}
		printf("\n");
	}

}

int main(int argc, char* argv[])
{
// 	if (argc != 3)
// 	{
// 		help();
// 		//return -1;
// 	}
	char path1[1024];
	char path2[1024];
	strcpy(path1,"Lena01.jpg");
	strcpy(path2,"Lena02.jpg");

	IplImage *image1,*image2;
	image1=cvLoadImage(path1, 1);
	image2=cvLoadImage(path2, 1);
	
	int w=image2->width;
	int h=image2->height;
	double angle,lamdax,lamday,dx,dy;

	double match_confidence =0.55;

	GpuMat img1(imread(path1, CV_LOAD_IMAGE_GRAYSCALE));
	GpuMat img2(imread(path2, CV_LOAD_IMAGE_GRAYSCALE));
	GpuMat img11(imread(path1, 1));
	GpuMat img22(imread(path2, 1));

	if (img1.empty() || img2.empty())
	{
		cout << "Can't read one of the images" << endl;
		//return -1;
	}
	double t = (double)getTickCount();
	SURF_GPU surf;

	// detecting keypoints & computing descriptors
	GpuMat keypoints1GPU, keypoints2GPU;
	GpuMat descriptors1GPU, descriptors2GPU;
	surf(img1, GpuMat(), keypoints1GPU, descriptors1GPU);
	surf(img2, GpuMat(), keypoints2GPU, descriptors2GPU);

	cout << "FOUND " << keypoints1GPU.cols << " keypoints on first image" << endl;
	cout << "FOUND " << keypoints2GPU.cols << " keypoints on second image" << endl;

	// matching descriptors
	BruteForceMatcher_GPU< L2<float> > matcher;
	vector<DMatch> good_matches,normal_matches;
	GpuMat trainIdx, distance,allDist;
	//matcher.matchSingle(descriptors1GPU, descriptors2GPU, trainIdx, distance);
	matcher.knnMatch(descriptors1GPU, descriptors2GPU, trainIdx, distance, allDist, 2);

	// downloading results
	vector<KeyPoint> keypoints1, keypoints2,keypoints11,keypoints22;
	vector<float> descriptors1, descriptors2;
	vector< vector<DMatch> > matches;
	surf.downloadKeypoints(keypoints1GPU, keypoints1);
	surf.downloadKeypoints(keypoints2GPU, keypoints2);
	surf.downloadDescriptors(descriptors1GPU, descriptors1);
	surf.downloadDescriptors(descriptors2GPU, descriptors2);
	BruteForceMatcher_GPU< L2<float> >::knnMatchDownload(trainIdx, distance, matches);
	cout<<"total "<<matches.size()<<" matches"<<endl;


	//count good matches 
	good_matches.clear();
	good_matches.reserve(matches.size());
	for (size_t i = 0; i < matches.size(); ++i)
	{
		if (matches[i].size() < 2)
			continue;

		const DMatch &m1 = matches[i][0];
		const DMatch &m2 = matches[i][1];

		if (m1.distance < m2.distance * match_confidence)
			good_matches.push_back(m1);
	}
	int n=good_matches.size();
	cout<<"total "<<n<<" good matches"<<endl;


	// drawing the results
	Mat img_matches;
	drawMatches(img11, keypoints1, img22, keypoints2, good_matches, img_matches);
	namedWindow("matches", 0);
	imshow("matches", img_matches);


	// find the homography matrix
	std::vector<Point2f> obj;  //  vector of points in sensed image
	std::vector<Point2f> scene;  // vector of points in reference image

	Point2f Leftimage[4];
	Point2f Rightimage[4];

	CvMat *srcpoint1,*dstpoint2;
	srcpoint1 = cvCreateMat(6,2,CV_32FC1);
	dstpoint2 = cvCreateMat(6,2,CV_32FC1);

 	srand(time(NULL));
	
	//int  j=rand()%(good_matches.size()-6);cout<<j<<endl;

	for (int i=0;i<6;i++)
	{
		int  j=rand()%(good_matches.size()-6);cout<<j<<endl;
		cvmSet(srcpoint1,i,0,keypoints1[good_matches[j].queryIdx].pt.x);
		cvmSet(srcpoint1,i,1,keypoints1[good_matches[j].queryIdx].pt.y);
		cvmSet(dstpoint2,i,0,keypoints2[good_matches[j].trainIdx].pt.x);
		cvmSet(dstpoint2,i,1,keypoints2[good_matches[j].trainIdx].pt.y);
	}
// 	for( ; j<j+4;  )  
// 	{  
// 		//-- Get the keypoints from the good matches  
// 		obj.push_back( keypoints1[ good_matches[j].queryIdx ].pt );  
// 		scene.push_back( keypoints2[ good_matches[j].trainIdx ].pt );   
// 	}  
 	
// 	for (int i=0;i<4;i++)
// 	{
// 		Leftimage[i]=keypoints1[ good_matches[j+i].queryIdx ].pt;
// 		Rightimage[i]=keypoints2[ good_matches[j+i].queryIdx ].pt;
// 	}


	for( int i = 0; i <4; i++ )  
	{  
		int k=rand()%(good_matches.size()-6); 
		//-- Get the keypoints from the good matches  
		obj.push_back( keypoints1[ good_matches[k].queryIdx ].pt );  
		scene.push_back( keypoints2[ good_matches[k].trainIdx ].pt );  
		cout<<"Use keypoint no: "<<k<<endl;
	}  

/////// use the normal matches to find homograhy matrix
// 	normal_matches.clear();
// 	normal_matches.reserve(matches.size());
	for( int i = 0; i < matches.size(); i++ )  
	{  
		const DMatch &m=matches[i][0]; 
        normal_matches.push_back(m);   
	}  
     cout<<"no of normal_matches "<<normal_matches.size()<<endl;


// 	 for( int i=k; i <k+6; i++ )  
// 	 {  
//   int k=rand()%(normal_matches.size()-6); cout<<k<<endl;
// 		 //-- Get the keypoints from the normal matches  
// 		 obj.push_back( keypoints1[ normal_matches[i].queryIdx ].pt );  
// 		 scene.push_back( keypoints2[ normal_matches[i].trainIdx].pt );  
// 		 cout<<i<<endl;
// 	 }  

// 	 for( int i = 0; i < normal_matches.size(); i++ )  
// 	 {  
// 	 	//-- Get the keypoints from the normal matches  
// 	 	obj.push_back( keypoints1[ normal_matches[i].queryIdx ].pt );  
// 	 	scene.push_back( keypoints2[ normal_matches[i].trainIdx ].pt );   
// 	 }  
// 	 for (int i=0;i<6;i++)
// 	 {
// 		 int  j=rand()%(normal_matches.size()-6);cout<<j<<endl;
// 		 cvmSet(srcpoint1,i,0,keypoints1[normal_matches[j].queryIdx].pt.x);
// 		 cvmSet(srcpoint1,i,1,keypoints1[normal_matches[j].queryIdx].pt.y);
// 		 cvmSet(dstpoint2,i,0,keypoints2[normal_matches[j].trainIdx].pt.x);
// 		 cvmSet(dstpoint2,i,1,keypoints2[normal_matches[j].trainIdx].pt.y);
// 	 }
	 
	   
	//Mat H= findHomography(obj,scene,0);
	//Mat H = findHomography( obj, scene, CV_RANSAC );  
	Mat H = findHomography( obj,scene,CV_LMEDS);
	//Mat H=getAffineTransform(obj,scene);
	// Mat H=getPerspectiveTransform(obj,scene);
	 //Mat H=getAffineTransform(Leftimage,Rightimage);
	CvMat *H2;
	H2 = cvCreateMat(3,3,CV_64FC1);
	cvFindHomography(srcpoint1,dstpoint2,H2,0);
	/*double fx=getHvalue(H2);*/
	printf("transform matrix\n");
// 	for(int i=0;i<3;i++)
// 	{
// 		for(int j=0;j<3;j++)
// 		{
// 			printf("%.4f ",H.at<double>(i,j));
// 		}
// 		printf("\n");
// 	}

	angle=asin(H.at<double>(1,0)); cout<<"The angle is "<<(angle*360)/(2*M_PI)<<endl;
	dx=H.at<double>(0,2);cout<<"dx is "<<dx<<endl;
	dy=H.at<double>(1,2);cout<<"dy is "<<dy<<endl;
	lamdax=H.at<double>(0,0)/cos(angle);cout<<"lamda of x is "<<lamdax<<endl;
	lamday=H.at<double>(1,1)/cos(angle);cout<<"lamda of y is "<<lamday<<endl;
	
  
	// use old data structure
 	CvMat *hh=cvCreateMat(2,3,CV_32FC1); 
	//CvMat *hh=cvCreateMat(2,3,CV_32FC1);
// 	for (int i=0;i<3;i++)
// 	{
// 		cvmSet(hh,0,i,H.at<double>(0,i));
// 	}
// 	for (int i=0;i<3;i++)
// 	{
// 		cvmSet(hh,1,i,H.at<double>(1,i));
// 	}
// 	for (int i=0;i<3;i++)
// 	{
// 		cvmSet(hh,2,i,H.at<double>(1,i));
// 	}
	for (int i=0;i<2;i++)
	{
		for (int j=0;j<3;j++)
		{
			cvmSet(hh,i,j,H.at<double>(i,j));
		}
	}
	//warpPerspective(img11,img22,H,Size(w,h),1);
	cvWarpPerspective(image1,image2,H2,CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS,cvScalarAll( 0 ) );

  //warpAffine(img11,img22,H,Size(w,h),1);
 //	cvWarpAffine(image1,image2,hh,CV_INTER_LINEAR + CV_WARP_FILL_OUTLIERS,cvScalarAll( 0 ));
//    display_matrix(hh,3,3);
 	Mat result;
 	result=img22;

// 	IplImage* result_img;
// 	result_img=add_images(hh,image1,image2);
    namedWindow("result", 1);
//	imshow("result",result);
 	cvShowImage("result",image2);
/*	cvShowImage("result",result_img);*/

	int i;
	double *est=NULL;
	IplImage* IP=NULL,*IC=NULL,*ICT=NULL,*I1=NULL;
	int t1,t2,par,Np;
	cout.precision(5);

    IP = cvLoadImage(path1, CV_LOAD_IMAGE_ANYDEPTH|CV_LOAD_IMAGE_GRAYSCALE);
	if(!IP){
    	printf(" ERROR can not load reference image \n");
    	return -1;
  	}
	I1 = cvLoadImage(path2, CV_LOAD_IMAGE_ANYDEPTH|CV_LOAD_IMAGE_GRAYSCALE);
	if(!I1){
    	printf(" ERROR can not load register image \n");
    	return -1;
  	}


	IC = cvCloneImage(IP);
	cvResize(I1,IC);
	ICT = cvCloneImage(IC);
    est = new double [7];
    Np=128; // Number of the particles
    par=5;  //Number of Parameters

   i=FP(IP,IC,ICT,est,par,Np);

    cout<<endl<<" Total = "<<IP->width<<"x"<<IP->height<<" pixeles";
//    cout<<endl<<" Iterations = "<<i<<endl;
    
	if (par==5)
	{
		cout<<endl<<"            Angel     Dx      Dy      lamdaX        lamdaY        ";
		cout<<endl<<" Estimated = ";
		for(int j=0;j<5;j++)
			cout<<est[j]<<"   ";
	} 
	else
	{
		cout<<endl<<"            Angel     Dx      Dy      lamdaX        lamdaY        gamaX        gamaY";
		cout<<endl<<" Estimated = ";
		for(int j=0;j<6;j++)
			cout<<est[j]<<"   ";
		cout<<est[6]<<endl;
	}

//     cvNamedWindow("Reference",1);
//     cvNamedWindow("Register",1);
    cvNamedWindow("Transf",1);
//     cvShowImage("Reference",IP);
//     cvShowImage("Register",IC);
    cvShowImage("Transf",ICT);

	obj.empty();
	scene.empty();
	waitKey(0);
	return 0;
}
