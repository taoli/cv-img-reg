
#include <highgui.h>
#include "image_add.h"
/*#include "xform.h"*/

static void display_matrix(CvMat*, int,int);
static IplImage* new_image_construct(CvMat*,IplImage*,IplImage*,double*,double*,
									 CvPoint2D32f* ,CvPoint2D32f*,CvPoint2D32f*,CvPoint2D32f*);
static IplImage** backward_transform_to_same_koord(CvMat*,CvMat*, IplImage*,IplImage*,IplImage*,double,double,
												   CvPoint2D32f ,CvPoint2D32f,CvPoint2D32f,CvPoint2D32f,
												   CvSeq*,CvPoint2D32f*,CvPoint2D32f*);
static IplImage* images_blending(IplImage**,CvSeq* ,CvPoint2D32f*,CvPoint2D32f*);
static double cacul_blending_alpha(CvPoint2D32f*,CvPoint2D32f*,int,int);
double calcu_erro(IplImage* input_img,CvSeq* overlap);


IplImage* add_images(CvMat* H, IplImage* img1,IplImage* img2)
{

	CvSeq* overlap;
	CvMemStorage* storage;
	storage = cvCreateMemStorage( 0 );
	overlap = cvCreateSeq( 0, sizeof(CvSeq), sizeof(struct CvPoint), storage );
	CvPoint2D32f top_left_point,top_right_point,bottom_left_point,bottom_right_point;
	CvPoint2D32f dst_point1[4];
	CvPoint2D32f dst_point2[4];
	double Min_x,Min_y;
	double* output_x = NULL;
	double* output_y = NULL;
	double erro;
	IplImage* new_img;
	IplImage** img_insamekoord;
	IplImage* result_img;
	CvMat* inverse_H = cvCreateMat(3,3,CV_64FC1);;
	cvInvert(H,inverse_H,CV_LU);
	new_img = new_image_construct(H,img1,img2,&Min_x,&Min_y,&top_left_point,&top_right_point,
		&bottom_left_point,&bottom_right_point);
	img_insamekoord = backward_transform_to_same_koord(inverse_H,H,img1,img2,new_img,Min_x,Min_y,
		top_left_point,top_right_point,bottom_left_point,
		bottom_right_point,overlap,dst_point1,dst_point2);

  /* cvNamedWindow( "new", 1 );
	cvShowImage( "new", img_insamekoord[0] );
	cvWaitKey( 0 );

	cvNamedWindow( "new2", 1 );
	cvShowImage( "new2", img_insamekoord[1]  );
	cvWaitKey( 0 );*/
	IplImage* result2 = cvCreateImage( cvSize( img_insamekoord[0]->width,img_insamekoord[0]->height ), IPL_DEPTH_8U, 3 );
	IplImage *destination = cvCreateImage( cvSize( img_insamekoord[0]->width, img_insamekoord[0]->height ), IPL_DEPTH_8U, 1 );
	IplImage *destination2 = cvCreateImage( cvSize( img_insamekoord[0]->width, img_insamekoord[0]->height ), IPL_DEPTH_8U, 1 );
	cvCvtColor( img_insamekoord[0], destination, CV_RGB2GRAY );
	cvCvtColor( img_insamekoord[1], destination2, CV_RGB2GRAY );
	//cvSub(destination,destination2,result2);	
	cvAdd( img_insamekoord[0],img_insamekoord[1],result2);
	//cvAddWeighted(img_insamekoord[0],0.5,img_insamekoord[1],0.5,0,result2);	
	//cvNamedWindow( "trans", 1 );
	//cvShowImage( "trans", result2 );
	//cvSaveImage( "D:\\result.bmp", result2 );


	erro=calcu_erro(result2,overlap);
	printf("erro is %f\n",erro);
	result_img = images_blending(img_insamekoord,overlap,dst_point1,dst_point2);

// 	for(int i = 0; i < 2; i++ )
// 	{
// 		cvReleaseImage( &(img_insamekoord)[i] );
// 	}
	////free( *img_insamekoord );
// 	delete(*img_insamekoord );
// 	*img_insamekoord = NULL;
// 	cvReleaseImage( &new_img );
// 	cvReleaseMemStorage( &storage );
	return result_img;

}



static void display_matrix(CvMat* H, int row ,int col)
{

	for(int i=0;i<row;i++)
	{
		for(int j=0;j<col;j++)
		{
			printf("%f ",cvGet2D(H,i,j).val[0]);
		}
		printf("\n");
	}


}

static IplImage* new_image_construct(CvMat* H,IplImage* img1,IplImage* img2,
									 double* Min_x,double* Min_y,
									 CvPoint2D32f* top_left_point,CvPoint2D32f* top_right_point,
									 CvPoint2D32f* bottom_left_point,CvPoint2D32f* bottom_right_point)
{
	IplImage* new_img; 
	CvMat* ResultMatrix=cvCreateMat(3,4,CV_64FC1);
	CvMat* ResultMatrix2=cvCreateMat(3,4,CV_64FC1);
	CvMat* Matrix=cvCreateMat(3,4,CV_64FC1);
	CvMat* inverse_H = cvCreateMat(3,3,CV_64FC1);;
	cvInvert(H,inverse_H,CV_SVD);
	double x[4],y[4],z[4],min_x,max_x,min_y,max_y;
	double width,height,new_width,new_height;
	double Array1[]={
		0,  img1->width-1,  0,               img1->width-1,
		0,  0,              img1->height-1,  img1->height-1,
		1,  1,              1,               1
	};

	cvSetData(Matrix,Array1,Matrix->step);
	cvMatMul(H,Matrix,ResultMatrix);
	/*printf(" --------Matrix H is ------\n");
	display_matrix(H,3,3);
	printf(" --------Matrix inverse_H is  ------\n");
	display_matrix(inverse_H,3,3);
	printf(" --------++++ ------\n");
	display_matrix(ResultMatrix,3,4);*/
	cvmMul(inverse_H,ResultMatrix,ResultMatrix2);
	//printf(" --------***** ------\n");
	//display_matrix(ResultMatrix2,3,4);

	for (int i = 0 ; i<4; i++)
		x[i] = (double)(cvGet2D(ResultMatrix,0,i).val[0]/cvGet2D(ResultMatrix,2,i).val[0]);
	for (int i = 0 ; i<4; i++)
		y[i] = (double)(cvGet2D(ResultMatrix,1,i).val[0]/cvGet2D(ResultMatrix,2,i).val[0]);
	for (int i = 0 ; i<4; i++)
		z[i] = (double)cvGet2D(ResultMatrix,2,i).val[0];

	(*top_left_point).x = x[0];
	(*top_left_point).y = y[0];

	(*top_right_point).x = x[1];
	(*top_right_point).y = y[1];

	(*bottom_left_point).x = x[2];
	(*bottom_left_point).y = y[2];

	(*bottom_right_point).x = x[3];
	(*bottom_right_point).y = y[3];

	max_x = x[0];
	min_x = x[0];
	for (int i = 0 ; i<3; i++)
	{
		if(x[i+1] > max_x)
			max_x =x[i+1];
		if(x[i+1]< min_x)
			min_x = x[i+1];
	}
	max_y = y[0];
	min_y = y[0];
	for (int i = 0 ; i<3; i++)
	{

		if(y[i+1] > max_y)
			max_y = y[i+1];
		if(y[i+1]<min_y)
			min_y = y[i+1];
	}
	*Min_x = min_x;
	*Min_y = min_y;
	width = max_x - min_x;
	height = max_y - min_y;
	if(min_x >=0 && min_y>=0)
	{
		if(img2->width >=min_x+width && img2->height >= min_y+height)
		{
			new_width = img2->width;
			new_height = img2->height;
		}
		if(img2->width < min_x+width && img2->height >= min_y+height)
		{
			new_width = min_x+width;
			new_height = img2->height;
		}
		if(img2->width >= min_x+width && img2->height < min_y+height)
		{
			new_width = img2->width;
			new_height = min_y+height;
		}
		if(img2->width < min_x+width && img2->height < min_y+height)
		{
			new_width = min_x+width;
			new_height = min_y+height;
		}

	}
	if(min_x < 0 && min_y>=0)
	{
		if(img2->width-min_x >=width && img2->height >= min_y+height)
		{
			new_width = img2->width-min_x;
			new_height = img2->height;
		}
		if(img2->width-min_x < width && img2->height >= min_y+height)
		{
			new_width = width;
			new_height = img2->height;
		}
		if(img2->width-min_x >= width && img2->height < min_y+height)
		{
			new_width = img2->width-min_x;
			new_height = min_y+height;
		}
		if(img2->width-min_x <width && img2->height < min_y+height)
		{
			new_width = width;
			new_height = min_y+height;
		}
	}
	if(min_x >= 0 && min_y < 0)
	{
		if(img2->width>=min_x+width && img2->height-min_y >= height)
		{
			new_width = img2->width;
			new_height = img2->height-min_y;
		}
		if(img2->width <min_x+width && img2->height-min_y >= height)
		{
			new_width = width+min_x;
			new_height = img2->height-min_y;
		}
		if(img2->width >= min_x+width && img2->height-min_y < height)
		{
			new_width = img2->width;
			new_height = height;
		}
		if(img2->width<min_x+width && img2->height-min_y < height)
		{
			new_width = width+min_x;
			new_height = height;
		}
	}
	if(min_x < 0 && min_y < 0)
	{
		if(img2->width-min_x>=width && img2->height-min_y>= height)
		{
			new_width = img2->width-min_x;
			new_height = img2->height-min_y;
		}
		if(img2->width -min_x<width && img2->height-min_y >= height)
		{
			new_width = width;
			new_height = img2->height-min_y;
		}
		if(img2->width-min_x >= width && img2->height-min_y < height)
		{
			new_width = img2->width-min_x;
			new_height = height;
		}
		if(img2->width-min_x<width && img2->height-min_y < height)
		{
			new_width = width;
			new_height = height;
		}
	}
	//printf("new_width is %f , new_height is %f\n",new_width,new_height);
	new_img = cvCreateImage( cvSize( new_width,new_height ), IPL_DEPTH_8U, 3 );
	return new_img;

}
static IplImage** backward_transform_to_same_koord(CvMat* inverse_H,CvMat* H, IplImage* img1,
												   IplImage* img2,IplImage* new_img,double min_x,double min_y,
												   CvPoint2D32f top_left_point,CvPoint2D32f top_right_point,
												   CvPoint2D32f bottom_left_point,CvPoint2D32f bottom_right_point,
												   CvSeq* overlap,CvPoint2D32f dst_point[4],CvPoint2D32f dst_point2[4])
{
	IplImage** output_img;
	CvMat* M=cvCreateMat(3,1,CV_64FC1);
	CvMat* RM=cvCreateMat(3,1,CV_64FC1);
	CvPoint2D32f src_point[4];
	CvPoint2D32f src_point2[4];
	int X,Y;
	output_img =(IplImage**) calloc( 2, sizeof( IplImage*) );
	for( int i = 0; i < 2; i++ )
		output_img[i] = (IplImage*)calloc( 1 , sizeof( IplImage* ) );
	output_img[0] = cvCreateImage(cvSize(new_img->width,new_img->height),IPL_DEPTH_8U, 3 );
	output_img[1] = cvCreateImage(cvSize(new_img->width,new_img->height),IPL_DEPTH_8U, 3 );
	if(min_x < 0 && min_y < 0)
	{
		dst_point[0].x = top_left_point.x - min_x;
		dst_point[0].y = top_left_point.y - min_y;

		dst_point[1].x = top_right_point.x - min_x;
		dst_point[1].y = top_right_point.y - min_y;

		dst_point[2].x = bottom_left_point.x - min_x;
		dst_point[2].y = bottom_left_point.y - min_y;

		dst_point[3].x = bottom_right_point.x - min_x;
		dst_point[3].y = bottom_right_point.y - min_y;
	}
	if(min_x < 0 && min_y >= 0)
	{		
		dst_point[0].x = top_left_point.x - min_x;
		dst_point[0].y = top_left_point.y - 0;

		dst_point[1].x = top_right_point.x - min_x;
		dst_point[1].y = top_right_point.y -0;

		dst_point[2].x = bottom_left_point.x - min_x;
		dst_point[2].y = bottom_left_point.y  - 0;

		dst_point[3].x = bottom_right_point.x - min_x;
		dst_point[3].y = bottom_right_point.y  - 0;

	}
	if(min_x >= 0 && min_y < 0)
	{
		dst_point[0].x = top_left_point.x - 0;
		dst_point[0].y = top_left_point.y - min_y;

		dst_point[1].x = top_right_point.x - 0;
		dst_point[1].y = top_right_point.y - min_y;

		dst_point[2].x = bottom_left_point.x - 0;
		dst_point[2].y = bottom_left_point.y - min_y;

		dst_point[3].x = bottom_right_point.x - 0;
		dst_point[3].y = bottom_right_point.y - min_y;
	}
	if(min_x >= 0 && min_y >= 0)
	{
		dst_point[0].x = top_left_point.x - 0;
		dst_point[0].y = top_left_point.y - 0;

		dst_point[1].x = top_right_point.x - 0;
		dst_point[1].y = top_right_point.y - 0;

		dst_point[2].x = bottom_left_point.x - 0;
		dst_point[2].y = bottom_left_point.y - 0;

		dst_point[3].x = bottom_right_point.x - 0;
		dst_point[3].y = bottom_right_point.y - 0;
	}
	src_point[0].x=0;
	src_point[0].y=0;

	src_point[1].x=img1->width-1;
	src_point[1].y=0;

	src_point[2].x=0;
	src_point[2].y=img1->height-1;

	src_point[3].x=img1->width-1;
	src_point[3].y=img1->height-1;

	float newm[9];
	CvMat newM = cvMat( 3, 3, CV_32F, newm );
	cvWarpPerspectiveQMatrix(src_point,dst_point,&newM);
	cvWarpPerspective(img1,output_img[0],&newM,CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS, cvScalarAll(0) );
	printf(" --------Matrix new_H is  ------\n");
	display_matrix(&newM,3,3);

	if(min_x < 0 && min_y < 0)
	{
		dst_point2[0].x = 0 - min_x;
		dst_point2[0].y = 0 - min_y;

		dst_point2[1].x = img2->width-1 - min_x;
		dst_point2[1].y = 0 - min_y;

		dst_point2[2].x = 0 - min_x;
		dst_point2[2].y = img2->height-1 - min_y;

		dst_point2[3].x = img2->width-1 - min_x;
		dst_point2[3].y = img2->height-1 - min_y;

	}
	if(min_x < 0 && min_y >= 0)
	{		
		dst_point2[0].x = 0 - min_x;
		dst_point2[0].y = 0;

		dst_point2[1].x = img2->width-1 - min_x;
		dst_point2[1].y = 0;

		dst_point2[2].x = 0 - min_x;
		dst_point2[2].y = img2->height-1;

		dst_point2[3].x = img2->width-1 - min_x;
		dst_point2[3].y = img2->height-1;

	}
	if(min_x >= 0 && min_y < 0)
	{
		dst_point2[0].x = 0;
		dst_point2[0].y = 0 - min_y;

		dst_point2[1].x = img2->width-1 ;
		dst_point2[1].y = 0 - min_y;

		dst_point2[2].x = 0;
		dst_point2[2].y = img2->height-1 - min_y;

		dst_point2[3].x = img2->width-1;
		dst_point2[3].y = img2->height-1 - min_y;

	}
	if(min_x >= 0 && min_y >= 0)
	{
		dst_point2[0].x = 0;
		dst_point2[0].y = 0;

		dst_point2[1].x = img2->width-1;
		dst_point2[1].y = 0;

		dst_point2[2].x = 0;
		dst_point2[2].y = img2->height-1;

		dst_point2[3].x = img2->width-1;
		dst_point2[3].y = img2->height-1;
	}

	src_point2[0].x=0;
	src_point2[0].y=0;

	src_point2[1].x=img2->width-1;
	src_point2[1].y=0;

	src_point2[2].x=0;
	src_point2[2].y=img2->height-1;

	src_point2[3].x=img2->width-1;
	src_point2[3].y=img2->height-1;


	float newm2[9];
	CvMat newM2 = cvMat( 3, 3, CV_32F, newm2 );
	cvWarpPerspectiveQMatrix(src_point2,dst_point2,&newM2);
	cvWarpPerspective(img2,output_img[1],&newM2,CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS, cvScalarAll(0) );
	printf(" --------Matrix new_H2 is  ------\n");
	display_matrix(&newM2,3,3);

	for(int y=0; y<new_img->height-1;y++)
	{
		for (int x=0;x<new_img->width-1;x++)
		{
			double Array[3];
			if(min_x < 0 && min_y < 0)
			{
				Array[0] = x+min_x;
				Array[1] = y+min_y;
				Array[2] = 1;

			}
			if(min_x < 0 && min_y >= 0)
			{
				Array[0] = x+min_x;
				Array[1] = y-0;
				Array[2] = 1;

			}
			if(min_x >= 0 && min_y < 0)
			{
				Array[0] = x-0;
				Array[1] = y+min_y;
				Array[2] = 1;

			}
			if(min_x >= 0 && min_y >= 0)
			{
				Array[0] = x-0;
				Array[1] = y-0;
				Array[2] = 1;

			}

			cvSetData(M,Array,M->step);
			cvmMul(inverse_H,M,RM);
			int xx,yy;
			if(min_x < 0 && min_y < 0)
			{
				xx = x+min_x;
				yy = y+min_y;

			}
			if(min_x < 0 && min_y >= 0)
			{
				xx = x+min_x;
				yy = y-0;

			}
			if(min_x >= 0 && min_y < 0)
			{
				xx = x-0;
				yy = y+min_y;

			}
			if(min_x >= 0 && min_y >= 0)
			{
				xx = x-0;
				yy = y-0;

			}

			X = cvRound(cvGet2D(RM,0,0).val[0]/cvGet2D(RM,2,0).val[0]);
			Y = cvRound(cvGet2D(RM,1,0).val[0]/cvGet2D(RM,2,0).val[0]);

			if(X>=0 && X<=img1->width && Y>=-1 && Y<=img1->height)
			{
				if(xx>=0 && xx<=img2->width && yy>=-1 && yy<=img2->height)
				{
					CvPoint point;
					point.x = x;
					point.y = y;
					cvSeqPush( overlap, &point );

				}
			}

		}
	}
	return output_img;
}

static IplImage* images_blending(IplImage** input_img,CvSeq* overlap,
								 CvPoint2D32f dst_point1[4],CvPoint2D32f dst_point2[4])
{

	IplImage* new_img;
	double alpha;

	new_img = cvCreateImage(cvSize(input_img[0]->width,input_img[0]->height),IPL_DEPTH_8U, 3 );
	cvAdd(input_img[0],input_img[1],new_img,NULL);
	uchar* img_tr = (uchar*)new_img->imageData;
	uchar* img_tr1 = (uchar*)input_img[0]->imageData;
	uchar* img_tr2 = (uchar*)input_img[1]->imageData;
	for ( int j = 0; j < overlap->total; j++ )
	{
		CvPoint* point = CV_GET_SEQ_ELEM( CvPoint, overlap, j );

		alpha = cacul_blending_alpha(dst_point1,dst_point2,point->x,point->y);
		//alpha =  0.5;
		if(alpha!=0)
		{		
			img_tr[point->x*3+point->y*new_img->widthStep] =alpha*img_tr1[point->x*3+point->y*input_img[0]->widthStep]
			+ (1-alpha)*img_tr2[point->x*3+point->y*input_img[1]->widthStep];


			img_tr[point->x*3+1+point->y*new_img->widthStep] =alpha*img_tr1[point->x*3+1+point->y*input_img[0]->widthStep]
			+ (1-alpha)*img_tr2[point->x*3+1+point->y*input_img[1]->widthStep];


			img_tr[point->x*3+2+point->y*new_img->widthStep] =alpha*img_tr1[point->x*3+2+point->y*input_img[0]->widthStep]
			+ (1-alpha)*img_tr2[point->x*3+2+point->y*input_img[1]->widthStep];
		}
		else
		{
			img_tr[point->x*3+point->y*new_img->widthStep] = 0;//img_tr1[point->x*3+point->y*input_img[0]->widthStep];
			img_tr[point->x*3+point->y*new_img->widthStep] = 0;//img_tr1[point->x*3+1+point->y*input_img[0]->widthStep];
			img_tr[point->x*3+point->y*new_img->widthStep] = 0;//img_tr1[point->x*3+2+point->y*input_img[0]->widthStep];
		}

	}
	/*cvNamedWindow( "new", 1 );
	cvShowImage( "new", input_img[0] );
	cvWaitKey( 0 );

	cvNamedWindow( "new2", 1 );
	cvShowImage( "new2", input_img[1]  );
	cvWaitKey( 0 );
	cvNamedWindow( "new3", 1 );
	cvShowImage( "new3", new_img);
	cvWaitKey( 0 );*/
	//cvSaveImage( "D:\\registration.bmp", new_img );
	return new_img;

}
static double cacul_blending_alpha(CvPoint2D32f* point1,CvPoint2D32f* point2,int x,int y)
{
	double A1[4], B1[4], C1[4],d1[4];
	double A2[4], B2[4], C2[4],d2[4];
	double min_d1,min_d2;
	double alpha;

	A1[0] = point1[1].y - point1[0].y;
	B1[0] = point1[0].x - point1[1].x;
	C1[0] = point1[0].x*(point1[0].y-point1[1].y) - point1[0].y*(point1[0].x-point1[1].x);

	A1[1] = point1[3].y - point1[1].y;
	B1[1] = point1[1].x - point1[3].x;
	C1[1] = point1[1].x*(point1[1].y-point1[3].y) - point1[1].y*(point1[1].x-point1[3].x);

	A1[2] = point1[3].y - point1[2].y;
	B1[2] = point1[2].x - point1[3].x;
	C1[2] = point1[2].x*(point1[2].y-point1[3].y) - point1[2].y*(point1[2].x-point1[3].x);

	A1[3] = point1[2].y - point1[0].y;
	B1[3] = point1[0].x - point1[2].x;
	C1[3] = point1[0].x*(point1[0].y-point1[2].y) - point1[0].y*(point1[0].x-point1[2].x);

	for (int i = 0; i<4;i++)
		d1[i] = fabs(A1[i]*x + B1[i]*y + C1[i])/sqrt(A1[i]*A1[i] + B1[i]*B1[i]);


	A2[0] = point2[1].y - point2[0].y;
	B2[0] = point2[0].x - point2[1].x;
	C2[0] = point2[0].x*(point2[0].y-point2[1].y) - point2[0].y*(point2[0].x-point2[1].x);

	A2[1] = point2[3].y - point2[1].y;
	B2[1] = point2[1].x - point2[3].x;
	C2[1] = point2[1].x*(point2[1].y-point2[3].y) - point2[1].y*(point2[1].x-point2[3].x);

	A2[2] = point2[3].y - point2[2].y;
	B2[2] = point2[2].x - point2[3].x;
	C2[2] = point2[2].x*(point2[2].y-point2[3].y) - point2[2].y*(point2[2].x-point2[3].x);

	A2[3] = point2[2].y - point2[0].y;
	B2[3] = point2[0].x - point2[2].x;
	C2[3] = point2[0].x*(point2[0].y-point2[2].y) - point2[0].y*(point2[0].x-point2[2].x);

	for (int i = 0; i<4;i++)
		d2[i] = fabs(A2[i]*x + B2[i]*y + C2[i])/sqrt(A2[i]*A2[i] + B2[i]*B2[i]);

	min_d1 = d1[0];
	for (int i = 0 ; i<3; i++)
	{
		if(d1[i+1]< min_d1)
			min_d1 = d1[i+1];
	}

	min_d2 = d2[0];
	for (int i = 0 ; i<3; i++)
	{
		if(d2[i+1]< min_d2)
			min_d2 = d2[i+1];
	}

	alpha = min_d1/(min_d1 + min_d2);
	return alpha;
}

double calcu_erro(IplImage* input_img,CvSeq* overlap)
{
	
	 double erro;
	//double alpha;

	//new_img = cvCreateImage(cvSize(input_img[0]->width,input_img[0]->height),IPL_DEPTH_8U, 3 );
	//cvAdd(input_img[0],input_img[1],new_img,NULL);
	//uchar* img_tr = (uchar*)new_img->imageData;
	uchar* img_tr1 = (uchar*)input_img->imageData;
	//uchar* img_tr2 = (uchar*)input_img[1]->imageData;
	erro =0;
	for ( int j = 0; j < overlap->total; j++ )
	{
		CvPoint* point = CV_GET_SEQ_ELEM( CvPoint, overlap, j );

		
			//printf("value is %d\n",img_tr1[point->x+point->y*input_img->widthStep])*(img_tr1[point->x+point->y*input_img->widthStep]);
			erro =erro +((img_tr1[point->x+point->y*input_img->widthStep]))*((img_tr1[point->x+point->y*input_img->widthStep]));
			//printf("erro is %E\n",erro);



	}
	erro = erro/overlap->total;
	
	return erro;

}