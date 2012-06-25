#include<stdio.h>
//#include <math.h>
#include <windows.h> 
#include <cstdlib>
#include<cv.h>      //OpenCV
#include<highgui.h> //OpenCV
#include<cxcore.h>  //OpenCV

#include "particle.h"

Particulas::~Particulas(){
	cvReleaseImage(&IP);
	cvReleaseImage(&IC);
	cvReleaseImage(&ICT);
	cvReleaseImage(&P);
	cvReleaseImage(&allP);
	cvReleaseImage(&P_1);
	cvReleaseImage(&Pu);
	cvReleaseImage(&Pv);
	cvReleaseImage(&Puv);
	cvReleaseImage(&Aux1);
	cvReleaseImage(&Aux2);
	cvReleaseImage(&Aux3);
	cvReleaseImage(&Aux4);
	cvReleaseMat(&mAff);

	delete [] var;	//Array for the Variance of disturbances
	delete [] est;	//Array for the estimated
	delete [] ide;	//Identity array
}

void Particulas::Create(IplImage* I,IplImage* C,IplImage*& allP,int N,double* allW,
	double*AccW,int nH,int Np,int par){

		IP=cvCloneImage(I);	
		IC=cvCloneImage(C);	
		ICT=cvCloneImage(C);
		cvCopyImage(I,IP);
		cvCopyImage(C,IC);
		width=IP->width;	//width
		height=IP->height;	//height
		this->N=N;			//Number of Particles in the Thread
		this->Np=Np;		//Total Number of Particles
		this->par=par;		//Number of Parameters for evaluation - 5 for the TRME

		ite=0;				//Iteraciones
		this->nH=nH;
		indH=nH*N;			//Index to bind the particles of threads
		//Particles
		P=cvCreateImage(cvSize(N,np),IPL_DEPTH_64F,1);
		P_1=cvCreateImage(cvSize(N,np),IPL_DEPTH_64F,1);
		this->allP=cvCloneImage(allP);

		//Transformacion Affine
		mAff=cvCreateMat(2,3,CV_64FC1);
		//Histograms
		Puv=cvCreateImage(cvSize(bins,bins),IPL_DEPTH_32F,1);	//Joint histogram
		Pu=cvCreateImage(cvSize(1,bins),IPL_DEPTH_32F,1);		//Histogram U
		Pv=cvCreateImage(cvSize(1,bins),IPL_DEPTH_32F,1);		//Histogram V
		//Auxiliares
		Aux1=cvCreateImage(cvSize(1,bins),IPL_DEPTH_32F,1);
		Aux2=cvCreateImage(cvSize(bins,bins),IPL_DEPTH_32F,1);

		// Auxiliares for calculating the gray value difference
		Aux3=cvCreateImage(cvSize(16,16),IPL_DEPTH_32F,1);
		Aux4=cvCreateImage(cvSize(16,16),IPL_DEPTH_32F,1);

		this->W=allW;			//weights
		this->AccW=AccW;		//Cumulative Weights
		nW=1.0/(double)this->Np;		//Factor to calculate the estimated

		est=new double [np];	//Estimated
		var=new double [np];	//Variances of the disturbances
		ide=new double [np];	//Identity

		var[0]=5; ///Angulo 4s = 20
		// 	var[1]=(double)(width*width)/64.0;  
		// 	var[2]=(double)(height*height)/64.0;  
		var[1]=(double)(width*width)/64.0;  
		var[2]=(double)(height*height)/64.0;  
		for(i=3;i<np;i++)
			var[i]=15.625e-3;  //
		// 5 parameters for TRME
		if(this->par==5)
			var[5]=var[6]=0.0;

		//identity
		for(i=0;i<3;i++)
			ide[i]=0.0;  
		for(i=3;i<np;i++)
			ide[i]=1.0; 

		fac=exp(log(1e-5)/this->Np);			// reduction factor for Variance disturbance

		c1=1.0/(sqrt(2.0*M_PI)*varLK);	//Standardization Cte of the Likelihood
		c2=-1.0/(2.0*varLK*varLK);		//Standardization Cte exponent of Likelihood

		cx=(double)width/2.0;	//Center X
		cy=(double)height/2.0;	//Center Y
		Histograma(IP,Pu,bins);	//Histogram 
		Hx=cvEntropy(Pu,Aux1);	//Entropy 

		// 	pixelsl(IP,ICT,Aux3,Aux4);
		// 	DX=graydis(Aux3,Aux4);


}
//Generates Particles
void Particulas::Generate(){
	BwImageFloat64 pP(P);	//Particulas
	BwImageFloat64 pP1(P_1);//Particulas t-1

	//Estimator of the Transformation Parameters
	if (true)
	{
		if(ite==1)
		{
			for(i=0;i<np;i++)
			{
				var[i]*=0.25;	//Variation of Particle
			}

		}
		if(ite>=1)	//Reduction of the variances
			for(i=0;i<np;i++)
			{
				var[i]*=fac;		//Reduce the variability of disturbances
			}
			for(i=0;i<N;i++)//Generation of N Particles
				for(j=0;j<np;j++)
					if(ite==0)	//Primera Iteracion
						switch(1){
						case 1: //Extencion Normal
							//Theta_k+1 = Theta_k + V_k
							pP[j][i]=ide[j]+rand_normal(var[j]);
							break;
					}
					else		//Resto de Iteraciones
						switch(1){
						case 1: //Extencion Normal
							//Theta_k+1 = Theta_k + V_k
							pP[j][i]=pP1[j][i]+rand_normal(var[j]);
							break;
					}								
					ite++;	//Increase the iteration
	}
}

//Resampling the Particles
void Particulas::Resampling(){
	BwImageFloat64 pP(allP);	//Particulas
	BwImageFloat64 pP1(P_1);	//Particulas t-1
	for(i=0;i<N;i++){
		u=rand_uniform();	//Number with uniform distribution [0,1]
		for(j=0;j<Np;j++)	//
			if(AccW[j]>=u){	//Cumulative distribution
				for(k=0;k<np;k++)
					pP1[k][i]=pP[k][j];	//Resampling
				break;
			}
	}
}
//Evaluates Particles in the dynamic model
void Particulas::Evalua(){
	BwImageFloat64 pP(P);	//Particles of the thread
	sumW=0.0;	//Sum of Weights
	for(i=0;i<N;i++){
		Affine(pP[0][i],pP[1][i],pP[2][i],pP[3][i],pP[4][i],pP[5][i],pP[6][i],-1);	//Perform affine transformation
	   //c4=getHvalue(mAff);
		//  	    cvJointHist(IP,ICT,Puv,Pu,Pv,bins);								//Joint Histogram
		// 	    IM=cvEntropy(Pu,Aux1)+cvEntropy(Pv,Aux1)-cvEntropy(Puv,Aux2);	//Mutual information
		 	    c3=Hx-IM;	//Entropia de X menos la IM(X,Y)
		 	    W[indH+i]=c1*exp(c2*c3*c3);	//Likelihood


		//      pixelsl(IP,ICT,Aux3,Aux4);
		// 		DX=graydis(Aux3,Aux4);
		//W[indH+i]=1/(1+DX/(varLK*varLK)); ////new Likelihood
		//		W[indH+i]=c1*exp(c2*DX*DX);	//Likelihood
		// 	    cout<<W[indH+i]<<endl; 
		sumW+=W[indH+i];			//Sum of the weights for normalization
	}
}
//Affine transformation 
void Particulas::Affine(double theta,double dx,double dy,double lambdax,double lambday,
	double gammax,double gammay,double fill){

		double theta_rad=theta*2.0*M_PI/360.0;
		a1=lambdax*cos(theta_rad);
		// 	a2=gammax*sin(theta_rad);// for muiltimodel registration
		// 	a3=gammay*sin(theta_rad);
		a2=sin(theta_rad);
		a3=sin(theta_rad);
		a4=lambday*cos(theta_rad);
		a5=(1.0-a1)*cx-a2*cy;
		a5+=dx;
		a6=a3*cx+(1.0-a4)*cy;
		a6+=dy;
		// 	a5=dx;
		// 	a6=dy;

		cvmSet(mAff,0,0,a1);
		cvmSet(mAff,0,1,a3);
		cvmSet(mAff,0,2,a5);
		cvmSet(mAff,1,0,-a2);
		cvmSet(mAff,1,1,a4);
		cvmSet(mAff,1,2,a6);
		cvZero(ICT);
		cvWarpAffine(IC,ICT,mAff,CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS,cvScalarAll(fill));
}
//Pasa las particulas del Hilo a el array que contiene todas
void Particulas::GetP(IplImage*& allP){
	BwImageFloat64 paP(allP);	
	BwImageFloat64 pP(P);		
	for(i=0;i<N;i++)
		for(j=0;j<np;j++)
			paP[j][indH+i]=pP[j][i];
}
//Reset Particles
void Particulas::Reset(IplImage* IP,IplImage* IC){
	cvCopyImage(IP,this->IP);
	cvCopyImage(IC,this->IC);
	//Varianzas
	var[0]=5; ///Angulo 4s = 20
	var[1]=(double)(width*width)/64.0;  
	var[2]=(double)(height*height)/64.0;  
	for(i=3;i<np;i++)
		var[i]=15.625e-3;  
	if(par==5)
		var[5]=var[6]=0.0;
	ite=0; //Iteraciones
}
//Calculate the estimated parameters
double* Particulas::Estimado(){
	BwImageFloat64 pP(allP);	//All Particles
	memset(est,0,sizeof(double)*np);
	for(i=0;i<Np;i++)			//Calculate the estimated parameters
		for(k=0;k<np;k++)
			est[k]+=pP[k][i]*nW;
	return est;
}

//Function to get the degradation of variations
double Particulas::VarDeg(){
	a1=0.0;
	for(i=0;i<7;i++)
		a1+=var[i]*var[i];
	return(sqrt(a1));
}

//-----------------------
//Multi-threads
DWORD WINAPI HiloGenEv(LPVOID Estructura){
	//void *HiloGenEv (void* estruct){
	Particulas* P = (Particulas* )Estructura;
	P->Generate();
	P->Evalua();
	return (DWORD)P;
}
DWORD WINAPI HiloResamp(LPVOID Estructura){
	//void *HiloResamp (void* estruct){
	Particulas* P = (Particulas* )Estructura;
	P->Resampling();
	return (DWORD)P;
}


// Particle Filter Function
int FP(IplImage*& IP,IplImage*& IC,IplImage*& ICT,double* estp,int par,int Np)
{ 
	int i,j,k,m,n,N,np;
	double sW,a1,a2;
	double *allW=NULL,*AccW=NULL,*est=NULL;
	IplImage* allP=NULL;
	IplImage* Puv=NULL,*Pu=NULL,*Pv=NULL,*Aux1=NULL,*Aux2=NULL,*Aux3=NULL,*Aux4=NULL;
	CvMat* mAff=NULL;

	HANDLE* idHilo=NULL;
	Particulas* P=NULL;
	SYSTEM_INFO info;
	GetSystemInfo(&info);
	const int NuC = info.dwNumberOfProcessors; // Number of Processors 

	srand ( time(NULL) );

	np = 7;	//Numero de Parametros
	N=(int)floor((double)Np/(double)NuC);	//Particles in Thread

	idHilo=new HANDLE [NuC]; 			//Thread IDs
	AccW=new double [Np];	//Acumulado de Pesos PDA
	allW=new double [Np];	//Todos los Pesos
	P=new Particulas [NuC];	//Particle structures for threads
	allP=cvCreateImage(cvSize(Np,np),IPL_DEPTH_64F,1);	//All Particles
	mAff=cvCreateMat(2,3,CV_64FC1);	//Affine transformation

	//created particles structures for threads
	for(i=0;i<NuC;i++)
		P[i].Create(IP,IC,allP,N,allW,AccW,i,Np,par);

	BwImageFloat64 pP(allP);

	Puv=cvCreateImage(cvSize(16,16),IPL_DEPTH_32F,1);	//Histogram haw
	Pu=cvCreateImage(cvSize(1,16),IPL_DEPTH_32F,1);		//Histograma de U
	Pv=cvCreateImage(cvSize(1,16),IPL_DEPTH_32F,1);		//Histograma de V
	//Auxiliares
	Aux1=cvCreateImage(cvSize(1,16),IPL_DEPTH_32F,1);
	Aux2=cvCreateImage(cvSize(16,16),IPL_DEPTH_32F,1);

	Aux3=cvCreateImage(cvSize(16,16),IPL_DEPTH_32F,1);
	Aux4=cvCreateImage(cvSize(16,16),IPL_DEPTH_32F,1);


	//Iterations begin
	for(i=0;i<100;i++)
	{
		//Create threads for evaluation
		for(j=0;j<NuC;j++)
			idHilo[j] = CreateThread(NULL,0,HiloGenEv,(void*)&P[j],0,NULL);
		//Wait for threads
		for(j=0;j<NuC;j++)
			WaitForSingleObject(idHilo[j], INFINITE);

		sW=0.0;
		//Collect all the particles to generate the PDA and the resampling
		for(j=0;j<NuC;j++){
			sW+=P[j].GetSumW();
			P[j].GetP(allP);
		}
		sW=1.0/sW;

		//Normalization of Weights
		n=0;
		a2=0.0;
		AccW[0]=0.0;
		for(j=0;j<NuC;j++)
			for(m=1;m<N;m++){
				n++;
				if(n>=Np)
					break;
				a1=allW[n]*sW;
				AccW[n]=AccW[n-1]+a1;	//PDA
				a2+=(a1*a1);
			}
			if((1.0/(double)(Np-4))>a2)//degradation of the particles
				break;
			if(i%10==0)
				printf(" Ite = %d  PartDeg = %.4f \n",i,a2);

			for(j=0;j<NuC;j++){
				P[j].SetP(allP);	// Send all the particles to each Thread for Resampling
				idHilo[j] = CreateThread(NULL,0,HiloResamp,(void*)&P[j],0,NULL);
			}
			//Wait for threads
			for(j=0;j<NuC;j++)
				WaitForSingleObject(idHilo[j], INFINITE);

	}//end iteraciones

	est=P[0].Estimado();	//Calculate the Estimated

	cvAffine(IC,ICT,est[0],est[1],est[2],est[3],est[4],est[5],est[6],mAff,0);//apply affine transformation


	memcpy(estp,est,sizeof(double)*np);

	printf("\n Number of Cores = %d",NuC);

	delete [] P;
	delete [] AccW;
	delete [] allW;
	delete [] idHilo;
	cvReleaseImage(&allP);
	cvReleaseImage(&Puv);
	cvReleaseImage(&Pu);
	cvReleaseImage(&Pv);
	cvReleaseImage(&Aux1);
	cvReleaseImage(&Aux2);
	cvReleaseImage(&Aux3);
	cvReleaseImage(&Aux4);
	cvReleaseMat(&mAff);

	return i;
}

//Gaussian noise
double rand_normal(double var){
	double U1,U2,V1,V2,X1;
	double S=2.0;
	if(var==0)
		return 0;
	else{
		while(S>=1){
			U1=rand_uniform();
			U2=rand_uniform();
			V1=2.0*U1-1.0;
			V2=2.0*U2-1.0;
			S=V1*V1+V2*V2;
		}
		X1=V1*sqrt((-2.0*log(S))/S);
		X1*=sqrt(var);
		//X2=(1.0/sqrt(2*M_PI*var))*exp(-(X1*X1)/(2*var));
		return X1;
	}
}

//------------------------------------------------------------------------------
// Entropy
double cvEntropy(IplImage*& Src,IplImage*& Aux){
	int i,j;
	CvScalar e;
	cvCopyImage(Src,Aux);
	cvLog(Aux,Aux);
	cvMul(Src,Aux,Aux);
	e=cvSum(Aux);
	return(-e.val[0]);
}
//--------------------------------------------------------------------------------------------------------------
// Gets the histogram of an Image;
void Histograma(IplImage*& Src,IplImage*& Hist,int bins){
	int i,j,k,a,step;
	int* levels;
	CvScalar sum;
	BwImage pSrc(Src);
	BwImageFloat pH(Hist);
	cvZero(Hist);

	step=(int)round(256.0/(float)bins);
	levels=new int [bins];
	levels[0]=step;
	for(i=1;i<bins;i++)
		levels[i]=levels[i-1]+step;

	for(i=0;i<Src->width;i+=FP_DELTA)
		for(j=0;j<Src->height;j+=FP_DELTA)
			for(k=0;k<bins;k++)
				if(pSrc[i][j]<levels[k]){
					pH[0][k]++;
					break;
				}

				sum=cvSum(Hist);
				sum.val[0]+=(1e-6)*bins;
				sum.val[0]=1.0/sum.val[0];
				for(i=0;i<bins;i++)
					pH[0][i]=(pH[0][i]+1e-6)*sum.val[0];

				delete [] levels;
}

//------------------------------------------------------------------------------
//Joint histogram of two images
void cvJointHist(IplImage*& Src1,IplImage*& Src2,IplImage*& JtHist,IplImage*& xHist,IplImage*& yHist,int bins){
	int i,j,k,l,step,a1=0,a2=0;
	int* levels;
	CvScalar sum;
	cvZero(JtHist);
	cvZero(xHist);
	cvZero(yHist);
	BwImageFloat pH(JtHist);
	BwImageFloat pX(xHist);
	BwImageFloat pY(yHist);
	BwImage pS1(Src1);
	BwImage pS2(Src2);

	levels = new int [bins];
	step=(int)round(256.0/(float)bins);
	levels[0]=step;
	for(i=1;i<bins;i++)
		levels[i]=levels[i-1]+step;

	for(i=0;i<Src1->width;i+=FP_DELTA)
		for(j=0;j<Src1->height;j+=FP_DELTA){
			for(k=0;k<bins;k++)
				if(pS1[i][j]<levels[k])
					break;
			for(l=0;l<bins;l++)
				if(pS2[i][j]<levels[l])
					break;
			pH[k][l]++;
			a2++;
		}
		sum=cvSum(JtHist);
		sum.val[0]+=(1e-6)*bins;
		sum.val[0]=1.0/sum.val[0];
		//inf=(a2*100.0)/(float)(Src1->width*Src2->height);
		for(i=0;i<bins;i++)
			for(j=0;j<bins;j++){
				pH[i][j]=(pH[i][j]+1e-6)*sum.val[0];
				pX[0][i]=pX[0][i]+pH[i][j];
				pY[0][j]=pY[0][j]+pH[i][j];
			}
			delete [] levels;
}

//--------------------------------------------------------------------------------------------------------------
//   Affine Transformation
void cvAffine(IplImage* Src,IplImage*& Dst,double theta,double dx,double dy,double lambdax,double lambday,double gammax,double gammay,CvMat*& Affine,double fill){
	//CvMat* Affine=cvCreateMat(3,3,CV_64FC1);
	//	CvMat* Affine=cvCreateMat(2,3,CV_64FC1);
	double a1,a2,a3,a4,a5,a6;
	double theta_rad=theta*2.0*M_PI/360.0;
	CvPoint2D32f center;
	center.x=(float)(Src->width/2.0);
	center.y=(float)(Src->height/2.0);


	a1=lambdax*cos(theta_rad);
	a2=gammax*sin(theta_rad);
	a3=gammay*sin(theta_rad);
	a4=lambday*cos(theta_rad);

	a5=(1-a1)*center.x-a2*center.y;
	a5+=dx;
	a6=a3*center.x+(1-a4)*center.y;
	a6+=dy;

	//cvReleaseImage(&Dst);
	//Dst=cvCloneImage(Src);
	cvmSet(Affine,0,0,a1);
	cvmSet(Affine,0,1,a3);
	cvmSet(Affine,0,2,a5);
	cvmSet(Affine,1,0,-a2);
	cvmSet(Affine,1,1,a4);
	cvmSet(Affine,1,2,a6);

	cvSetZero(Dst);
	//cvWarpPerspective(Src,Dst,Affine,CV_INTER_CUBIC+CV_WARP_INVERSE_MAP,cvScalarAll(0));
	cvWarpAffine(Src,Dst,Affine,CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS,cvScalarAll(fill));
	//    cvReleaseMat(&Affine);
}

double TRME(double* S,double* Se,int np){
	// S -> Real
	// Se -> estimated
	// np -> Number of Parameters
	double sum=0.0;
	for(int i=0;i<np;i++)
		sum+=fabs((S[i]-Se[i])/S[i]);
	return(sum/(double)np);
}

double graydis(IplImage* src1,IplImage* src2)
{
	IplImage *dst=NULL;
	dst=cvCreateImage(cvSize(src1->width,src1->height),src1->depth,src1->nChannels);
	cvAbsDiff(src1,src2,dst); // absolute difference between two arrays
	CvScalar s;
	// 	for (int i=0;i<width1;i++)
	// 	{
	// 		for (int j=0;j<height1;j++)
	// 		{
	// 			s=cvGet2D(dst,i,j);
	// 			dx+=s.val[0];
	// 		}
	// 	}
	s=cvSum(dst);
	return s.val[0]/10000;
	cvReleaseImage(&dst);
}

void  pixelsl (IplImage* Src1,IplImage* Src2,IplImage* aux11,IplImage* aux22)
{
	// 	    cvZero(aux11);
	// 	    cvZero(aux22);
	// 		BwImage ps1(Src1);
	// 		BwImage ps2(Src2);
	// 		BwImage aux33(aux11);
	// 		BwImage aux44(aux22);
	// 
	//        // IplImage *src11,*src22;
	// // 		ps1=cvCreateImage(cvSize(Src1->width,Src1->height),Src1->depth,1);
	// // 		ps2=cvCreateImage(cvSize(Src1->width,Src1->height),Src1->depth,1);
	// // 		cvCvtColor(Src1,ps1,CV_BGR2GRAY);
	// // 		cvCvtColor(Src2,ps2,CV_BGR2GRAY);
	// 
	// 		for (int k=0;k<16;k++)
	// 		{
	// 			for (int l=0;l<16;l++)
	// 			{
	// 				int i=rand()%256;
	// 				int j=rand()%256;
	// 				aux33[k][l]=ps1[i][j];
	// 				aux44[k][l]=ps2[i][j];
	// 			}
	// 		}
	//srand ( time(NULL) );

	for (int k=0;k<16;k++)
	{
		for (int l=0;l<16;l++)
		{
			int i=rand()%255;
			int j=rand()%255;
			CvScalar s=cvGet2D(Src1,i,j);
			CvScalar y=cvGet2D(Src2,i,j);
			cvSet2D(aux11,k,l,s);
			cvSet2D(aux22,k,l,y);			
		}
	}
}

double getHvalue(CvMat *H)
{
	double angle,lamdax,lamday,dx,dy;
	angle=asin(cvmGet(H,1,0)); 
	dx=cvmGet(H,0,2);
	dy=cvmGet(H,1,2);
	lamdax=cvmGet(H,0,0)/cos(angle);
	lamday=cvmGet(H,1,1)/cos(angle);
  //double estnew[4];
// 	estnew[0]=(angle*360)/(2*M_PI);
// 	estnew[1]=dx;
// 	estnew[2]=dy;
// 	estnew[3]=lamdax;
// 	estnew[4]=lamday;
// 	double s=0;
// 	for (int i=0;i<4;i++)
// 	{
// 		s+=estnew[i];
// 	}
// 	return s;
// 	delete estnew;
	double s;
	s=angle+lamdax+lamday+dx+dy;
	return s;
}
double MatDis(CvMat *H1, CvMat* H2)
{
	CvMat *dst=NULL;
	dst=cvCreateMat(3,3,CV_32FC1);
	cvAbsDiff(H1,H2,dst);
	CvScalar S;
	S=cvSum(dst);
	return S.val[0];
}







