//**  Libreria Estatica En Linux para el Registro de Imagenes con FP y Opt. por Esquinas
//**  GNU/Linux, gcc/g++
//**  Isnardo Reducindo Ruiz  - isnardo.rr@gmail.com
//**  Ultima Modificacion: 12 - Octubre - 2010
//**  FC, UASLP

#ifndef _REGISTRO_H
#define _REGISTRO_H
#include<stdio.h>
#include <windows.h> 
#include<cv.h>      //OpenCV
#include<highgui.h> //OpenCV
#include<cxcore.h>  //OpenCV
//#include<sys/param.h> // Para obtener el numero de Nucleos en Linux
//#include<pthread.h>   //Para crear Hilos en LINUX
//#include<string>      // Para obtener el numero de nucleos en Linux

#define FP_DELTA         8
#define M_PI        3.1415926
const static double varLK = 0.2;
using namespace std;

inline   const   int   round(const   double   x)   { 
	return   ((int)(x   +   0.5)); 
} 
//inline void GetCurrentPath(char* buffer){getcwd(buffer,MAXPATHLEN);}//Regresa el directorio actual en LINUX
inline double rand_uniform(void){return ((double)rand()/((double)(RAND_MAX)+(double)(1)));};//Random number with uniform distribution
double rand_normal(double var);	// Random Number with Normal Distribution
double cvEntropy(IplImage*& Src,IplImage*& Aux);//Entropy
void Histograma(IplImage*& Src,IplImage*& Hist,int bins);//Histogram of an Image
void cvAffine(IplImage* Src,IplImage*& Dst,double theta,double dx,double dy,
    double lambdax,double lambday,double gammax,double gammay,CvMat*& Affine,double fill);
void AffMatTransf(double* par,CvPoint2D32f center,double* mat); 
void cvJointHist(IplImage*& Src1,IplImage*& Src2,IplImage*& JtHist,
    IplImage*& xHist,IplImage*& yHist,int bins);	//Joint histogram
void chess_Window(IplImage*& Src1,IplImage*& Src2,const char* name);//Displays the image
void AddNoise(IplImage* I,IplImage* In,double std);	//Added Gaussian noise to an Image
void AddNoise(IplImage* I,double std);		//Added Gaussian noise to an Image
double TRME(double* S,double* Se,int np);	//True relative Mean Error
//Registration by Particle Filter
int FP(IplImage*& IP,IplImage*& IC,IplImage*& ICT,double* estp,int par,
       int Np);	
int NumCores(); 	//Number of Cores on PC

double graydis(IplImage* src1,IplImage* src2);
void  pixelsl (IplImage*Src1,IplImage* Src2,IplImage* aux11,IplImage* aux22);
double getHvalue(CvMat* H);
double MatDis(CvMat* H1, CvMat* H2);

//Fast access to IplImage
template<class T>
class Image
{
   private:
   IplImage* imgp;
   public:
   Image(IplImage* img=0) {imgp=img;}
   ~Image(){imgp=0;}
   void operator=(IplImage* img) {imgp=img;}
   inline T* operator[](const int rowIndx){
     return ((T*)(imgp->imageData+rowIndx*imgp->widthStep));}
};
typedef struct{
  unsigned char b,g,r;
}RgbPixel;
typedef struct{
   float b,g,r;
}RgbPixelFloat;

typedef struct{
   double b,g,r;
}RgbPixelFloat64;

typedef Image<RgbPixel>		RgbImage;
typedef Image<RgbPixelFloat>	RgbImageFloat;
typedef Image<RgbPixelFloat64>	RgbImageFloat64;
typedef Image<unsigned char>	BwImage;
typedef Image<float>		BwImageFloat;
typedef Image<double>		BwImageFloat64;
//-----------------------
//Functions of threads
DWORD WINAPI HiloGenEv (LPVOID Estructura);	//Generate and evaluate Particles
DWORD WINAPI HiloResamp(LPVOID Estructura);	//Resamples Particles

//Particle class
class Particulas{
	public:
		Particulas(){};		//Constructor
		~Particulas();		//Destructor
		//Create variables in Class Particle
		void Create(IplImage* I,IplImage* C,IplImage*& allP,int N,double* allW,
		    double*AccW,int nH,int Np,int par);
		void Evalua();		//Evaluates particleS and obtain the weights
		void Generate();	//Generates N Particles
		void Resampling();	//Resamples Particles 
		void GetP(IplImage*& allP);		//Llena el Array de Todas las Particulas con las del Hilo // Fill the Array of all particles in the Threads
		void SetP(IplImage*& allP){cvCopyImage(allP,this->allP);};	//Envia el Array
		void Reset(IplImage* IP,IplImage* IC);	//Resets the iteration
		double* Estimado();
		double GetSumW(){return sumW;};	//Generates the Sum of Weights
		double VarDeg();
		//affine transformation 
		void Affine(double theta,double dx,double dy,double lambdax,double lambday,
		    double gammax,double gammay,double fill);
		void AffineM(double a00,double a01,double a02,double a10,double a11,double a12,double fill);
	private:
		IplImage* IP;	//Reference image
		IplImage* IC;	//Register image
		IplImage* ICT;	//Register Image Transform
		IplImage* P;	//Particles in Thread
		IplImage* P_1;	//Particles in Thread t-1
		IplImage* Aux1,*Aux2;	//Auxiliaries for the entropies
		IplImage* Puv,*Pu,*Pv;	//Histograms
		IplImage* allP;	//Array of all particles of all threads
		CvMat* mAff;	//Matrix of Affine transformations 

		IplImage* Aux3,*Aux4; // // for calculate the disffence of similar gray values

		double* W;		//Particle Weights
		double* AccW;	//Cumulative Weights 
		double* var;	//Array para las Varianzas de las perturbaciones // Array for the Variance of disturbances
		double* varVel; //Array for the Velocity of disturbances
		double* varAc;  //Array for Acceleration of disturbances
		double* varM;	//Array for the variation of the transformation matrix
		double* est;	//Array for the estimated parameters
		double* ide;	//Identity array
		double* ideM;	//Identity matrix 

		double fac;		// Reduction factor of the Variance of Disturbances
		double facVel;	//Reduction factor of the Velocity of the Disturbance
		double facAc;	//Reduction factor of the Acceleration of Disturbances
		double u;		//Uniform random number for the Resampling
		double c1,c2,c3,c4;//Constants for the Likelihood
		double sumW;	//Sum of Weights
		double IM,Hx;	//Mutual Information and Entropy
		double cx,cy;	//Rotation center for transformation
		double a1,a2,a3,a4,a5,a6; //Auxiliaries for Affine Transfer 
		double nW;		//Factor to calculate the estimated 1.0/Np 

		double DX;      // match similar gray values 

		int width,height;	//Length and width of images
		int nH,indH;		//Number of Particles
		int i,j,k,ite;		//Counters
		int Np;				//Total Number of Particles
		int N;				//Number of Particles in Thread
		int par;			//Number of parameters to be evaluated - 5 for the TRME
		int type_reg;		//Tipo de Registro Parcial y No-Parcial // Record Type and Non-Partial Partial
		int type_ext;		//Tipo de Extencion del Filtro (Normal, Velocida,Aceleracion)// Extension of the filter type (Normal, velocities, acceleration)
		int type_est;		//Type Parameters estimated 7 - Matrix 6

		const static int  np = 7;		//Parameters Number
		const static char bins = 16;	//Histogram Beans
		
};
#endif

