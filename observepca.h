/** @file
 * 
 * Moghaddam's PCA DIFS + DFFS (distance-in-feature-space + distance-from-feature-space) 
 * observation model for particle filter
 * CvParticleState must have x, y, width, height, and angle
 *
 * The MIT License
 * 
 * Copyright (c) 2008, Naotoshi Seo <sonots(at)sonots.com>
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#ifndef CV_PARTICLE_OBSERVE_PCADIFFS_H
#define CV_PARTICLE_OBSERVE_PCADIFFS_H

#include "opencvx/cvparticle.h"
#include "opencvx/cvrect32f.h"
#include "opencvx/cvcropimageroi.h"
#include "opencvx/cvpcadiffs.h"
#include "opencvx/cvgaussnorm.h"
#include <iostream>
using namespace std;

/********************************* Globals ******************************************/
int    num_observes = 1;

/******************************* Globals in this file ******************************/
CvMat *eigenvalues;
CvMat *eigenvectors;
CvMat *eigenavg;
CvSize feature_size;

/****************************** Function Prototypes ********************************/
#ifndef NO_DOXYGEN
void cvParticleObserveInitialize();
void cvParticleObserveFinalize();
void icvPreprocess( const IplImage* patch, CvMat *mat );
void icvGetFeatures( const CvParticle* p, const IplImage* frame, CvMat* features );
void cvParticleObserveMeasure( CvParticle* p, IplImage* cur_frame, IplImage *pre_frame );
#endif

/****************************** Functions ******************************************/

/**
 * Initialization
 */
void cvParticleObserveInitialize( const char* data_pcaval, 
                                  const char* data_pcavec,
                                  const char* data_pcaavg,
                                  CvSize feat_size = cvSize(0, 0) )
{
    if( (eigenvalues = (CvMat*)cvLoad( data_pcaval )) == NULL ) {
        cerr << data_pcaval << " is not loadable." << endl << flush;
        exit( 1 );
    }
    if( (eigenvectors = (CvMat*)cvLoad( data_pcavec )) == NULL ) {
        cerr << data_pcavec << " is not loadable." << endl << flush;
        exit( 1 );
    }
    if( (eigenavg = (CvMat*)cvLoad( data_pcaavg )) == NULL ) {
        cerr << data_pcaavg << " is not loadable." << endl << flush;
        exit( 1 );
    }
    if( feat_size.width == 0 && feat_size.height == 0 ) {
        feature_size = cvSize( (int)sqrt((double)eigenavg->rows), (int)sqrt((double)eigenavg->rows) );
    } else {
        feature_size = feat_size;
    }
    if( eigenavg->rows != feature_size.width * feature_size.height ) {
        cerr << " Feature size is wrong " << feature_size.width << " x " << feature_size.height << endl << flush;
        exit( 1 );
    }
}

/**
 * Finalization
 */
void cvParticleObserveFinalize()
{
    cvReleaseMat( &eigenvalues );
    cvReleaseMat( &eigenvectors );
    cvReleaseMat( &eigenavg );
}

/**
 * Preprocess as did in training PCA subspace
 */
void icvPreprocess( const IplImage* patch, CvMat *mat )
{
    IplImage *gry;
    if( patch->nChannels != 1 ) {
        gry = cvCreateImage( cvGetSize(patch), patch->depth, 1 );
        cvCvtColor( patch, gry, CV_BGR2GRAY );
    } else {
        gry = (IplImage*)patch;
    }
    IplImage *resize = cvCreateImage( cvSize(mat->rows, mat->cols), patch->depth, 1 );

    cvResize( gry, resize );
    cvConvert( resize, mat );
    cvGaussNormImage( mat, mat );

    cvReleaseImage( &resize );
    if( gry != patch )
        cvReleaseImage( &gry );
}

/**
 * Get observation features
 *
 * CvParticleState must have x, y, width, height, angle
 */
void icvGetFeatures( const CvParticle* p, const IplImage* frame, CvMat* features )
{
    int feature_height = feature_size.height;
    int feature_width  = feature_size.width;
    //cvNamedWindow( "patch" );
    CvMat* normed = cvCreateMat( feature_height, feature_width, CV_64FC1 );
    CvMat* normedT = cvCreateMat( feature_width, feature_height, CV_64FC1 );
    CvMat* feature, featurehdr;
    IplImage *patch;
    for( int n = 0; n < p->num_particles; n++ ) {
        CvParticleState s = cvParticleStateGet( p, n );
        CvBox32f box32f = cvBox32f( s.x, s.y, s.width, s.height, s.angle );
        CvRect32f rect32f = cvRect32fFromBox32f( box32f );

        // get image patch and preprocess
        patch = cvCreateImage( cvSize( cvRound( s.width ), cvRound( s.height ) ), 
                               frame->depth, frame->nChannels );
        cvCropImageROI( (IplImage*)frame, patch, rect32f );
        //cvShowImage( "patch", patch );
        //cvWaitKey( 10 );
        icvPreprocess( patch, normed );
        cvReleaseImage( &patch );

        // vectorize
        cvT( normed, normedT ); // transpose to make the same with matlab's reshape
        feature = cvReshape( normedT, &featurehdr, 1, feature_height * feature_width );

        cvSetCol( feature, features, n );
    }
    cvReleaseMat( &normedT );
    cvReleaseMat( &normed );
}

/**
 * Measure and weight particles. 
 *
 * The proposal function q is set p(xt|xt-1) in SIR/Condensation, and it results 
 * that "weights" are set to be proportional to the likelihood probability 
 * (Normalize later).
 * Rewrite here if you want to use a different proposal function q. 
 *
 * CvParticleState s must have s.x, s.y, s.width, s.height, s.angle
 *
 * @param particle
 * @param frame
 * @param reference
 */
void cvParticleObserveMeasure( CvParticle* p, IplImage* frame )
{
    int feature_height = feature_size.height;
    int feature_width  = feature_size.width;

    // extract features from particle states
    CvMat* features = cvCreateMat( feature_height*feature_width, p->num_particles, CV_64FC1 );
    icvGetFeatures( p, frame, features );
    
    // Likelihood measurments
    cvMatPcaDiffs( features, eigenavg, eigenvalues, eigenvectors, p->weights, 0, true );

    cvReleaseMat( &features );
}

#endif
