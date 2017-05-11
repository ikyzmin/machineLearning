#include <stdio.h>
#include "cvsvm.h"

#pragma warning(disable : 4996)

using namespace cv;



// Функция обучения SVM-модели.
//
void trainSVM(const cv::Mat & trainSamples,
              const cv::Mat & trainClasses,
              //const CvSVMParams & params,
              Ptr<ml::SVM> svm)
{
	Ptr<ml::TrainData> trainData = ml::TrainData::create(trainSamples, ml::ROW_SAMPLE, trainClasses);
	svm->trainAuto(trainData);
}


Mat getSupportVectors(const Ptr<ml::SVM> svm)
{
  	return svm->getSupportVectors();
}


Ptr<ml::SVM> populateSVM()
{
	Ptr<cv::ml::SVM> params = cv::ml::SVM::create();
	params->setType(ml::SVM::C_SVC);
	params->setTermCriteria(
        cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000000, 0.0001));
    return params;
}

