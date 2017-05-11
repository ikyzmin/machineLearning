#ifndef SUPPORT_VECTOR_MACHINE
#define SUPPORT_VECTOR_MACHINE

#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"

cv::Ptr<cv::ml::SVM> populateSVM();

void trainSVM(const cv::Mat & trainSamples,
              const cv::Mat & trainClasses,
              cv::Ptr<cv::ml::SVM> svm);



int getSVMPrediction(const cv::Mat & sample,
	cv::ml::SVM & model);



cv::Mat getSupportVectors(const cv::Ptr<cv::ml::SVM> svm);


#endif