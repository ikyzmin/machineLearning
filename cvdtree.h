#ifndef DECISION_TREE
#define DECISION_TREE

#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"

using namespace cv;


Ptr<ml::DTrees> populateDTree();



void trainDTree(const cv::Mat & trainSamples,
	const cv::Mat & trainClasses,
	const cv::Ptr<ml::DTrees> params);



#endif