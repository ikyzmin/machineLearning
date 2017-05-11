#ifndef RANDOM_FOREST_TREE
#define RANDOM_FOREST_TREE

#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"
using namespace cv;


Ptr<ml::RTrees> populateRTrees();

void trainRTrees(const cv::Mat & trainSamples,
                const cv::Mat & trainClasses,
                const Ptr<ml::RTrees>  params);



#endif