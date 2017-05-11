#ifndef GRADIENT_BOOSTED_TREES
#define GRADIENT_BOOSTED_TREES

#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"

using namespace cv;



// Функция чтения параметров алгоритма обучения градиентного
// бустинга деревьев решений с консоли.


Ptr<ml::Boost> populateBoostedTree();


void trainBoosted(const cv::Mat & trainSamples,
                const cv::Mat & trainClasses,
                const Ptr<ml::Boost>  params);


#endif