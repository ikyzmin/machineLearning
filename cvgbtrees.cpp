#include "cvgbtrees.h"

using namespace cv;



// Функция обучения градиентного бустинга
// деревьев решений.

void trainBoosted(const cv::Mat & trainSamples,
                const cv::Mat & trainClasses,
                const Ptr<ml::Boost>  params
                )
{

    Ptr<ml::TrainData> trainData = ml::TrainData::create(trainSamples, ml::ROW_SAMPLE, trainClasses);
    params->train(trainData);
}


Ptr<ml::Boost>  populateBoostedTree()
{

    Ptr<ml::Boost> params = ml::Boost::create();
    int treeDepth = -1;
    printf("max depth of each tree = ");
    scanf("%d", &(treeDepth));
    params->setMaxDepth(treeDepth);


    int treesNum = -1;
    printf("number of trees to build = ");
    scanf("%d", &(treesNum));
    params->setWeakCount(treesNum);


    float learningRate = 0.0f;
    printf("learning rate (shrinkage) = ");
    scanf("%f", &(learningRate));
    params->setUseSurrogates(false);


    return params;
}
