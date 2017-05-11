#include "cvrtrees.h"

using namespace cv;



// Функция обучения случайного леса.

void trainRTrees(const Mat & trainSamples,
                const Mat & trainClasses,
                const Ptr<ml::RTrees>  params)
{

    Ptr<ml::TrainData> trainData = ml::TrainData::create(trainSamples, ml::ROW_SAMPLE, trainClasses);
    params->train(trainData);
}






Ptr<ml::RTrees> populateRTrees()
{

    Ptr<ml::RTrees> params= ml::RTrees::create();

    int treeDepth = -1;
    printf("max depth of each tree = ");
    scanf("%d", &(treeDepth));
    params->setMaxDepth(treeDepth);

    int treesNum = -1;
    printf("number of trees to build = ");
    scanf("%d", &(treesNum));
    params->setMinSampleCount(treesNum);
    params->setCalculateVarImportance(false);

    int activeVarsNum = -1;
    printf("number of active variables (set 0 to use the sqrt(total number of features)) = ");
    scanf("%d", &(activeVarsNum));
    params->setActiveVarCount(treesNum);
    params->setMaxCategories(10);
    params->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER,50,0.1));

    return params;
}
