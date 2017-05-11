#include "cvdtree.h"

using namespace cv;


void trainDTree(const Mat & trainSamples,
                const Mat & trainClasses,
				Ptr<ml::DTrees> dtree)
{
   	Ptr<ml::TrainData> trainData = ml::TrainData::create(trainSamples, ml::ROW_SAMPLE, trainClasses);
	dtree->train(trainData);
}

Ptr<ml::DTrees> populateDTree()
{
	Ptr<ml::DTrees> params = ml::DTrees::create();

    params->setUseSurrogates(false);
    params->setTruncatePrunedTree(true);
    params->setUse1SERule(true);
	int tmp;
    printf("maximal tree depth = ");
    scanf("%d", &tmp);
	params->setMaxDepth(tmp);
    printf("minimal number of samples in leaf = ");
    scanf("%d", &tmp);
	params->setMinSampleCount(tmp);

    int doPruning = 0;
    printf("apply pruning (0/1) = ");
    scanf("%d", &(doPruning));
    params->setCVFolds((doPruning == 0) ? 0 : 5);
    return params;
}

