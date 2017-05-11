#include <stdio.h>
#include <map>
#include "opencv2/core/core.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "cvsvm.h"
#include "cvdtree.h"
#include "cvrtrees.h"
#include "cvgbtrees.h"
#include "auxilary.h"
#include "errorMetrics.h"

#pragma warning(disable : 4996)

using namespace cv;
using std::map;


// Максимальная длина строки с названием графического окна
const int maxWinNameLen = 1000;
// Код символа ESC
const int escCode = 27;

// Перечисление задающее действие выбранное в главном меню
enum {LOAD_DATA = 0, USE_SVM, USE_DTREE, USE_RTREES, USE_GBTREES,USE_K_MEANS};


/*
// Функция печати главного меню и выбора пунтка пользователем.
// 
// API
// int getMainMenuAction()
//
// РЕЗУЛЬТАТ
// Номер выбранного пользователем пункта меню.
*/
int getMainMenuAction()
{
    int menuItemIdx = -1;
    const int menuTabsCount = 6;
    const char* menu[] = 
    {
        "0 - load points",
        "1 - SVM",
        "2 - decision tree",
        "3 - random forest",
        "4 - boosted trees",
        "5 - k-means"
    };

    while (true)
    {
        // Печатаем основное меню
        printf("Main menu:\n");
        for (int i = 0; i < menuTabsCount; i++)
        {
            printf("\t%s\n", menu[i]);
        }
        printf("\n");
        // Выбор пользователя
        printf("Your choice = ");
        scanf("%d", &menuItemIdx);
        // Проверка на ввыод корректного номера
        if (menuItemIdx >= 0 && menuItemIdx < menuTabsCount)
        {
            break;
        }
    }
    return menuItemIdx;
}

Mat generateDataset()
{
    int n = 300;
    Mat data(3 * n, 2, CV_32F);
    randn(data(Range(0, n), Range(0, 1)), 0.0, 0.05);
    randn(data(Range(0, n), Range(1, 2)), 0.5, 0.25);
    randn(data(Range(n, 2 * n), Range(0, 1)), 0.7, 0.25);
    randn(data(Range(n, 2 * n), Range(1, 2)), 0.0, 0.05);
    randn(data(Range(2 * n, 3 * n), Range(0, 1)),
          0.7, 0.15);
    randn(data(Range(2 * n, 3 * n), Range(1, 2)),
          0.8, 0.15);
    return data;
}

int main(int argc, char** argv)
{
    // матрицы для хранения обучающей и тестовой выборок
    Mat featuresTrain;
    Mat classesTrain;
    Mat featuresTest;
    Mat classesTest;

    // кол-ва моделей каждого типа построенных на текущих данных
    int svmModelsNum = 0;
    int dtreeModelsNum = 0;
    int rtreesModelsNum = 0;
    int gbtreesModelsNum = 0;

    // имя текущего графического окна
    char winName[maxWinNameLen] = {0};

    // отображение номеров классов в BGR-цвета
    map<int, Scalar> classColors;
    classColors[0] = Scalar(255, 191, 0);
    classColors[1] = Scalar(0, 215, 255);
    classColors[2] = Scalar(71, 99, 255);
    classColors[3] = Scalar(0, 252, 124);
    classColors[4] = Scalar(240, 32, 160);

    // загружены ли данные
    bool isDataLoaded = false;

    int ans;
    do
    {
        // вызов функции выбора пункта меню
        int activeMenuItem = getMainMenuAction();
        Ptr<ml::StatModel> model;
        //getPredictedClassLabel * predictFunction;
		PredictionFunction * predictFunction;

        switch (activeMenuItem)
        {
        case LOAD_DATA:
            {
                destroyAllWindows();
                readDatasetFromFile(featuresTrain,
                    classesTrain,
                    featuresTest,
                    classesTest);
                svmModelsNum = 0;
                dtreeModelsNum = 0;
                rtreesModelsNum = 0;
                gbtreesModelsNum = 0;
                continue;
            }; break;

        case USE_SVM:
            {
               Ptr<ml::SVM> svm = populateSVM();
              trainSVM(featuresTrain,
                       classesTrain,
                       svm);
				 sprintf(winName, "SVM #%d\0", svmModelsNum++);
                model = svm;
				predictFunction = new SVMPrediction();
            }; break;
			
        case USE_DTREE:
            {
               	Ptr<ml::DTrees> dtree = populateDTree();
                trainDTree(featuresTrain,
                    classesTrain,
	                dtree);

                sprintf(winName, "Decision tree #%d\0", dtreeModelsNum++);
                model = dtree;
                predictFunction = new DTreePrediction();
            }; break;

        case USE_RTREES:
            {
                 Ptr<ml::RTrees> params = populateRTrees();
                trainRTrees(featuresTrain, classesTrain, params);

                sprintf(winName, "Random forest #%d\0", rtreesModelsNum++);
                model = params;
                predictFunction = new RTreePrediction();
            }; break;
        case USE_GBTREES:
            {
                // Запрашиваем параметры алгоритма обучения
                Ptr<ml::Boost> params = populateBoostedTree();

                // Запускаем алгоритм обучения
                trainBoosted(featuresTrain, classesTrain, params);

                sprintf(winName, "Gradient boosting #%d\0", gbtreesModelsNum++);
                model = params;
                predictFunction = new BoostPrediction();
            }; break;
            case USE_K_MEANS:
                Mat samples;
                readDatasetFromFileForKMeans(samples);
                Mat labels;
                Mat centers;
                Mat img(400, 400, CV_8UC3, Scalar(255, 255, 255));
                kmeans(samples,
                       3,
                       labels,
                       TermCriteria(TermCriteria::COUNT +
                                    TermCriteria::EPS, 10000, 0.001),
                       10,
                       KMEANS_PP_CENTERS,
                       centers);

                drawPoints(img, samples, labels, getRanges(samples), classColors, 0);
                drawPoints(img, centers, labels, getRanges(samples), classColors, 2);
                namedWindow("clusters");
                imshow("clusters", img);
                break;
        }

        if (activeMenuItem!=USE_K_MEANS){
        // Вычисляем ошибки на обучающей и тестовой выборках.
        float trainError = getClassificationError(featuresTrain,
            classesTrain,
            model,
			false,
            predictFunction);
        float testError = getClassificationError(featuresTest,
            classesTest,
            model,
			true,
            predictFunction);
        
        printf("========== %s ===========\n", winName);
        printf("error on train set: %.3f\n", trainError);
        printf("error on test set: %.3f\n", testError);

        // Если пространство признаков двумерное -- делаем визуализацию
        if (featuresTrain.cols == 2) {
            // Создаем белое изображение с разрешением 400х400
            Mat img(400, 400, CV_8UC3, Scalar(255, 255, 255));
            // Рисуем разбиение пространсва признаков на области
            drawPartition(img,
                          classColors,
                          getRanges(featuresTest),
                          Size(77, 77),
                          model,
                          predictFunction);
            // Отрисовываем точки тестовой выборки
            drawPoints(img, featuresTest, classesTest, getRanges(featuresTest), classColors, 1);
            // Отрисовываем точки обучающей выборки
            drawPoints(img, featuresTrain, classesTrain, getRanges(featuresTest), classColors, 0);

            //CvSVM * svm = dynamic_cast<CvSVM*>(model);
            Ptr<ml::SVM> svm = model.dynamicCast<ml::SVM>();
            if (svm) {
                // Отрисовываем опорные векторы
                Mat supportVectors = getSupportVectors(svm);
                drawPoints(img, supportVectors, classesTrain, getRanges(featuresTest), classColors, 2);
                //svm = 0;
            }
            //*/
            // Выводим изображение в новом окне
            namedWindow(winName);
            imshow(winName, img);
        }
        }
        //delete model;
        //model = 0;
        predictFunction = 0;

        printf("Do you want to continue? ESC - exit\n");
        // Ожидаем нажатия клавиши в графическом окне.
        ans = waitKey();
    }
    while (ans != escCode);

    // Закрываем все графические окна.
    destroyAllWindows();
    return 0;
}