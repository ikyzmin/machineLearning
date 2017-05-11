cmake_minimum_required(VERSION 3.6)
project(machineLearning)

set(CMAKE_CXX_STANDARD 11)

set(OpenCV_DIR /home/openCV/release_qt)

find_package(OpenCV REQUIRED)

set(SOURCE_FILES main.cpp
        cvdtree.cpp
        cvgbtrees.cpp
        cvsvm.cpp
        cvrtrees.cpp
        errorMetrics.cpp
        drawingFunctions.cpp
        auxilary.cpp
        dataset1.yml
        dataset2.yml
        dataset3.yml
        dataset4.yml
        datasetHighDim.yml
        datasetMulticlass.yml
        dataset1KMeans.yml
        dataset2KMeans.yml
        dataset3KMeans.yml
        dataset4KMeans.yml)


add_executable(machineLearning ${SOURCE_FILES})

target_link_libraries(machineLearning ${OpenCV_LIBS})
