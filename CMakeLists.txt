cmake_minimum_required(VERSION 3.0)
project(orbslam2)

##########################
# cmake flags
##########################
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE "Release")


##########################
# find package
##########################
find_package(OpenCV 3 REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(glog REQUIRED)
find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED)
find_package(Pangolin REQUIRED)
find_package(Protobuf REQUIRED)


##########################
# pybind11
##########################
add_subdirectory(${PROJECT_SOURCE_DIR}/Thirdparty/pybind11)


##########################
# build pybind module
##########################
pybind11_add_module(${PROJECT_NAME}
  src/System.cc
  src/Tracking.cc
  src/LocalMapping.cc
  src/LoopClosing.cc
  src/ORBextractor.cc
  src/ORBmatcher.cc
  src/FrameDrawer.cc
  src/MapPoint.cc
  src/KeyFrame.cc
  src/Map.cc
  src/MapDrawer.cc
  src/CeresOptimizer.cc
  src/PnPsolver.cc
  src/Frame.cc
  src/KeyFrameDatabase.cc
  src/Sim3Solver.cc
  src/Initializer.cc
  src/Viewer.cc
  src/MatEigenConverter.cc
  src/Osmap.cc
  src/osmap.pb.cc
  src/Wrapper.cc
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include/
    ${PROJECT_SOURCE_DIR}/Thirdparty/
    ${PROJECT_SOURCE_DIR}/Thirdparty/pybind11/include/
    /usr/include/python3.8/
)

target_link_libraries(${PROJECT_NAME} 
  PUBLIC
    Eigen3::Eigen
    Sophus::Sophus
    glog
  PRIVATE
    Ceres::ceres
    ${Pangolin_LIBRARIES}
    ${OpenCV_LIBS}
    ${PROTOBUF_LIBRARIES}
    ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
)
