cmake_minimum_required(VERSION 2.8.3)
project(ceres_mono_orb_slam2)

#########################################################################
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_BUILD_TYPE "Release")
#########################################################################


# OpenCV
find_package(OpenCV 3 REQUIRED)
# Eigen3
find_package(Eigen3 REQUIRED NO_MODULE)
# glog
find_package(glog REQUIRED)
# Ceres
find_package(Ceres REQUIRED)
# Sophus
find_package(Sophus REQUIRED)
# Pangolin
find_package(Pangolin REQUIRED)
# Protobuf
find_package(Protobuf REQUIRED)


###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/Thirdparty
)

add_library(${PROJECT_NAME} SHARED
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
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/Thirdparty
)

target_link_libraries(${PROJECT_NAME} 
  PUBLIC
    Eigen3::Eigen
    Sophus::Sophus
    glog
    ${Pangolin_LIBRARIES}
    ${OpenCV_LIBS}
    ${PROTOBUF_LIBRARIES}
  PRIVATE
    Ceres::ceres
    ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
    
)

add_executable(mono_euroc
  example/mono_euroc.cc
)

target_include_directories(mono_euroc
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(mono_euroc
  PUBLIC
    ${PROJECT_NAME}
)
