#!/bin/bash

echo ""
echo "Building DBoW2 lib!"
echo ""

cd Thirdparty/DBoW2

mkdir build
cd build/
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="../install/"
make -j4
cd ../../..


echo ""
echo "Building Sophus lib!"
echo ""

cd Thirdparty/Sophus

mkdir build
mkdir install
cd build/
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="../install/"
make -j4 install
cd ../../..

echo ""
echo "Building Ceres lib!"
echo ""

cd Thirdparty/ceres-solver
mkdir build
mkdir install
cd build/
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=14 -DCMAKE_CXX_FLAGS="-march=native" -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF
make -j4 install
cd ../../..

 