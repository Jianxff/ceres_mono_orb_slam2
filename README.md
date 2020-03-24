# Mono Camera ORB-SLAM2 with ceres solver optimizer

## What this repository for?
1. Only for mono camera slam
2. Remove g2o from ORB-SLAM2 and use ceres solver instead to get rid of annoying warnings
3. Use Eigen::Matrix instead of cv::Mat for matrix calculation
4. Tested on rgbd_dataset_freiburg2_desk, rgbd_dataset_freiburg2_360_kidnap and kitti

## Usage
**PreInstall**

[Install ceres_solver](http://www.ceres-solver.org/installation.html)

[Install Pangolin](https://github.com/stevenlovegrove/Pangolin)

[Download Datasets](#Monocular)

**Build**
```bash
$ git clone https://github.com/b51/ceres_mono_orb_slam2.git
$ cd ceres_mono_orb_slam2
$ mkdir build && cd build
$ cmake .. && make -j4
```

**Run**
```bash
$ cd ceres_mono_orb_slam2
$ cd vocabulary
$ tar zxvf ORBvoc.txt.tar.gz
$ cd ../build
$ ./mono_slam --voc ../vocabulary/ORBvoc.txt --config ../configs/KITTI00-02.yaml --images path/to/kitti_images
```

## TODO
  - [X] Sim3 Optimizer to ceres
  - [X] g2oSim3 to be remove
  - [X] Totally remove g2o to get rid of the tons annoying Warnings
  - [X] cv::Mat for matrix to be remove
  - [X] Fix CeresOptimizer, sometimes get "Matrix not positive definite" warning
  - [X] Fix Relocalization, sometimes may get stuck
  - [X] Fix LocalBundleAdjustment with add inv sigma to error calculation
  - [X] EssentialGraph in CeresOptimizer need to fix
  - [ ] Reconstruct to make code modular, at least make feature matcher as an independent module
  - [ ] Add IMU data to get scale information
  - [ ] Add map save for map reusing


## Error and Jacobian calculation of sim(3)
**!!! Chrome extension [TeX All the Things](https://chrome.google.com/webstore/detail/tex-all-the-things/cbimabofgmfdkicghcadidpemeenbffn) is neccessary for Latex below display !!!**

### Some neccessary equations before Jacobian calculation
**1. With the property of Lie Algebra Adjoint, [Reference](https://blog.csdn.net/heyijia0327/article/details/51773578)**
$$\mathbf{S * exp(\hat{\zeta}) * S^{-1} = exp[(Adj(S) * \zeta)^{\Lambda}]}$$
We can get equations below
$$\mathbf{S * exp(\hat{\zeta}) = exp[(Adj(S) * \zeta)^{\Lambda}] * S}$$
and
$$\mathbf{exp(\hat{\zeta}) * S^{-1} = S^{-1} * exp[(Adj(S) * \zeta)^{\Lambda}]}$$

**2. Baker-Campbell-Hausdorf equations, [STATE ESTIMATION FOR ROBOTICS](http://asrl.utias.utoronto.ca/~tdb/bib/barfoot_ser17.pdf) P.234**

$\hspace{2cm}\mathbf{ln(S_1 S_2)^{v} = ln(exp[\hat{\xi_1]}exp[\hat{\xi_2]})^{v}}$

$\hspace{4cm}\mathbf{ = \xi_1 + \xi_2 + \frac{1}{2} \xi_1^{\lambda} \xi_2 + \frac{1}{12}\xi_1^{\lambda} \xi_1^{\lambda} \xi_2 + \frac{1}{12}\xi_2^{\lambda} \xi_2^{\lambda} \xi_1 + \dots}$

$\hspace{4cm}\mathbf{\approx J_l(\xi_2)^{-1}\xi_1 + \xi_2}\hspace{2cm}$ (if $\xi_1 small$)

$\hspace{4cm}\mathbf{\approx \xi_2 + J_r(\xi_1)^{-1}\xi_2}\hspace{2cm}$ (if $\xi_2 small$)

With
$\hspace{3cm}\mathbf{J_l(\xi)^{-1} = \sum_{n = 0}^{\infty} \frac{B_n}{n!} (\xi^{\lambda})^{n}}$

$\hspace{3cm}\mathbf{J_r(\xi)^{-1} = \sum_{n = 0}^{\infty} \frac{B_n}{n!} (-\xi^{\lambda})^{n}}$

With$\hspace{1cm}B_0 = 1, B_1 = -\frac{1}{2}, B_2 = \frac{1}{6}, B_3 = 0, B_4 = -\frac{1}{30}\dots$, $\hspace{5mm}\mathbf{\xi^{\lambda} = adj(\xi)}$, $\hspace{5mm}$ is adjoint matrix of $\xi$

**3. Adjoint Matrix of sim(3)**
a) Main property of adjoint matrix on Lie Algebras, [Reference: LIE GROUPS AND LIE ALGEBRAS, 1.6](http://www.math.jhu.edu/~fspinu/423/7.pdf)
$$\mathbf{[x, y] = adj(x)y}$$

b) sim3 Lie Brackets, [Reference: Local Accuracy and Global Consistency for Efficient Visual SLAM, P.184, A.3.4](https://www.doc.ic.ac.uk/~ajd/Publications/Strasdat-H-2012-PhD-Thesis.pdf):
$$\mathbf{[x, y] = [\begin{bmatrix} \nu \newline \omega \newline \sigma\end{bmatrix} \begin{bmatrix} \tau \newline \varphi \newline \varsigma \end{bmatrix}] = \begin{bmatrix}\omega \times \tau + \nu \times \varphi + \sigma\tau - \varsigma\nu \newline \omega \times \varphi \newline  0 \end{bmatrix}}$$
$$\mathbf{= \begin{bmatrix}(\hat{\omega} + \sigma I)\tau + \nu \times \varphi - \varsigma\nu \newline \omega \times \varphi \newline  0 \end{bmatrix}}$$
$$\hspace{-4cm} = \mathbf{adj(x) y}$$

We can get $\hspace{4cm}\mathbf{\xi^{\lambda} = adj(\xi) = \begin{bmatrix} (\hat{\omega} + \sigma I) & \hat{\nu} & -{\nu} \newline 0 & \hat{\omega} & 0 \newline 0 & 0 & 0 \end{bmatrix}}$

### Left multiplication/Right multiplication for pose update
sim(3) update with **Left multiplication/Right multiplication** has affect on Jacobian calculation, Formula derivation below used Right multiplication as example

### Jacobian Calculation of sim(3)
$$\mathbf{error = S_{ji}  S_{iw}  S_{jw}^{-1}}$$

Derivation of $\mathbf{Jacobian_i}$

$\hspace{6cm}\mathbf{ln(error(\xi_i + \delta_i))^v = ln(S_{ji}S_{iw}exp(\hat{\delta_i})S_{jw}^{-1})^{v}}$

$\hspace{10cm}\mathbf{= ln(S_{ji}S_{iw}S_{jw}^{-1}exp[(Adj(S_{jw}){\delta_i})^{\Lambda}])^{v}}$

$\hspace{10cm}\mathbf{= ln(exp(\xi_{error}) \cdot exp[(Adj(S_{jw}){\delta_i})^{\Lambda}])^{v}}$

$\hspace{10cm}\mathbf{= \xi_{error} + J_r(\xi_{error})^{-1} Adj(S_{jw}){\delta_i}}$

$$\mathbf{Jacobian_i = \frac{\partial ln(error)^v}{\partial \delta_i} = \lim_{\delta_i \to 0} \frac{ln(error(\xi_i + \delta_i))^v - ln(error)^v}{\delta_i} = J_r(\xi_{error})^{-1} Adj(S_{jw}) }$$

Same to $\mathbf{Jacobian_j}$

$\hspace{6cm}\mathbf{ln(error(\xi_i + \delta_j)) = ln(S_{ji}S_{iw}(S_{jw}exp(\hat{\delta_j}))^{-1})^{v}}$
$\hspace{10cm}\mathbf{= ln(S_{ji}S_{iw}exp(-\hat{\delta_j})S_{jw}^{-1})^{v}}$

$\hspace{10cm}\mathbf{= ln(exp(\xi_{error}) \cdot exp[(Adj(S_{jw})(-\delta_j))^{\Lambda}])^{v}}$

$\hspace{10cm}\mathbf{= \xi_{error} - J_r(\xi_{error})^{-1} Adj(S_{jw}){\delta_j}}$

$$\mathbf{Jacobian_j = \frac{\partial ln(error)^v}{\partial \delta_j} = \lim_{\delta_j \to 0} \frac{ln(error(\xi_i + \delta_j))^v - ln(error)^v}{\delta_j} = -J_r(\xi_{error})^{-1} Adj(S_{jw}) }$$


# ORB-SLAM2
**Authors:** [Raul Mur-Artal](http://webdiis.unizar.es/~raulmur/), [Juan D. Tardos](http://webdiis.unizar.es/~jdtardos/), [J. M. M. Montiel](http://webdiis.unizar.es/~josemari/) and [Dorian Galvez-Lopez](http://doriangalvez.com/) ([DBoW2](https://github.com/dorian3d/DBoW2))

**13 Jan 2017**: OpenCV 3 and Eigen 3.3 are now supported.

**22 Dec 2016**: Added AR demo (see section 7).

ORB-SLAM2 is a real-time SLAM library for **Monocular**, **Stereo** and **RGB-D** cameras that computes the camera trajectory and a sparse 3D reconstruction (in the stereo and RGB-D case with true scale). It is able to detect loops and relocalize the camera in real time. We provide examples to run the SLAM system in the [KITTI dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) as stereo or monocular, in the [TUM dataset](http://vision.in.tum.de/data/datasets/rgbd-dataset) as RGB-D or monocular, and in the [EuRoC dataset](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) as stereo or monocular. We also provide a ROS node to process live monocular, stereo or RGB-D streams. **The library can be compiled without ROS**. ORB-SLAM2 provides a GUI to change between a *SLAM Mode* and *Localization Mode*, see section 9 of this document.

<a href="https://www.youtube.com/embed/ufvPS5wJAx0" target="_blank"><img src="http://img.youtube.com/vi/ufvPS5wJAx0/0.jpg" 
alt="ORB-SLAM2" width="240" height="180" border="10" /></a>
<a href="https://www.youtube.com/embed/T-9PYCKhDLM" target="_blank"><img src="http://img.youtube.com/vi/T-9PYCKhDLM/0.jpg" 
alt="ORB-SLAM2" width="240" height="180" border="10" /></a>
<a href="https://www.youtube.com/embed/kPwy8yA4CKM" target="_blank"><img src="http://img.youtube.com/vi/kPwy8yA4CKM/0.jpg" 
alt="ORB-SLAM2" width="240" height="180" border="10" /></a>


### Related Publications:

[Monocular] Raúl Mur-Artal, J. M. M. Montiel and Juan D. Tardós. **ORB-SLAM: A Versatile and Accurate Monocular SLAM System**. *IEEE Transactions on Robotics,* vol. 31, no. 5, pp. 1147-1163, 2015. (**2015 IEEE Transactions on Robotics Best Paper Award**). **[PDF](http://webdiis.unizar.es/~raulmur/MurMontielTardosTRO15.pdf)**.

[Stereo and RGB-D] Raúl Mur-Artal and Juan D. Tardós. **ORB-SLAM2: an Open-Source SLAM System for Monocular, Stereo and RGB-D Cameras**. *IEEE Transactions on Robotics,* vol. 33, no. 5, pp. 1255-1262, 2017. **[PDF](https://128.84.21.199/pdf/1610.06475.pdf)**.

[DBoW2 Place Recognizer] Dorian Gálvez-López and Juan D. Tardós. **Bags of Binary Words for Fast Place Recognition in Image Sequences**. *IEEE Transactions on Robotics,* vol. 28, no. 5, pp.  1188-1197, 2012. **[PDF](http://doriangalvez.com/php/dl.php?dlp=GalvezTRO12.pdf)**

# 1. License

ORB-SLAM2 is released under a [GPLv3 license](https://github.com/raulmur/ORB_SLAM2/blob/master/License-gpl.txt). For a list of all code/library dependencies (and associated licenses), please see [Dependencies.md](https://github.com/raulmur/ORB_SLAM2/blob/master/Dependencies.md).

For a closed-source version of ORB-SLAM2 for commercial purposes, please contact the authors: orbslam (at) unizar (dot) es.

If you use ORB-SLAM2 (Monocular) in an academic work, please cite:

    @article{murTRO2015,
      title={{ORB-SLAM}: a Versatile and Accurate Monocular {SLAM} System},
      author={Mur-Artal, Ra\'ul, Montiel, J. M. M. and Tard\'os, Juan D.},
      journal={IEEE Transactions on Robotics},
      volume={31},
      number={5},
      pages={1147--1163},
      doi = {10.1109/TRO.2015.2463671},
      year={2015}
     }

if you use ORB-SLAM2 (Stereo or RGB-D) in an academic work, please cite:

    @article{murORB2,
      title={{ORB-SLAM2}: an Open-Source {SLAM} System for Monocular, Stereo and {RGB-D} Cameras},
      author={Mur-Artal, Ra\'ul and Tard\'os, Juan D.},
      journal={IEEE Transactions on Robotics},
      volume={33},
      number={5},
      pages={1255--1262},
      doi = {10.1109/TRO.2017.2705103},
      year={2017}
     }

# 2. Prerequisites
We have tested the library in **Ubuntu 12.04**, **14.04** and **16.04**, but it should be easy to compile in other platforms. A powerful computer (e.g. i7) will ensure real-time performance and provide more stable and accurate results.

## C++11 or C++0x Compiler
We use the new thread and chrono functionalities of C++11.

## Pangolin
We use [Pangolin](https://github.com/stevenlovegrove/Pangolin) for visualization and user interface. Dowload and install instructions can be found at: https://github.com/stevenlovegrove/Pangolin.

## OpenCV
We use [OpenCV](http://opencv.org) to manipulate images and features. Dowload and install instructions can be found at: http://opencv.org. **Required at leat 2.4.3. Tested with OpenCV 2.4.11 and OpenCV 3.2**.

## Eigen3
Required by g2o (see below). Download and install instructions can be found at: http://eigen.tuxfamily.org. **Required at least 3.1.0**.

## DBoW2 and g2o (Included in Thirdparty folder)
We use modified versions of the [DBoW2](https://github.com/dorian3d/DBoW2) library to perform place recognition and [g2o](https://github.com/RainerKuemmerle/g2o) library to perform non-linear optimizations. Both modified libraries (which are BSD) are included in the *Thirdparty* folder.

## ROS (optional)
We provide some examples to process the live input of a monocular, stereo or RGB-D camera using [ROS](ros.org). Building these examples is optional. In case you want to use ROS, a version Hydro or newer is needed.

# 3. Building ORB-SLAM2 library and examples

Clone the repository:
```
git clone https://github.com/raulmur/ORB_SLAM2.git ORB_SLAM2
```

We provide a script `build.sh` to build the *Thirdparty* libraries and *ORB-SLAM2*. Please make sure you have installed all required dependencies (see section 2). Execute:
```
cd ORB_SLAM2
chmod +x build.sh
./build.sh
```

This will create **libORB_SLAM2.so**  at *lib* folder and the executables **mono_tum**, **mono_kitti**, **rgbd_tum**, **stereo_kitti**, **mono_euroc** and **stereo_euroc** in *Examples* folder.

# <a name='Monocular'></a>4. Monocular Examples

## TUM Dataset

1. Download a sequence from http://vision.in.tum.de/data/datasets/rgbd-dataset/download and uncompress it.

2. Execute the following command. Change `TUMX.yaml` to TUM1.yaml,TUM2.yaml or TUM3.yaml for freiburg1, freiburg2 and freiburg3 sequences respectively. Change `PATH_TO_SEQUENCE_FOLDER`to the uncompressed sequence folder.
```
./Examples/Monocular/mono_tum Vocabulary/ORBvoc.txt Examples/Monocular/TUMX.yaml PATH_TO_SEQUENCE_FOLDER
```

## KITTI Dataset  

1. Download the dataset (grayscale images) from http://www.cvlibs.net/datasets/kitti/eval_odometry.php 

2. Execute the following command. Change `KITTIX.yaml`by KITTI00-02.yaml, KITTI03.yaml or KITTI04-12.yaml for sequence 0 to 2, 3, and 4 to 12 respectively. Change `PATH_TO_DATASET_FOLDER` to the uncompressed dataset folder. Change `SEQUENCE_NUMBER` to 00, 01, 02,.., 11. 
```
./Examples/Monocular/mono_kitti Vocabulary/ORBvoc.txt Examples/Monocular/KITTIX.yaml PATH_TO_DATASET_FOLDER/dataset/sequences/SEQUENCE_NUMBER
```

## EuRoC Dataset

1. Download a sequence (ASL format) from http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

2. Execute the following first command for V1 and V2 sequences, or the second command for MH sequences. Change PATH_TO_SEQUENCE_FOLDER and SEQUENCE according to the sequence you want to run.
```
./Examples/Monocular/mono_euroc Vocabulary/ORBvoc.txt Examples/Monocular/EuRoC.yaml PATH_TO_SEQUENCE_FOLDER/mav0/cam0/data Examples/Monocular/EuRoC_TimeStamps/SEQUENCE.txt 
```

```
./Examples/Monocular/mono_euroc Vocabulary/ORBvoc.txt Examples/Monocular/EuRoC.yaml PATH_TO_SEQUENCE/cam0/data Examples/Monocular/EuRoC_TimeStamps/SEQUENCE.txt 
```

# 5. Stereo Examples

## KITTI Dataset

1. Download the dataset (grayscale images) from http://www.cvlibs.net/datasets/kitti/eval_odometry.php 

2. Execute the following command. Change `KITTIX.yaml`to KITTI00-02.yaml, KITTI03.yaml or KITTI04-12.yaml for sequence 0 to 2, 3, and 4 to 12 respectively. Change `PATH_TO_DATASET_FOLDER` to the uncompressed dataset folder. Change `SEQUENCE_NUMBER` to 00, 01, 02,.., 11. 
```
./Examples/Stereo/stereo_kitti Vocabulary/ORBvoc.txt Examples/Stereo/KITTIX.yaml PATH_TO_DATASET_FOLDER/dataset/sequences/SEQUENCE_NUMBER
```

## EuRoC Dataset

1. Download a sequence (ASL format) from http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

2. Execute the following first command for V1 and V2 sequences, or the second command for MH sequences. Change PATH_TO_SEQUENCE_FOLDER and SEQUENCE according to the sequence you want to run.
```
./Examples/Stereo/stereo_euroc Vocabulary/ORBvoc.txt Examples/Stereo/EuRoC.yaml PATH_TO_SEQUENCE/mav0/cam0/data PATH_TO_SEQUENCE/mav0/cam1/data Examples/Stereo/EuRoC_TimeStamps/SEQUENCE.txt
```
```
./Examples/Stereo/stereo_euroc Vocabulary/ORBvoc.txt Examples/Stereo/EuRoC.yaml PATH_TO_SEQUENCE/cam0/data PATH_TO_SEQUENCE/cam1/data Examples/Stereo/EuRoC_TimeStamps/SEQUENCE.txt
```

# 6. RGB-D Example

## TUM Dataset

1. Download a sequence from http://vision.in.tum.de/data/datasets/rgbd-dataset/download and uncompress it.

2. Associate RGB images and depth images using the python script [associate.py](http://vision.in.tum.de/data/datasets/rgbd-dataset/tools). We already provide associations for some of the sequences in *Examples/RGB-D/associations/*. You can generate your own associations file executing:

  ```
  python associate.py PATH_TO_SEQUENCE/rgb.txt PATH_TO_SEQUENCE/depth.txt > associations.txt
  ```

3. Execute the following command. Change `TUMX.yaml` to TUM1.yaml,TUM2.yaml or TUM3.yaml for freiburg1, freiburg2 and freiburg3 sequences respectively. Change `PATH_TO_SEQUENCE_FOLDER`to the uncompressed sequence folder. Change `ASSOCIATIONS_FILE` to the path to the corresponding associations file.

  ```
  ./Examples/RGB-D/rgbd_tum Vocabulary/ORBvoc.txt Examples/RGB-D/TUMX.yaml PATH_TO_SEQUENCE_FOLDER ASSOCIATIONS_FILE
  ```

# 7. ROS Examples

### Building the nodes for mono, monoAR, stereo and RGB-D
1. Add the path including *Examples/ROS/ORB_SLAM2* to the ROS_PACKAGE_PATH environment variable. Open .bashrc file and add at the end the following line. Replace PATH by the folder where you cloned ORB_SLAM2:

  ```
  export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:PATH/ORB_SLAM2/Examples/ROS
  ```
  
2. Execute `build_ros.sh` script:

  ```
  chmod +x build_ros.sh
  ./build_ros.sh
  ```
  
### Running Monocular Node
For a monocular input from topic `/camera/image_raw` run node ORB_SLAM2/Mono. You will need to provide the vocabulary file and a settings file. See the monocular examples above.

  ```
  rosrun ORB_SLAM2 Mono PATH_TO_VOCABULARY PATH_TO_SETTINGS_FILE
  ```
  
### Running Monocular Augmented Reality Demo
This is a demo of augmented reality where you can use an interface to insert virtual cubes in planar regions of the scene.
The node reads images from topic `/camera/image_raw`.

  ```
  rosrun ORB_SLAM2 MonoAR PATH_TO_VOCABULARY PATH_TO_SETTINGS_FILE
  ```
  
### Running Stereo Node
For a stereo input from topic `/camera/left/image_raw` and `/camera/right/image_raw` run node ORB_SLAM2/Stereo. You will need to provide the vocabulary file and a settings file. If you **provide rectification matrices** (see Examples/Stereo/EuRoC.yaml example), the node will recitify the images online, **otherwise images must be pre-rectified**.

  ```
  rosrun ORB_SLAM2 Stereo PATH_TO_VOCABULARY PATH_TO_SETTINGS_FILE ONLINE_RECTIFICATION
  ```
  
**Example**: Download a rosbag (e.g. V1_01_easy.bag) from the EuRoC dataset (http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets). Open 3 tabs on the terminal and run the following command at each tab:
  ```
  roscore
  ```
  
  ```
  rosrun ORB_SLAM2 Stereo Vocabulary/ORBvoc.txt Examples/Stereo/EuRoC.yaml true
  ```
  
  ```
  rosbag play --pause V1_01_easy.bag /cam0/image_raw:=/camera/left/image_raw /cam1/image_raw:=/camera/right/image_raw
  ```
  
Once ORB-SLAM2 has loaded the vocabulary, press space in the rosbag tab. Enjoy!. Note: a powerful computer is required to run the most exigent sequences of this dataset.

### Running RGB_D Node
For an RGB-D input from topics `/camera/rgb/image_raw` and `/camera/depth_registered/image_raw`, run node ORB_SLAM2/RGBD. You will need to provide the vocabulary file and a settings file. See the RGB-D example above.

  ```
  rosrun ORB_SLAM2 RGBD PATH_TO_VOCABULARY PATH_TO_SETTINGS_FILE
  ```
  
# 8. Processing your own sequences
You will need to create a settings file with the calibration of your camera. See the settings file provided for the TUM and KITTI datasets for monocular, stereo and RGB-D cameras. We use the calibration model of OpenCV. See the examples to learn how to create a program that makes use of the ORB-SLAM2 library and how to pass images to the SLAM system. Stereo input must be synchronized and rectified. RGB-D input must be synchronized and depth registered.

# 9. SLAM and Localization Modes
You can change between the *SLAM* and *Localization mode* using the GUI of the map viewer.

### SLAM Mode
This is the default mode. The system runs in parallal three threads: Tracking, Local Mapping and Loop Closing. The system localizes the camera, builds new map and tries to close loops.

### Localization Mode
This mode can be used when you have a good map of your working area. In this mode the Local Mapping and Loop Closing are deactivated. The system localizes the camera in the map (which is no longer updated), using relocalization if needed. 

