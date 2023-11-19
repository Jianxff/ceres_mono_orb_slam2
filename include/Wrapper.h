#include <thread>
#include <memory>
#include <queue>

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>

#include "System.h"
#include "Osmap.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

class ImageStream{
public:
  ImageStream(bool force_realtime = false);
  // add new image to image queue
  void addNewImage(cv::Mat& im, double time_ms);
  // get new image from image queue
  bool getNewImage(cv::Mat& im, double& time);
  // release stream
  void release();

private:
  bool force_realtime_ = false;
  bool new_img_abailable_ = false;
  bool released_ = false;
  std::mutex img_mutex_;
  std::queue<cv::Mat> img_stream_;
  std::queue<double> img_times_;

};


class Session {
public:
  // constructor from config file
  Session(const std::string voc_file, const std::string config_file, bool force_realtime);
  // fast constructor from image size
  Session(const std::string voc_file, const int imwidth, const int imheight, bool force_realtime);


public:
  // add new image frame
  void addTrack(py::array_t<uint8_t>& input, double time_ms = -1);
  // get feature points
  py::array_t<float> getFeatures();
  // get current map
  py::array_t<uint8_t> getFrame();
  // get tracking status
  py::array_t<size_t> getTrackingState();
  // get camera twc
  Eigen::Matrix4d getCameraPoseMatrix();
  // set map save status
  void setSaveMap(bool save_map, const std::string map_name);
  // load map
  void loadMap(bool track_only, const std::string filename);
  // save pcd
  void savePointCloud(const std::string filename);
  // enable viewer thread
  void enableViewer(bool off_screen = true);
  // stop session
  void release();
  // cancel session
  void cancel();

private:
  // run thread
  void run();
  // get image from webrtc frame
  cv::Mat getImageBGR(py::array_t<uint8_t>& input);

  bool released_ = false;
	bool visualize_ = false;
  bool exit_required_ = false;
  bool save_map_ = false;
  bool save_pcd_ = false;
  
  std::string map_name_ = "";
  std::string pcd_name_ = "";

	std::shared_ptr<ORB_SLAM2::System> psystem_ = nullptr;
	std::shared_ptr<ORB_SLAM2::Viewer> pviewer_ = nullptr;
  std::shared_ptr<ImageStream> pstream_ = nullptr;
  std::shared_ptr<ORB_SLAM2::Osmap> posmap_ = nullptr;

  std::thread viewer_thread_;
  std::thread system_thread_;

};


PYBIND11_MODULE(orbslam2, m) {
  m.doc() = "Ceres Mono ORB-SLAM2 Python Wrapper";

  py::class_<Session>(m, "Session")
    .def(py::init<const std::string, const std::string, bool>(), py::arg("voc_file"), py::arg("config_file"), py::arg("force_realtime") = false)
    .def(py::init<const std::string, const int, const int, bool>(), py::arg("voc_file"), py::arg("imwidth"), py::arg("imheight"), py::arg("force_realtime") = false)
    .def("enable_viewer", &Session::enableViewer, py::arg("off_screen") = true)
    .def("add_track", &Session::addTrack, py::arg("image"), py::arg("time_ms") = -1)
    .def("tracking_state", &Session::getTrackingState)
    .def("get_position", &Session::getCameraPoseMatrix)
    .def("get_frame", &Session::getFrame)
    .def("get_features", &Session::getFeatures)
    .def("save_map", &Session::setSaveMap, py::arg("save_map"), py::arg("map_name") = "")
    .def("load_map", &Session::loadMap, py::arg("track_only") = true, py::arg("filename"))
    .def("save_pcd", &Session::savePointCloud, py::arg("filename"))
    .def("release", &Session::release)
    .def("cancel", &Session::cancel);
}


