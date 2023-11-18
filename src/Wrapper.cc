#include <thread>
#include <memory>
#include <chrono>
#include <queue>

#include <opencv2/core.hpp>
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
  ImageStream(bool force_realtime = false) : force_realtime_(force_realtime) {}

  // add new image to image queue
  void addNewImage(cv::Mat& im, double time_ms) {
    if(released_) return;

    std::lock_guard<std::mutex> lock(img_mutex_);
    img_stream_.push(im);
    img_times_.push(time_ms);

    new_img_abailable_ = true;
  }

  // get new image from image queue
  bool getNewImage(cv::Mat& im, double& time) {
    if(released_) return false;

    std::lock_guard<std::mutex> lock(img_mutex_);
    if(!new_img_abailable_) return false;

    do {
        im = img_stream_.front();
        img_stream_.pop();

        time = img_times_.front();
        img_times_.pop();

        if( !force_realtime_ ) break; // check force_realtime to skip frames

    } while( !img_stream_.empty() );

    new_img_abailable_ = !img_stream_.empty();

    return true;
  }

  void release() {
    released_ = true;
  }

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
  // static constructor
  Session(const std::string voc_file, const std::string config_file, bool force_realtime) {
    psystem_.reset(new ORB_SLAM2::System(voc_file,config_file));
    pstream_.reset(new ImageStream(force_realtime));
    posmap_.reset(new ORB_SLAM2::Osmap(*psystem_));
    system_thread_ = std::thread(&Session::run, this);
  }

  Session(const std::string voc_file, const int imwidth, const int imheight, bool force_realtime) {
    psystem_.reset(new ORB_SLAM2::System(voc_file, imwidth, imheight));
    pstream_.reset(new ImageStream(force_realtime));
    posmap_.reset(new ORB_SLAM2::Osmap(*psystem_));
    system_thread_ = std::thread(&Session::run, this);
  }


public:
  // add new image frame
  void addTrack(py::array_t<uint8_t>& input, double time_ms = -1){
    if(released_) return;

    cv::Mat image = getImageBGR(input);
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    if(time_ms < 0) {
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
      time_ms = (double)ms.count();
    }

    pstream_->addNewImage(image, time_ms);
  }

  // get feature points
  py::array_t<float> getFeatures() {
    if(released_) return py::array_t<float>();

    auto& kps = psystem_->tracker_->current_frame_.keypoints_;
    std::vector<float> features;
    features.reserve(kps.size() * 2);
    for(auto& kp : kps) {
      features.emplace_back(kp.pt.x);
      features.emplace_back(kp.pt.y);
    }

    return py::array_t<float>({(int)kps.size(), 2}, features.data());
  }

  // get current map
  py::array_t<uint8_t> getFrame() {
    if(released_) 
      return py::array_t<uint8_t>();

    if(!visualize_) {
      puts("Viewer not enabled!");
      return py::array_t<uint8_t>();
    }

    cv::Mat frame = pviewer_->GetFrame();
    return py::array_t<uint8_t>({frame.rows, frame.cols, 3}, frame.data);
  }

  // get tracking status
  int getTrackingState() {
    if(released_) return -1;

    int state = psystem_->GetTrackingState();
    return state;
  }

  // get camera twc
  Eigen::Matrix4d getCameraPoseMatrix() {
    if(released_) return Eigen::Matrix4d::Identity();

    Eigen::Matrix4d Twc = psystem_->GetTwc();
    return Twc;
  }

  // set map save status
  void setSaveMap(bool save_map, const std::string map_name) {
    if(released_) return;

    if(save_map && map_name.length() > 0) {
      save_map_ = true;
      map_name_ = map_name;
      std::cout << "Map " << map_name << " will be saved when sytem stop." << std::endl;
    } else {
      save_map_ = false;
    }
  }

  // load map
  void loadMap(bool track_only, const std::string filename){
    if(released_) return;

    posmap_->mapLoad(filename);
    std::cout << "Map loaded from " << filename << std::endl;

    if(track_only) {
      psystem_->ActivateLocalizationMode();
      std::cout << "Localization mode activated" << std::endl;
    }
  }
 
  // enable viewer thread
  void enableViewer() {
    pviewer_.reset(new ORB_SLAM2::Viewer(psystem_.get()));
    viewer_thread_ = std::thread(&ORB_SLAM2::Viewer::Run, pviewer_);
    visualize_ = true;
  }

  // stop session
  void release() {
    if(released_) return;
    pstream_->release();
    // stop viewer if enabled
    if(pviewer_ != nullptr)
      pviewer_->exit_required_ = true;
    
    exit_required_ = true;
    viewer_thread_.join();

    psystem_->Shutdown();
    system_thread_.join();

    if(save_map_) {
      posmap_->options.set(ORB_SLAM2::Osmap::NO_FEATURES_DESCRIPTORS | ORB_SLAM2::Osmap::ONLY_MAPPOINTS_FEATURES, 1); 
      posmap_->mapSave(map_name_);
      std::cout << "Map " << map_name_ << " saved" << std::endl;
    }

    psystem_.reset();
    pviewer_.reset();
    pstream_.reset();

    visualize_ = false;

    cv::destroyAllWindows();

    released_ = true;
    puts("Session Released");
  }



private:
  // run thread
  void run() {
    cv::Mat img;
    double time;
    while( !exit_required_ ) {
      if(pstream_->getNewImage(img, time)) {
        psystem_->TrackMonocular(img, time);
      } else 
        std::this_thread::sleep_for(std::chrono::milliseconds(1000 / 60));
    }

    std::cout << "Stop Processing New Frame" << std::endl;
  }

  // get image from webrtc frame
  cv::Mat getImageBGR(py::array_t<uint8_t>& input) {
    if(input.ndim() != 3) 
      throw std::runtime_error("get Image : number of dimensions must be 3");
    py::buffer_info buf = input.request();
		cv::Mat image(buf.shape[0], buf.shape[1], CV_8UC3, (uint8_t*)buf.ptr);
    return image;
  }


  bool released_ = false;
	bool visualize_ = false;
  bool exit_required_ = false;
  bool save_map_ = false;
  
  std::string map_name_ = "";

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
    .def("enable_viewer", &Session::enableViewer)
    .def("add_track", &Session::addTrack, py::arg("image"), py::arg("time_ms") = -1)
    .def("tracking_state", &Session::getTrackingState)
    .def("get_position", &Session::getCameraPoseMatrix)
    .def("get_frame", &Session::getFrame)
    .def("get_features", &Session::getFeatures)
    .def("save_map", &Session::setSaveMap, py::arg("save_map"), py::arg("map_name") = "")
    .def("load_map", &Session::loadMap, py::arg("track_only") = true, py::arg("filename"))
    .def("release", &Session::release);
}


