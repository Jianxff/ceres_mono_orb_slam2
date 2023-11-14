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
    std::lock_guard<std::mutex> lock(img_mutex_);
    img_stream_.push(im);
    img_times_.push(time_ms);

    new_img_abailable_ = true;
  }

  // get new image from image queue
  bool getNewImage(cv::Mat& im, double& time) {
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

private:
  bool force_realtime_ = false;
  bool new_img_abailable_ = false;
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
    initialized_ = true;
  }

  Session(const std::string voc_file, const int imwidth, const int imheight, bool force_realtime) {
    psystem_.reset(new ORB_SLAM2::System(voc_file, imwidth, imheight));
    pstream_.reset(new ImageStream(force_realtime));
    posmap_.reset(new ORB_SLAM2::Osmap(*psystem_));
    system_thread_ = std::thread(&Session::run, this);
    initialized_ = true;
  }

public:
  // enable viewer thread
  void enableViewer(const std::string config_file) {
    if(!initialized_) {
			puts("Not initialized!\n");
			return;
		}

    pviewer_.reset(new ORB_SLAM2::Viewer(psystem_.get(), config_file));
    viewer_thread_ = std::thread(&ORB_SLAM2::Viewer::Run, pviewer_);
    visualize_ = true;
  }

  // add new image frame
  void addTrack(py::array_t<uint8_t>& input, double time_ms = -1){
    if(!initialized_) {
        puts("Not initialized!\n");
        return;
    }

    cv::Mat image = getImageBGR(input);
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    if(time_ms < 0) {
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch());
      time_ms = (double)ms.count();
    }

    pstream_->addNewImage(image, time_ms);
  }

  // get tracking status
  int getTrackingState() {
    if(!initialized_) {
      puts("Not initialized!\n");
      return -2;
    }
    int state = psystem_->GetTrackingState();
    return state;
  }

  // get camera twc
  Eigen::Matrix4d getCameraPoseMatrix() {
    if(!initialized_) {
      puts("Not initialized!\n");
      return Eigen::Matrix4d::Identity();
    }

    Eigen::Matrix4d Twc = psystem_->GetTwc();
    return Twc;
  }

  // save map
  void saveMap(const std::string filename) {
    if(!initialized_) {
      puts("Not initialized!\n");
      return;
    }
    posmap_->options.set(ORB_SLAM2::Osmap::NO_FEATURES_DESCRIPTORS | ORB_SLAM2::Osmap::ONLY_MAPPOINTS_FEATURES, 1); 
    posmap_->mapSave(filename);

    std::cout << "Map saved to " << filename << std::endl;
  } 

  // load map
  void loadMap(const std::string filename){
    if(!initialized_) {
      puts("Not initialized!\n");
      return;
    }
    posmap_->mapLoad(filename);
    psystem_->ActivateLocalizationMode();

    std::cout << "Map loaded from " << filename << std::endl;
  }


  // stop session
  void stop(std::string traj_filename = "") {
    exit_required_ = true;
    pviewer_->RequestFinish();
    pviewer_->RequestStop();

    system_thread_.join();
    viewer_thread_.join();

    psystem_->Shutdown();
    if(traj_filename.length() > 0)
      psystem_->SaveKeyFrameTrajectoryTUM(traj_filename + ".txt");

    psystem_.reset();
    pviewer_.reset();
    pstream_.reset();

    initialized_ = false;
    visualize_ = false;

    puts("[Session] All threads Stopped");
  }


private:
  void run() {
    cv::Mat img;
    double time;
    while( !exit_required_ ) {
      if(pstream_->getNewImage(img, time))
        psystem_->TrackMonocular(img, time);
       else 
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }


  cv::Mat getImageBGR(py::array_t<uint8_t>& input) {
    if(input.ndim() != 3) 
      throw std::runtime_error("get Image : number of dimensions must be 3");
    py::buffer_info buf = input.request();
		cv::Mat image(buf.shape[0], buf.shape[1], CV_8UC3, (uint8_t*)buf.ptr);
    return image;
  }

  bool initialized_ = false;
	bool visualize_ = false;
  bool exit_required_ = false;

	std::shared_ptr<ORB_SLAM2::System> psystem_;
	std::shared_ptr<ORB_SLAM2::Viewer> pviewer_;
  std::shared_ptr<ImageStream> pstream_;
  std::shared_ptr<ORB_SLAM2::Osmap> posmap_;

  std::thread viewer_thread_;
  std::thread system_thread_;

};


PYBIND11_MODULE(orbslam2, m) {
  m.doc() = "ORB-SLAM2 python wrapper";

  py::class_<Session>(m, "Session")
    .def(py::init<const std::string, const std::string, bool>(), py::arg("voc_file"), py::arg("config_file"), py::arg("force_realtime") = false)
    .def(py::init<const std::string, const int, const int, bool>(), py::arg("voc_file"), py::arg("imwidth"), py::arg("imheight"), py::arg("force_realtime") = false)
    .def("enable_viewer", &Session::enableViewer, py::arg("config_file"))
    .def("add_track", &Session::addTrack, py::arg("image"), py::arg("time_ms") = -1)
    .def("tracking_state", &Session::getTrackingState)
    .def("camera_pose", &Session::getCameraPoseMatrix)
    .def("save_map", &Session::saveMap, py::arg("filename"))
    .def("load_map", &Session::loadMap, py::arg("filename"))
    .def("stop", &Session::stop, py::arg("traj_filename") = "");
}


