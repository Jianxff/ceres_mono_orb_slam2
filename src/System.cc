/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: System.cc
 *
 *          Created On: Tue 03 Sep 2019 06:27:56 PM CST
 *     Licensed under The GPLv3 License [see LICENSE for details]
 *
 ************************************************************************/

#include "System.h"

#include <pangolin/pangolin.h>
#include <unistd.h>
#include <iomanip>
#include <thread>

namespace ORB_SLAM2 {

System::System(const std::string& voc_file, const int imwidth, const int imheight)
  : do_reset_(false),
    is_activate_localization_mode_(false),
    is_deactivate_localization_mode_(false) 
{
  // Load ORB Vocabulary
  std::cout << std::endl
            << "Loading ORB Vocabulary. This could take a while..."
            << std::endl;

  vocabulary_ = new ORBVocabulary();
  // bool voc_loaded = vocabulary_->loadFromTextFile(voc_file);
  bool voc_loaded = vocabulary_->loadFromBinaryFile(voc_file);
  if (!voc_loaded) {
    LOG(ERROR) << "Wrong path to vocabulary.";
    LOG(FATAL) << "Failed to open at: " << voc_file;
  }

  LOG(INFO) << "Vocabulary loaded!";

  // Create KeyFrame Database
  keyframe_database_ = new KeyFrameDatabase(*vocabulary_);

  // Create the Map
  map_ = new Map();

  // Initialize the Tracking thread
  //(it will live in the main thread of execution, the one that called this
  // constructor)
  tracker_ = new Tracking(this, vocabulary_, map_, keyframe_database_, imwidth, imheight);

  // Initialize the Local Mapping thread and launch
  local_mapper_ = new LocalMapping(map_, true);
  thread_local_mapping_ = new thread(&ORB_SLAM2::LocalMapping::Run, local_mapper_);

  // Initialize the Loop Closing thread and launch
  loop_closer_ = new LoopClosing(map_, keyframe_database_, vocabulary_, false);
  thread_loop_closing_ = new thread(&ORB_SLAM2::LoopClosing::Run, loop_closer_);

  // Set pointers between threads
  tracker_->SetLocalMapper(local_mapper_);
  tracker_->SetLoopClosing(loop_closer_);

  local_mapper_->SetTracker(tracker_);
  local_mapper_->SetLoopCloser(loop_closer_);

  loop_closer_->SetTracker(tracker_);
  loop_closer_->SetLocalMapper(local_mapper_);

  Twc_ = Eigen::Matrix4d::Identity();
}


System::System(const string& voc_file, const string& string_setting_file)
  : do_reset_(false),
    is_activate_localization_mode_(false),
    is_deactivate_localization_mode_(false) 
{
  std::cout
      << std::endl
      << "ORB-SLAM2 Copyright (C) 2014-2016 Raul Mur-Artal, University of "
         "Zaragoza."
      << std::endl
      << "ceres_mono_orb_slam2 Copyright (C) 2019 b51live"
      << std::endl
      << "This program comes with ABSOLUTELY NO WARRANTY;" << std::endl
      << "This is free software, and you are welcome to redistribute it"
      << std::endl
      << "under certain conditions. See LICENSE.txt." << std::endl
      << std::endl;

  // Check settings file
  cv::FileStorage fsSettings(string_setting_file.c_str(),
                             cv::FileStorage::READ);
  if (!fsSettings.isOpened()) {
    LOG(FATAL) << "Failed to open settings file at: " << string_setting_file;
  }

  // Load ORB Vocabulary
  std::cout << std::endl
            << "Loading ORB Vocabulary. This could take a while..."
            << std::endl;

  vocabulary_ = new ORBVocabulary();
  // bool voc_loaded = vocabulary_->loadFromTextFile(voc_file);
  bool voc_loaded = vocabulary_->loadFromBinaryFile(voc_file);
  if (!voc_loaded) {
    LOG(ERROR) << "Wrong path to vocabulary.";
    LOG(FATAL) << "Failed to open at: " << voc_file;
  }

  LOG(INFO) << "Vocabulary loaded!";

  // Create KeyFrame Database
  keyframe_database_ = new KeyFrameDatabase(*vocabulary_);

  // Create the Map
  map_ = new Map();

  // Initialize the Tracking thread
  //(it will live in the main thread of execution, the one that called this
  // constructor)
  tracker_ = new Tracking(this, vocabulary_, map_, keyframe_database_, string_setting_file);

  // Initialize the Local Mapping thread and launch
  local_mapper_ = new LocalMapping(map_, true);
  thread_local_mapping_ =
      new thread(&ORB_SLAM2::LocalMapping::Run, local_mapper_);

  // Initialize the Loop Closing thread and launch
  loop_closer_ = new LoopClosing(map_, keyframe_database_, vocabulary_, false);
  thread_loop_closing_ = new thread(&ORB_SLAM2::LoopClosing::Run, loop_closer_);

  // Set pointers between threads
  tracker_->SetLocalMapper(local_mapper_);
  tracker_->SetLoopClosing(loop_closer_);

  local_mapper_->SetTracker(tracker_);
  local_mapper_->SetLoopCloser(loop_closer_);

  loop_closer_->SetTracker(tracker_);
  loop_closer_->SetLocalMapper(local_mapper_);

  Twc_ = Eigen::Matrix4d::Identity();
}

Eigen::Matrix4d System::TrackMonocular(const cv::Mat& img,
                                            const double& timestamp) {
  {
    unique_lock<mutex> lock(mutex_mode_);
    if (is_activate_localization_mode_) {
      local_mapper_->RequestStop();

      // Wait until Local Mapping has effectively stopped
      while (!local_mapper_->isStopped()) {
        usleep(1000);
      }
      tracker_->InformOnlyTracking(true);
      is_activate_localization_mode_ = false;
    }

    if (is_deactivate_localization_mode_) {
      tracker_->InformOnlyTracking(false);
      local_mapper_->Release();
      is_deactivate_localization_mode_ = false;
    }
  }

  {
    unique_lock<mutex> lock(mutex_reset_);
    if (do_reset_) {
      Twc_ = Eigen::Matrix4d::Identity();
      tracker_->Reset();
      do_reset_ = false;
    }
  }

  Eigen::Matrix4d Tcw = tracker_->GrabImageMonocular(img, timestamp);
  {
    std::lock_guard<mutex> lock(mutex_pose_);
    Twc_ = Tcw.inverse();
  }

  unique_lock<mutex> lock2(mutex_state_);
  tracking_state_ = tracker_->state_;
  tracked_map_points_ = tracker_->current_frame_.map_points_;
  tracked_undistort_keypoints_ = tracker_->current_frame_.undistort_keypoints_;

  return Tcw;
}

Eigen::Matrix4d System::GetTwc() {
  std::lock_guard<mutex> lock(mutex_pose_);
  return Twc_;
}

void System::ActivateLocalizationMode() {
  unique_lock<mutex> lock(mutex_mode_);
  is_activate_localization_mode_ = true;
}

void System::DeactivateLocalizationMode() {
  unique_lock<mutex> lock(mutex_mode_);
  is_deactivate_localization_mode_ = true;
}

bool System::MapChanged() {
  static int n = 0;
  int curn = map_->GetLastBigChangeIdx();
  if (n < curn) {
    n = curn;
    return true;
  } else {
    return false;
  }
}

void System::Reset() {
  unique_lock<mutex> lock(mutex_reset_);
  do_reset_ = true;
}

void System::Shutdown() {
  local_mapper_->RequestFinish();
  loop_closer_->RequestFinish();

  // Wait until all thread have effectively stopped
  thread_local_mapping_->join();
  thread_loop_closing_->join();
}

void System::SaveTrajectoryTUM(const string& filename) {
  LOG(INFO) << "Cannot save camera trajectory for mono";
  return;
}


void System::SaveKeyFrameTrajectoryTUM(const string& filename) {
  LOG(INFO) << "Saving keyframe trajectory to " << filename << " ...";

  std::vector<KeyFrame*> keyframes = map_->GetAllKeyFrames();
  sort(keyframes.begin(), keyframes.end(), KeyFrame::lId);

  // Transform all keyframes so that the first keyframe is at the origin.
  // After a loop closure the first keyframe might not be at the origin.
  // cv::Mat Two = vpKFs[0]->GetPoseInverse();
  ofstream f;
  f.open(filename.c_str());
  f << fixed;

  for (size_t i = 0; i < keyframes.size(); i++) {
    KeyFrame* keyframe = keyframes[i];

    if (keyframe->isBad()) {
      continue;
    }

    Eigen::Matrix3d R = keyframe->GetRotation().transpose();
    Eigen::Quaterniond q(R);
    Eigen::Vector3d t = keyframe->GetCameraCenter();
    f << setprecision(6) << keyframe->timestamp_ << setprecision(7) << " "
      << t[0] << " " << t[1] << " " << t[2] << " " << q.x() << " " << q.y()
      << " " << q.z() << " " << q.w() << endl;
  }
  f.close();
  LOG(INFO) << "trajectory saved!";
}

int System::GetTrackingState() {
  unique_lock<mutex> lock(mutex_state_);
  return tracking_state_;
}

std::vector<MapPoint*> System::GetTrackedMapPoints() {
  unique_lock<mutex> lock(mutex_state_);
  return tracked_map_points_;
}

std::vector<cv::KeyPoint> System::GetTrackedKeyPointsUn() {
  unique_lock<mutex> lock(mutex_state_);
  return tracked_undistort_keypoints_;
}

}  // namespace ORB_SLAM2
