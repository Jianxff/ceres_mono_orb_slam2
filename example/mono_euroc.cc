/*************************************************************************
 *
 *              Author: b51
 *                Mail: b51live@gmail.com
 *            FileName: main.cc
 *
 *          Created On: Tue 03 Sep 2019 06:19:56 PM CST
 *     Licensed under The GPLv3 License [see LICENSE for details]
 *
 ************************************************************************/

// ./mono_euroc ../vocabulary/ORBvocS.bin ../configs/EuRoC.yaml ~/dataset/mav0/cam0/data ../configs/EuRoC_TimeStamps/V101.txt 

#include <glog/logging.h>
#include <iostream>

#include "System.h"
#include "Osmap.h"

void LoadImages(const string &strImagePath, const string &strPathTimes,
                vector<string> &vstrImages, vector<double> &vTimeStamps)
{
    ifstream fTimes;
    fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(5000);
    vstrImages.reserve(5000);
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            vstrImages.push_back(strImagePath + "/" + ss.str() + ".png");
            double t;
            ss >> t;
            vTimeStamps.push_back(t/1e9);

        }
    }
}


int main(int argc, char* argv[]) {
  google::InitGoogleLogging(argv[0]);
  google::SetStderrLogging(google::GLOG_INFO);

  if(argc != 6 && argc != 5)
  {
    LOG(FATAL) << "Usage: ./mono_euroc path_to_vocabulary path_to_settings path_to_image_folder path_to_times_file";
  }

  vector<string> vstrImageFilenames;
  vector<double> vTimestamps;
  LoadImages(string(argv[3]), string(argv[4]), vstrImageFilenames, vTimestamps);

  int n_images = vstrImageFilenames.size();

  ORB_SLAM2::System system(argv[1], argv[2], true);
  ORB_SLAM2::Osmap osmap = ORB_SLAM2::Osmap(system);

  if(argc == 6) {
    osmap.mapLoad("map.yaml");
    system.ActivateLocalizationMode();
  }
  

  std::vector<float> times_track;
  times_track.resize(n_images);

  LOG(INFO) << "----------";
  LOG(INFO) << "Start processing sequence ...";
  LOG(INFO) << "Images in the sequence: " << n_images;

  for (int i = 0; i < n_images; i++) {
    cv::Mat img = cv::imread(vstrImageFilenames[i],CV_LOAD_IMAGE_UNCHANGED);
    double tframe = vTimestamps[i];

    if (img.empty()) {
      LOG(FATAL) << "Failed to load image at: "
                 << vstrImageFilenames[i];
    }
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

    system.TrackMonocular(img, tframe);

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

    double ttrack =
        std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1)
            .count();

    times_track[i] = ttrack;

    double T = 0;
    if (i < n_images - 1) {
      T = vTimestamps[i + 1] - tframe;
    } else if (i > 0) {
      T = tframe - vTimestamps[i - 1];
    }

    if (ttrack < T) {
      usleep((T - ttrack) * 1e6);
    }
    // if (i == n_images - 1) cv::waitKey(0);
  }

  system.Shutdown();
  sort(times_track.begin(), times_track.end());
  float total_time = 0;
  for (int i = 0; i < n_images; i++) {
    total_time += times_track[i];
  }
  LOG(INFO) << "-------------";
  LOG(INFO) << "median tracking time: " << times_track[n_images / 2];
  LOG(INFO) << "mean tracking time: " << total_time / n_images;

  system.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
  
  
  if(argc == 5) {
    LOG(INFO) << "Saving map to map.yaml";
    osmap.options.set(ORB_SLAM2::Osmap::NO_FEATURES_DESCRIPTORS | ORB_SLAM2::Osmap::ONLY_MAPPOINTS_FEATURES, 1); 
    osmap.mapSave("map");
  }

  cv::waitKey(0);
  return 0;
}
