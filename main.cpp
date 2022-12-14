#include <iostream>
#include <stdlib.h>
#include <stdint.h>
#include <limits>
#include <cmath>
#include <thread>
#include <memory>
#include <dirent.h>
#include <atomic>
#include <condition_variable>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/viz.hpp>

constexpr int MIN_MASK_REPEAT_COUNT = 5;
constexpr int MAX_MASK_REPEAT_COUNT = 40;
constexpr float BEST_DIFFERENCE_COEF = 0.3f;
constexpr float MATCH_WIDTH_COEF = 4;

static cv::viz::Viz3d window("Show image");
static std::atomic_bool exitFlag(false);
static std::atomic_int nextFileFlag(0);
static std::string needLoadPathFile;
static cv::Mat currentImage;
static std::atomic_bool readFileSuccessFinished{false};
static std::mutex readFileMutex;
static std::condition_variable condVariableReadFile;
static std::mutex pressKeyMutex;
static std::condition_variable condVariablePressKey;

int findRepetableImageSize(const cv::Mat& stereoImage)
{
  if (stereoImage.empty()) {
    return 0;
  }

  double minDifferenceBetweenPixels = -1;
  int betterRepeatImageSize = 0;

  for (int i = stereoImage.cols / MAX_MASK_REPEAT_COUNT; 
           i < stereoImage.cols / MIN_MASK_REPEAT_COUNT; ++i) {            
    double currentDifference = 0;

    for (int j = 0; j < stereoImage.rows; ++j) {
      for (int k = 0; k < stereoImage.cols - i; ++k) {

        auto countChannels = stereoImage.channels();
        for (int channel = 0; channel < countChannels; ++channel) {
          currentDifference += 
            std::abs(stereoImage.at<uint8_t>(j, k * countChannels + channel) -
                     stereoImage.at<uint8_t>(j, (k + i) * countChannels + channel));
        }
      }
    }

    if ((currentDifference < minDifferenceBetweenPixels || minDifferenceBetweenPixels < 0) && 
        ( !betterRepeatImageSize ||
          i % betterRepeatImageSize != 0 || 
          (minDifferenceBetweenPixels - currentDifference) > minDifferenceBetweenPixels * BEST_DIFFERENCE_COEF)) {
      betterRepeatImageSize = i;
      minDifferenceBetweenPixels = currentDifference;
    }
  }
  return betterRepeatImageSize;
}

cv::Mat reconstructionDepth(const cv::Mat &stereoImage, const int repetableImageSize)
{
  if (stereoImage.empty() || 
      !repetableImageSize || 
      repetableImageSize < 1 ||
      (stereoImage.rows / 2) < repetableImageSize || 
      (stereoImage.cols / 2) < repetableImageSize) {
        return {};
  }

  cv::Mat depthMap(cv::Size(stereoImage.cols - repetableImageSize, 
                  stereoImage.rows),
                  CV_8UC1, cv::Scalar::all(0));

  int matchWidth = repetableImageSize / MATCH_WIDTH_COEF;
  for (int i = 0; i < stereoImage.rows; ++i) {
    for (int j = 0; j < stereoImage.cols - repetableImageSize - matchWidth; ++j) {

      int maxDepth = 0;
      double minimunDifference = -1.0f;
      for (int currentDepth = 0; currentDepth < repetableImageSize; ++currentDepth) {
        double currentDifference = 0;
        for (int channel = 0; channel < stereoImage.channels(); ++channel) {

          cv::Mat d;
          cv::absdiff(
              stereoImage(cv::Rect(j + repetableImageSize, i, matchWidth, 1)),
              stereoImage(cv::Rect(j + currentDepth, i, matchWidth, 1)), 
              d);

          auto sumScalar = cv::sum(d);
          for (int l = 0; l < sumScalar.channels; ++l) {
            currentDifference += sumScalar[l];
          }
        }

        if (minimunDifference > currentDifference || minimunDifference < 0) {
          maxDepth = currentDepth;
          minimunDifference = currentDifference;
        }

      }
      depthMap.at<uint8_t>(i, j) = maxDepth;
     }

      if ((i % 8) == 0) {
        std::cout << "Processing " << std::dec << i << std::endl;
      }          
  }
  return depthMap;    
}

using List3DPoints = std::vector<cv::Point3d>;
List3DPoints imageDepthTo3DPoints(const cv::Mat& depthImage) {
  List3DPoints list;
  for (int i = 0; i < depthImage.rows; ++i) {
      for (int j = 0; j < depthImage.cols; ++j) {
          auto p = depthImage.at<uint8_t>(i, j);
          if (p > 0) {
            list.push_back(cv::Point3f(i, j, p * sqrt(p)));
          }
      }
    }
  return list;
}

void keyboardViz3dHandle(const cv::viz::KeyboardEvent &w, void *t)
{
  cv::viz::Viz3d *fen = static_cast<cv::viz::Viz3d *>(t);
  std::cout << "Pressed "<< w.symbol << std::endl;
  if (w.symbol == "Escape") {
    exitFlag.store(true);
    condVariablePressKey.notify_all();
    window.close();
  } else if (w.symbol == "Left") {
    nextFileFlag = -1;
    condVariablePressKey.notify_all();
    window.close();
  } else if (w.symbol == "Right") {
    nextFileFlag = 1;
    condVariablePressKey.notify_all();
    window.close();
  }
}

int main(int argc, char *argv[])
{
  std::cout << "Start application" << std::endl;

  if (argc != 2) {
    return EXIT_FAILURE;
  }

  window.registerKeyboardCallback(keyboardViz3dHandle, &window);

  std::string path(argv[1]);
  std::cout << "Selected path: " << path << std::endl;
  std::thread threadReadFiles([&path]() {
    readFileSuccessFinished.store(false);
    DIR *dir; 
    struct dirent *diread;
    std::vector<std::string> files;

    if ((dir = opendir(path.data())) != nullptr) {
        while ((diread = readdir(dir)) != nullptr) {
            std::string file(diread->d_name);
            if (file.substr(file.find_last_of(".") + 1) == "jpg" && diread->d_type == DT_REG) {
              files.push_back(diread->d_name);
            }
        }
        closedir(dir);
    } else {
        return;
    }

    if (files.empty()) {
      return;
    }

    std::sort(files.begin(), files.end());

    for (auto i = files.cbegin(); i != files.cend(); ++i) {
      std::cout << *i << std::endl;
    }

    auto fileInterator = files.cbegin();
    while (!exitFlag.load()) {

      if (fileInterator == files.cend()) {
        fileInterator = files.cbegin();  
      }

      needLoadPathFile = path + "/" + *fileInterator; 
      std::cout << "Read file: " << needLoadPathFile << std::endl;      
      currentImage = cv::imread(needLoadPathFile);
      readFileSuccessFinished.store(true);
      condVariableReadFile.notify_all();

      std::unique_lock<std::mutex> lk(pressKeyMutex);
      condVariablePressKey.wait(lk, []{ return nextFileFlag.load() || exitFlag.load(); });
      if (!nextFileFlag.load()) {
        return;
      }

      if (nextFileFlag == -1) {
        if (fileInterator == files.cbegin()) {
          fileInterator = --files.cend();
          --fileInterator;
        } else {
          --fileInterator;
        }
      } else if (nextFileFlag == 1) {
        ++fileInterator;
      }
      nextFileFlag.store(0);
      lk.unlock();
    }
  });

  std::thread threadShowImage([]() {
    while (!exitFlag.load()) {
      std::unique_lock<std::mutex> lk(readFileMutex);
      condVariableReadFile.wait(lk, []{ return readFileSuccessFinished.load(); });

      auto sizeReapetableImage = findRepetableImageSize(currentImage);
      if (sizeReapetableImage) {
          std::cout << "Size repetable image: " << std::to_string(sizeReapetableImage) << std::endl;
          auto depthImage = reconstructionDepth(currentImage, sizeReapetableImage);
          auto p = imageDepthTo3DPoints(depthImage);

          window.showWidget("3D", cv::viz::WPaintedCloud(p));
          window.spin();
      }
      lk.unlock();
    }
  });

  threadReadFiles.join();
  threadShowImage.join();

  std::cout << "Close application" << std::endl;
  return EXIT_SUCCESS;
}