#include <iostream>
#include <stdlib.h>
#include <stdint.h>
#include <limits>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/viz.hpp>

constexpr int MIN_MASK_REPEAT_COUNT = 5;
constexpr int MAX_MASK_REPEAT_COUNT = 40;
constexpr float BEST_DIFFERENCE_COEF = 0.3f;
constexpr int MATCH_WIDTH = 20;

int findRepetableImageSize(const cv::Mat& stereoImage)
{
  if (stereoImage.empty()) {
    return 0;
  }

  double minDifferenceBetweenPixels = 0;
  int betterRepeatImageSize = 0;

  for (int i = stereoImage.cols / MAX_MASK_REPEAT_COUNT; 
           i < stereoImage.cols / MIN_MASK_REPEAT_COUNT; ++i) {            
    double currentDifference = 0;

    for (int j = 0; j < stereoImage.rows; ++j) {
      for (int k = 0; k < stereoImage.cols - i; ++k) {

        auto countChannels = stereoImage.channels();
        for (int channel = 0; channel < countChannels; ++channel) {
          currentDifference += 
            std::abs(stereoImage.at<double>(j, k * countChannels + channel) -
                     stereoImage.at<double>(j, (k + i) * countChannels + channel));
        }
      }
    }

    if (currentDifference < minDifferenceBetweenPixels && 
        (!betterRepeatImageSize || 
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

  for (int i = 0; i < stereoImage.rows; ++i) {
    for (int j = 0; j < stereoImage.cols - repetableImageSize - MATCH_WIDTH; ++j) {
      
      cv::Rect zoneAfterOffset(j + repetableImageSize, i, MATCH_WIDTH, 1);
 
      int betterDepth = 0;
      double minimunDifference = -1.0f;
      for (int currentDepth = 0; currentDepth < repetableImageSize / 2; ++currentDepth) {
        double currentDifference = 0;
        for (int channel = 0; channel < stereoImage.channels(); ++channel) {

          cv::Mat d;
          cv::absdiff(
              stereoImage(cv::Rect(j + repetableImageSize, i, MATCH_WIDTH, 1)),
              stereoImage(cv::Rect(j + currentDepth, i, MATCH_WIDTH, 1)), d);

          auto sumScalar = cv::sum(d);
          for (int l = 0; l < sumScalar.channels; ++l) {
            currentDifference += sumScalar[l];
          }
        }

        if (minimunDifference > currentDifference) {
          betterDepth = currentDepth;
          minimunDifference = currentDifference;
        }
      }
      depthMap.at<uint8_t>(i, j) = betterDepth;
    }          
  }
  return depthMap;    
}


int main(int argc, char *argv[])
{
  std::cout << "Start application" << std::endl;

  cv::viz::Viz3d window("Show image");
  cv::Mat image = cv::imread("1.jpg");

  auto sizeReapetableImage = findRepetableImageSize(image);
  if (sizeReapetableImage) {
    auto depthImage = reconstructionDepth(image, sizeReapetableImage);
    window.showWidget("3D", cv::viz::WPaintedCloud(image));
    window.spin();
  }

  std::cout << "Close application" << std::endl;
  return EXIT_SUCCESS;
}