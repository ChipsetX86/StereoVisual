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
constexpr float MATCH_WIDTH_COEF = 4;

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
      for (int currentDepth = 0; currentDepth < repetableImageSize / 2; ++currentDepth) {
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
  if (w.action) {
      std::cout << "you pressed "<< w.code<<" = "<<w.symbol<< " in viz window "<<fen->getWindowName()<<"\n";
  }
}

int main(int argc, char *argv[])
{
  std::cout << "Start application" << std::endl;

  cv::Mat image = cv::imread("1.jpg");

  auto sizeReapetableImage = findRepetableImageSize(image);
  if (sizeReapetableImage) {
    std::cout << "Size repetable image: " << std::to_string(sizeReapetableImage) << std::endl;
    auto depthImage = reconstructionDepth(image, sizeReapetableImage);

    auto p = imageDepthTo3DPoints(depthImage);

    cv::viz::Viz3d window("Show image");
    window.showWidget("3D", cv::viz::WPaintedCloud(p));
    window.registerKeyboardCallback(keyboardViz3dHandle, &window);
    window.spin();
  }

  std::cout << "Close application" << std::endl;
  return EXIT_SUCCESS;
}