/********************************************************************************
  Project: Sport Video Analysis
  Author: Rajmonda Bardhi (Student ID: 2071810)
  Course: Computer Vision — University of Padova
  Instructor: Prof. Stefano Ghidoni
  Notes: Original work by the author. Built with C++17 and OpenCV on the official Virtual Lab.
         No external source code beyond standard libraries and OpenCV.
********************************************************************************/
#ifndef PLAYER_DETECTION_H
#define PLAYER_DETECTION_H
#include <opencv2/opencv.hpp>
#include <vector>
std::vector<cv::Rect> detectPlayers(const cv::Mat &frame, cv::Ptr<cv::BackgroundSubtractor> &bgSub);
#endif
