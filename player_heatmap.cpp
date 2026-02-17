/********************************************************************************
  Project: Sport Video Analysis
  Author: Rajmonda Bardhi (Student ID: 2071810)
  Course: Computer Vision — University of Padova
  Instructor: Prof. Stefano Ghidoni
  Notes: Original work by the author. Built with C++17 and OpenCV on the official Virtual Lab.
         No external source code beyond standard libraries and OpenCV.
********************************************************************************/
#include "player_heatmap.h"

Heatmap::Heatmap(){
    // Team A = red, Team B = blue, Unknown = green (BGR format).
    colors.push_back(cv::Scalar(0, 0, 255));
    colors.push_back(cv::Scalar(255, 0, 0));
    colors.push_back(cv::Scalar(0, 255, 0));
}

// update — Accumulate team-colored circles at each detection center into a
// floating-point image. This builds a spatial density map of player positions
// (Lecture 11_2 "Density estimation": estimating the underlying probability
// density of player locations from discrete observations).
void Heatmap::update(const cv::Mat &frame, const std::vector<std::pair<cv::Rect,int> > &classifiedPlayers){
    if(accum.empty()){
        accum = cv::Mat::zeros(frame.size(), CV_32FC3);
        first = frame.clone();
    }

    for(size_t i = 0; i < classifiedPlayers.size(); i++){
        cv::Mat detectionLayer = cv::Mat::zeros(frame.size(), CV_32FC3);

        int teamIndex = classifiedPlayers[i].second;
        if(teamIndex < 0 || teamIndex >= (int)colors.size())
            teamIndex = 2; // Unknown -> green

        cv::Point playerCenter = (classifiedPlayers[i].first.tl() + classifiedPlayers[i].first.br()) * 0.5;
        cv::circle(detectionLayer, playerCenter, 20, colors[teamIndex], -1, cv::LINE_AA);

        accum += detectionLayer;
    }
}

// saveAndShow — Smooth the accumulated heatmap with a Gaussian kernel and
// overlay it on the first frame for visualization.
void Heatmap::saveAndShow(){
    if(accum.empty()) return;

    cv::Mat blurredHeatmap, heatmapImage, overlayImage;

    // Gaussian smoothing — Lecture 06_1 "Spatial filtering", Lecture 06_2
    // "Linear filters": Gaussian kernel produces smooth, isotropic blurring
    // to turn discrete detection points into a continuous density visualization.
    cv::GaussianBlur(accum, blurredHeatmap, cv::Size(0, 0), 15);

    // Intensity normalization — Lecture 05_1 "Histogram equalization": mapping
    // pixel values to the full [0,255] range to maximize visual contrast.
    cv::normalize(blurredHeatmap, blurredHeatmap, 0, 255, cv::NORM_MINMAX);
    blurredHeatmap.convertTo(heatmapImage, CV_8UC3);

    cv::addWeighted(first, 0.5, heatmapImage, 0.5, 0, overlayImage);

    cv::namedWindow("Combined Heatmap", cv::WINDOW_NORMAL);
    cv::namedWindow("Heatmap Overlay", cv::WINDOW_NORMAL);
    cv::resizeWindow("Combined Heatmap", 1280, 720);
    cv::resizeWindow("Heatmap Overlay", 1280, 720);
    cv::imshow("Combined Heatmap", heatmapImage);
    cv::imshow("Heatmap Overlay", overlayImage);
    cv::imwrite("combined_heatmap.png", heatmapImage);
    cv::imwrite("heatmap_overlay.png", overlayImage);
}
