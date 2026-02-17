/********************************************************************************
  Project: Sport Video Analysis
  Author: Rajmonda Bardhi (Student ID: 2071810)
  Course: Computer Vision — University of Padova
  Instructor: Prof. Stefano Ghidoni
  Notes: Original work by the author. Built with C++17 and OpenCV on the official Virtual Lab.
         No external source code beyond standard libraries and OpenCV.
********************************************************************************/
#include <opencv2/opencv.hpp>
#include <fstream>
#include <vector>
#include <iostream>
#include "player_detection.h"
#include "team_classification.h"
#include "player_heatmap.h"

int main(int argc, char **argv){
    if(argc < 2){
        std::cerr << "Usage: " << argv[0] << " <video_file>\n";
        return -1;
    }

    cv::VideoCapture videoCapture(argv[1]);
    if(!videoCapture.isOpened()){
        std::cerr << "Error: could not open " << argv[1] << "\n";
        return -1;
    }

    std::ofstream detectionCsv("ours.csv");
    detectionCsv << "frame,x1,y1,x2,y2,team\n";

    // MOG2 background subtraction — models each pixel as a Mixture of Gaussians
    // to separate moving foreground (players) from static background (field).
    // Lecture 11_2 "Density estimation": Gaussian Mixture Models for density
    // estimation; history=500 frames, varThreshold=16, detectShadows=false.
    cv::Ptr<cv::BackgroundSubtractor> bgSubtractor = cv::createBackgroundSubtractorMOG2(500, 16, false);

    double fps = videoCapture.get(cv::CAP_PROP_FPS);
    int frameDelay = fps > 0 ? (int)(1000.0 / fps) : 30;

    cv::Mat frame;
    int frameIndex = 0;
    Heatmap heatmap;

    cv::namedWindow("Football Player Detection", cv::WINDOW_NORMAL);
    cv::namedWindow("Green Field Mask", cv::WINDOW_NORMAL);
    cv::namedWindow("Players", cv::WINDOW_NORMAL);
    cv::resizeWindow("Football Player Detection", 1280, 720);
    cv::resizeWindow("Green Field Mask", 1280, 720);
    cv::resizeWindow("Players", 1280, 720);

    // Team A = red, Team B = blue, Unknown = green (BGR format).
    std::vector<cv::Scalar> teamDrawColors;
    teamDrawColors.push_back(cv::Scalar(0, 0, 255));
    teamDrawColors.push_back(cv::Scalar(255, 0, 0));
    teamDrawColors.push_back(cv::Scalar(0, 255, 0));

    while(videoCapture.read(frame)){
        std::vector<cv::Rect> playerBoxes = detectPlayers(frame, bgSubtractor);
        std::vector<std::pair<cv::Rect,int> > classifiedPlayers = classifyPlayers(frame, playerBoxes);

        // Write detection results to CSV.
        for(size_t i = 0; i < classifiedPlayers.size(); i++){
            cv::Rect box = classifiedPlayers[i].first;
            int teamLabel = classifiedPlayers[i].second;
            detectionCsv << frameIndex << ","
                         << box.x << "," << box.y << ","
                         << (box.x + box.width) << "," << (box.y + box.height) << ","
                         << teamLabel << "\n";
        }

        // Draw bounding boxes and team labels on the frame.
        for(size_t i = 0; i < classifiedPlayers.size(); i++){
            cv::Rect box = classifiedPlayers[i].first;
            int teamLabel = classifiedPlayers[i].second;
            int colorIndex = (teamLabel == 0 || teamLabel == 1) ? teamLabel : 2;

            cv::rectangle(frame, box, teamDrawColors[colorIndex], 2);

            std::string labelText = (teamLabel == 0) ? "Team A" :
                                    (teamLabel == 1) ? "Team B" : "Unknown";
            cv::putText(frame, labelText, box.tl() + cv::Point(0, -5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, teamDrawColors[colorIndex], 1);
        }

        heatmap.update(frame, classifiedPlayers);
        frameIndex++;

        cv::imshow("Football Player Detection", frame);
        char key = (char)cv::waitKey(frameDelay);
        if(key == 27 || key == 'q') break;
    }

    heatmap.saveAndShow();
    cv::waitKey(0);

    detectionCsv.close();
    videoCapture.release();
    cv::destroyAllWindows();
    return 0;
}
