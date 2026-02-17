/********************************************************************************
  Project: Sport Video Analysis
  Author: Rajmonda Bardhi (Student ID: 2071810)
  Course: Computer Vision — University of Padova
  Instructor: Prof. Stefano Ghidoni
  Notes: Original work by the author. Built with C++17 and OpenCV on the official Virtual Lab.
         No external source code beyond standard libraries and OpenCV.
********************************************************************************/
#include "player_detection.h"

// maskGreenField — Segment the playing field using HSV color thresholding
// (Lab 3 "Mouse callback and color segmentation"; Lecture 10_3 "Thresholding Otsu").
// HSV is preferred over RGB because it separates chrominance from luminance,
// making the green detection robust to illumination changes (Lab 3, slide 4).
static cv::Mat maskGreenField(const cv::Mat &hsvFrame){
    cv::Mat greenMask, dilatedMask, erodedMask, fieldMask;

    // HSV green range — Lab 3 "Color segmentation": select hue range for the
    // dominant field color (Lecture 10_3: threshold sensitivity to illumination).
    cv::inRange(hsvFrame, cv::Scalar(40,40,40), cv::Scalar(90,255,255), greenMask);

    // Morphological dilation then erosion to fill small holes in the field mask
    // (Lecture 10_1 "Morphological operators": structuring element operations).
    cv::Mat morphKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5));
    cv::dilate(greenMask, dilatedMask, morphKernel);
    cv::erode(dilatedMask, erodedMask, morphKernel);
    cv::erode(erodedMask, erodedMask, morphKernel);
    cv::erode(erodedMask, erodedMask, morphKernel);
    cv::erode(erodedMask, erodedMask, morphKernel);

    std::vector<std::vector<cv::Point> > fieldContours;
    cv::findContours(erodedMask, fieldContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    fieldMask = cv::Mat::zeros(greenMask.size(), CV_8UC1);

    // Region-based segmentation (Lecture 10_2): keep green contours above a
    // minimum area threshold to filter noise while preserving the field shape.
    for(size_t i = 0; i < fieldContours.size(); i++){
        if(cv::contourArea(fieldContours[i]) > 1000.0)
            cv::drawContours(fieldMask, fieldContours, (int)i, cv::Scalar(255), cv::FILLED);
    }

    cv::imshow("Green Field Mask", fieldMask);
    return fieldMask;
}

// maskGreenPlayers — Isolate non-field pixels (potential players) within the
// field-masked region using color-based segmentation (Lab 3; Lecture 10_3).
// Inverts a combined mask of green + black + shadow pixels so that only
// player-colored pixels remain.
static cv::Mat maskGreenPlayers(const cv::Mat &fieldRegionBgr){
    cv::Mat hsvImage;
    cv::cvtColor(fieldRegionBgr, hsvImage, cv::COLOR_BGR2HSV);

    cv::Mat greenMask, blackMask, shadowMask, excludeMask;

    // Remove green field pixels — same HSV range as maskGreenField
    // (Lab 3 "Color segmentation": HSV-based color filtering).
    cv::inRange(hsvImage, cv::Scalar(40,40,40), cv::Scalar(90,255,255), greenMask);

    // Shadow suppression — Lecture 10_3 "Thresholding Otsu": threshold sensitivity
    // to noise and non-uniform illumination. Shadows have low Value (brightness).
    // Masking all pixels with V<50 aggressively removes shadow regions.
    cv::inRange(hsvImage, cv::Scalar(0,0,0), cv::Scalar(180,255,50), shadowMask);
    cv::inRange(hsvImage, cv::Scalar(0,0,0), cv::Scalar(10,10,10), blackMask);

    cv::bitwise_or(greenMask, blackMask, excludeMask);
    cv::bitwise_or(excludeMask, shadowMask, excludeMask);
    cv::bitwise_not(excludeMask, excludeMask);

    // Dilation to connect nearby player pixels — Lecture 10_1 "Morphological operators":
    // dilation expands foreground regions, bridging small gaps in the player silhouette.
    int dilationRadius = 5;
    cv::Mat dilationKernel = cv::getStructuringElement(
        cv::MORPH_RECT,
        cv::Size(2*dilationRadius+1, 2*dilationRadius+1),
        cv::Point(dilationRadius, dilationRadius)
    );
    cv::dilate(excludeMask, excludeMask, dilationKernel);

    cv::Mat playerVisualization;
    fieldRegionBgr.copyTo(playerVisualization, excludeMask);
    cv::imshow("Players", playerVisualization);

    return excludeMask;
}

// mergeOverlappingBoxes — Agglomerative clustering of overlapping bounding boxes
// (Lecture 11_1 "K-means", slide 14: agglomerative clustering merges clusters
// recursively based on proximity). Here overlap/containment is used as the
// similarity criterion to fuse fragmented detections into single player boxes.
static std::vector<cv::Rect> mergeOverlappingBoxes(const std::vector<cv::Rect> &inputBoxes){
    std::vector<cv::Rect> mergedBoxes;
    std::vector<bool> consumed(inputBoxes.size(), false);

    for(size_t i = 0; i < inputBoxes.size(); i++){
        if(consumed[i]) continue;
        cv::Rect currentBox = inputBoxes[i];
        bool mergeOccurred;
        do {
            mergeOccurred = false;
            for(size_t j = 0; j < inputBoxes.size(); j++){
                if(i == j || consumed[j]) continue;
                cv::Rect candidateBox = inputBoxes[j];
                bool overlaps = (currentBox & candidateBox).area() > 0
                    || currentBox.contains(candidateBox.tl())
                    || currentBox.contains(candidateBox.br())
                    || candidateBox.contains(currentBox.tl())
                    || candidateBox.contains(currentBox.br());
                if(overlaps){
                    currentBox = currentBox | candidateBox;
                    consumed[j] = true;
                    mergeOccurred = true;
                }
            }
        } while(mergeOccurred);
        mergedBoxes.push_back(currentBox);
        consumed[i] = true;
    }

    // Remove boxes fully contained inside larger boxes.
    std::vector<cv::Rect> filteredBoxes;
    for(size_t i = 0; i < mergedBoxes.size(); i++){
        bool isContained = false;
        for(size_t j = 0; j < mergedBoxes.size(); j++){
            if(i == j) continue;
            if(mergedBoxes[j].contains(mergedBoxes[i].tl()) && mergedBoxes[j].contains(mergedBoxes[i].br())){
                isContained = true;
                break;
            }
        }
        if(!isContained) filteredBoxes.push_back(mergedBoxes[i]);
    }

    return filteredBoxes;
}

// detectPlayers — Main detection pipeline combining background subtraction,
// color segmentation, and morphological refinement.
std::vector<cv::Rect> detectPlayers(const cv::Mat &frame, cv::Ptr<cv::BackgroundSubtractor> &bgSub){
    cv::Mat foregroundMask, hsvFrame, fieldMask, fieldRegionBgr, playerColorMask, combinedMask;

    // MOG2 background subtraction to extract moving foreground objects.
    bgSub->apply(frame, foregroundMask, 0.01);
    cv::cvtColor(frame, hsvFrame, cv::COLOR_BGR2HSV);

    fieldMask = maskGreenField(hsvFrame);

    fieldRegionBgr = cv::Mat::zeros(frame.size(), frame.type());
    frame.copyTo(fieldRegionBgr, fieldMask);

    playerColorMask = maskGreenPlayers(fieldRegionBgr);

    // Combine foreground motion mask with player color mask and restrict to field
    // (Lecture 10_2 "Intro to segmentation": combining multiple segmentation cues).
    cv::bitwise_and(foregroundMask, playerColorMask, combinedMask);
    cv::bitwise_and(combinedMask, fieldMask, combinedMask);

    // Morphological opening — Lecture 10_1 "Morphological operators", slide 16:
    // "Opening: erosion + dilation. Effects: Removes thin protrusions."
    // Eliminates small noise blobs and thin shadow remnants from the combined mask.
    cv::Mat openingKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5,5));
    cv::morphologyEx(combinedMask, combinedMask, cv::MORPH_OPEN, openingKernel);

    // Contour extraction and bounding box filtering — Lecture 10_2 "Intro to
    // segmentation": external contours delineate connected foreground regions.
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Rect> playerBoxes;
    cv::findContours(combinedMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for(size_t i = 0; i < contours.size(); i++){
        // Area filter: reject small noise blobs (Lecture 10_1: morphological size filtering).
        double contourArea = cv::contourArea(contours[i]);
        if(contourArea < 30) continue;

        cv::Rect boundingBox = cv::boundingRect(contours[i]);

        // Size constraints: player bounding boxes fall within typical pixel dimensions.
        if(boundingBox.width < 10 || boundingBox.height < 20 ||
           boundingBox.width > 100 || boundingBox.height > 200) continue;

        // Aspect ratio constraint — Lecture 10_2 "Intro to segmentation": region
        // properties (shape descriptors) distinguish player silhouettes from shadows.
        // Players are taller than wide; shadows are wide and flat.
        if(boundingBox.height < boundingBox.width) continue;

        playerBoxes.push_back(boundingBox);
    }

    return mergeOverlappingBoxes(playerBoxes);
}
