/********************************************************************************
  Project: Sport Video Analysis
  Author: Rajmonda Bardhi (Student ID: 2071810)
  Course: Computer Vision — University of Padova
  Instructor: Prof. Stefano Ghidoni
  Notes: Original work by the author. Built with C++17 and OpenCV on the official Virtual Lab.
         No external source code beyond standard libraries and OpenCV.
********************************************************************************/
#include "detection.h"

// maskGreenField — Segment the playing field using HSV color thresholding
// (Lab 3 "Mouse callback and color segmentation"; Lecture 10_3 "Thresholding Otsu").
// HSV is preferred over RGB because it separates chrominance from luminance,
// making the green detection robust to illumination changes (Lab 3, slide 4).
static cv::Mat maskGreenField(const cv::Mat &hsv){
    cv::Mat mask,dilated,eroded,out;
    // HSV green range — Lab 3 "Color segmentation": select hue range for the
    // dominant field color (Lecture 10_3: threshold sensitivity to illumination).
    cv::inRange(hsv,cv::Scalar(40,40,40),cv::Scalar(90,255,255),mask);
    // Morphological dilation then erosion to fill small holes in the field mask
    // (Lecture 10_1 "Morphological operators": structuring element operations).
    cv::Mat k=cv::getStructuringElement(cv::MORPH_RECT,cv::Size(5,5));
    cv::dilate(mask,dilated,k);
    cv::erode(dilated,eroded,k);
    cv::erode(eroded,eroded,k);
    cv::erode(eroded,eroded,k);
    cv::erode(eroded,eroded,k);
    std::vector<std::vector<cv::Point> > cts;
    cv::findContours(eroded,cts,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
    out=cv::Mat::zeros(mask.size(),CV_8UC1);
    // Region-based segmentation (Lecture 10_2): keep green contours above a
    // minimum area threshold to filter noise while preserving the field shape.
    for(size_t i=0;i<cts.size();i++){
        if(cv::contourArea(cts[i])>1000.0) cv::drawContours(out,cts,(int)i,cv::Scalar(255),cv::FILLED);
    }
    cv::imshow("Green Field Mask",out);
    return out;
}

// maskGreenPlayers — Isolate non-field pixels (potential players) within the
// field-masked region using color-based segmentation (Lab 3; Lecture 10_3).
// Inverts a combined mask of green + black + shadow pixels so that only
// player-colored pixels remain.
static cv::Mat maskGreenPlayers(const cv::Mat &hsvFieldMaskedBgr){
    cv::Mat hsv; cv::cvtColor(hsvFieldMaskedBgr,hsv,cv::COLOR_BGR2HSV);
    cv::Mat greenMask,blackMask,mask;
    // Remove green field pixels — same HSV range as maskGreenField
    // (Lab 3 "Color segmentation": HSV-based color filtering).
    cv::inRange(hsv,cv::Scalar(40,40,40),cv::Scalar(90,255,255),greenMask);
    // Shadow suppression — Lecture 10_3 "Thresholding Otsu": threshold sensitivity
    // to noise and non-uniform illumination. Shadows have low Value (brightness).
    // Masking all pixels with V<50 aggressively removes shadow regions.
    cv::Mat shadowMask;
    cv::inRange(hsv,cv::Scalar(0,0,0),cv::Scalar(180,255,50),shadowMask);
    cv::inRange(hsv,cv::Scalar(0,0,0),cv::Scalar(10,10,10),blackMask);
    cv::bitwise_or(greenMask,blackMask,mask);
    cv::bitwise_or(mask,shadowMask,mask);
    cv::bitwise_not(mask,mask);
    // Dilation to connect nearby player pixels — Lecture 10_1 "Morphological operators":
    // dilation expands foreground regions, bridging small gaps in the player silhouette.
    int d=5;
    cv::Mat elem=cv::getStructuringElement(cv::MORPH_RECT,cv::Size(2*d+1,2*d+1),cv::Point(d,d));
    cv::dilate(mask,mask,elem);
    cv::Mat result;
    hsvFieldMaskedBgr.copyTo(result,mask);
    cv::imshow("Players",result);
    return mask;
}

// mergeBoxes — Agglomerative clustering of overlapping bounding boxes
// (Lecture 11_1 "K-means", slide 14: agglomerative clustering merges clusters
// recursively based on proximity). Here overlap/containment is used as the
// similarity criterion to fuse fragmented detections into single player boxes.
static std::vector<cv::Rect> mergeBoxes(const std::vector<cv::Rect> &inputBoxes){
    std::vector<cv::Rect> merged; std::vector<bool> used(inputBoxes.size(),false);
    for(size_t i=0;i<inputBoxes.size();i++){
        if(used[i]) continue;
        cv::Rect cur=inputBoxes[i]; bool changed;
        do{
            changed=false;
            for(size_t j=0;j<inputBoxes.size();j++){
                if(i==j||used[j]) continue;
                cv::Rect o=inputBoxes[j];
                if((cur&o).area()>0||cur.contains(o.tl())||cur.contains(o.br())||o.contains(cur.tl())||o.contains(cur.br())){
                    cur=cur|o; used[j]=true; changed=true;
                }
            }
        }while(changed);
        merged.push_back(cur); used[i]=true;
    }
    std::vector<cv::Rect> cleaned;
    for(size_t i=0;i<merged.size();i++){
        bool inside=false;
        for(size_t j=0;j<merged.size();j++){
            if(i==j) continue;
            if(merged[j].contains(merged[i].tl())&&merged[j].contains(merged[i].br())){ inside=true; break; }
        }
        if(!inside) cleaned.push_back(merged[i]);
    }
    return cleaned;
}

// detectPlayers — Main detection pipeline combining background subtraction,
// color segmentation, and morphological refinement.
std::vector<cv::Rect> detectPlayers(const cv::Mat &frame, cv::Ptr<cv::BackgroundSubtractor> &bgSub){
    cv::Mat fgMask,hsv,fieldMask,fieldMaskedBgr,playersMask,combined;
    // MOG2 background subtraction to extract moving foreground objects.
    bgSub->apply(frame,fgMask,0.01);
    cv::cvtColor(frame,hsv,cv::COLOR_BGR2HSV);
    fieldMask=maskGreenField(hsv);
    fieldMaskedBgr=cv::Mat::zeros(frame.size(),frame.type());
    frame.copyTo(fieldMaskedBgr,fieldMask);
    playersMask=maskGreenPlayers(fieldMaskedBgr);
    // Combine foreground motion mask with player color mask and restrict to field
    // (Lecture 10_2 "Intro to segmentation": combining multiple segmentation cues).
    cv::bitwise_and(fgMask,playersMask,combined);
    cv::bitwise_and(combined,fieldMask,combined);
    // Morphological opening — Lecture 10_1 "Morphological operators", slide 16:
    // "Opening: erosion + dilation. Effects: Removes thin protrusions."
    // Eliminates small noise blobs and thin shadow remnants from the combined mask.
    cv::Mat openKernel=cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(5,5));
    cv::morphologyEx(combined,combined,cv::MORPH_OPEN,openKernel);
    // Contour extraction and bounding box filtering — Lecture 10_2 "Intro to
    // segmentation": external contours delineate connected foreground regions.
    std::vector<std::vector<cv::Point> > contours; std::vector<cv::Rect> boxes;
    cv::findContours(combined,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);
    for(size_t i=0;i<contours.size();i++){
        // Area filter: reject small noise blobs (Lecture 10_1: morphological size filtering).
        double area=cv::contourArea(contours[i]); if(area<30) continue;
        cv::Rect b=cv::boundingRect(contours[i]);
        // Size constraints: player bounding boxes fall within typical pixel dimensions.
        if(b.width<10||b.height<20||b.width>100||b.height>200) continue;
        // Aspect ratio constraint — Lecture 10_2 "Intro to segmentation": region
        // properties (shape descriptors) distinguish player silhouettes from shadows.
        // Players are taller than wide; shadows are wide and flat.
        if(b.height<b.width) continue;
        boxes.push_back(b);
    }
    return mergeBoxes(boxes);
}
