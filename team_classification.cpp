/********************************************************************************
  Project: Sport Video Analysis
  Author: Rajmonda Bardhi (Student ID: 2071810)
  Course: Computer Vision — University of Padova
  Instructor: Prof. Stefano Ghidoni
  Notes: Original work by the author. Built with C++17 and OpenCV on the official Virtual Lab.
         No external source code beyond standard libraries and OpenCV.
********************************************************************************/
#include "team_classification.h"
#include <map>

static std::vector<cv::Mat> teamFeatureAnchors;
static int anchorFrameCount = 0;
static const int MAX_ANCHOR_FRAMES = 10;
static bool teamAnchorsInitialized = false;
static std::map<int, std::pair<cv::Rect,int> > previousFrameBoxes;
static int nextTrackingID = 0;
static const int NUM_TEAMS = 2;

// extractJerseyColorFeature — Extract a CIELab color feature vector from the
// upper body (jersey) region of a player ROI, excluding green field pixels and
// shadow pixels. CIELab is perceptually uniform, meaning Euclidean distance in
// Lab space correlates with perceived color difference (Lecture 11_1 "K-means",
// slide 7: feature vector representation using color information; Lab 3:
// HSV-based color segmentation to isolate regions).
static cv::Vec3f extractJerseyColorFeature(const cv::Mat &playerRoi){
    // Focus on upper 60% of ROI — the jersey/shirt area is most discriminative
    // for team classification. Lower body (shorts, legs, feet) adds noise.
    int jerseyHeight = (int)(playerRoi.rows * 0.6);
    if(jerseyHeight < 1) jerseyHeight = playerRoi.rows;
    cv::Mat jerseyRegion = playerRoi(cv::Rect(0, 0, playerRoi.cols, jerseyHeight));

    cv::Mat hsvJersey;
    cv::cvtColor(jerseyRegion, hsvJersey, cv::COLOR_BGR2HSV);

    // Mask out green field pixels and shadow pixels within the ROI
    // (Lab 3: HSV color filtering; shadows have V<50).
    cv::Mat greenMask, shadowMask, excludeMask;
    cv::inRange(hsvJersey, cv::Scalar(40,40,40), cv::Scalar(90,255,255), greenMask);
    cv::inRange(hsvJersey, cv::Scalar(0,0,0), cv::Scalar(180,255,50), shadowMask);
    cv::bitwise_or(greenMask, shadowMask, excludeMask);

    // Convert to CIELab for perceptually uniform color features.
    cv::Mat labImage;
    cv::cvtColor(jerseyRegion, labImage, cv::COLOR_BGR2Lab);
    labImage.convertTo(labImage, CV_32F);
    cv::Mat labPixels = labImage.reshape(1, labImage.rows * labImage.cols);

    std::vector<float> lightness, channelA, channelB;
    for(int i = 0; i < labPixels.rows; i++){
        if(excludeMask.at<uchar>(i) == 0){
            cv::Vec3f pixel = labPixels.at<cv::Vec3f>(i);
            lightness.push_back(pixel[0]);
            channelA.push_back(pixel[1]);
            channelB.push_back(pixel[2]);
        }
    }

    if(lightness.empty()) return cv::Vec3f(0, 0, 0);

    // Use median for robustness against outlier pixels (partial occlusion, noise).
    std::sort(lightness.begin(), lightness.end());
    std::sort(channelA.begin(), channelA.end());
    std::sort(channelB.begin(), channelB.end());
    int medianIdx = (int)lightness.size() / 2;

    return cv::Vec3f(lightness[medianIdx], channelA[medianIdx], channelB[medianIdx]);
}

// findClosestTrackedPlayer — Simple nearest-neighbor tracking using Euclidean
// distance between box centers (Lecture 11_1 "K-means", slide 12: Euclidean
// distance function for comparing feature vectors — here spatial position features).
static int findClosestTrackedPlayer(const cv::Rect &currentBox,
                                     const std::map<int, std::pair<cv::Rect,int> > &trackedPlayers){
    int closestID = -1;
    double minDistance = 50.0;

    for(std::map<int, std::pair<cv::Rect,int> >::const_iterator it = trackedPlayers.begin();
        it != trackedPlayers.end(); ++it){
        cv::Rect previousBox = it->second.first;
        cv::Point2f currentCenter = (currentBox.tl() + currentBox.br()) * 0.5f;
        cv::Point2f previousCenter = (previousBox.tl() + previousBox.br()) * 0.5f;
        double distance = cv::norm(currentCenter - previousCenter);
        if(distance < minDistance){
            minDistance = distance;
            closestID = it->first;
        }
    }

    return closestID;
}

// classifyPlayers — Assign each detected player to a team using K-means
// clustering on CIELab color features (Lecture 11_1 "K-means": partition data
// into k clusters by minimizing within-cluster sum of squares; k=2 for two teams).
// Temporal anchoring stabilizes cluster assignments across frames by maintaining
// exponential moving average of cluster centers over the first 10 frames.
std::vector<std::pair<cv::Rect,int> > classifyPlayers(const cv::Mat &frame, const std::vector<cv::Rect> &boxes){
    // Extract color features for each detected player.
    std::vector<cv::Vec3f> playerFeatures;
    playerFeatures.reserve(boxes.size());

    for(size_t i = 0; i < boxes.size(); i++){
        cv::Rect safeBox = boxes[i] & cv::Rect(0, 0, frame.cols, frame.rows);
        if(safeBox.area() <= 0){
            playerFeatures.push_back(cv::Vec3f(0, 0, 0));
            continue;
        }
        cv::Mat playerRoi = frame(safeBox);
        cv::resize(playerRoi, playerRoi, cv::Size(32, 64));
        playerFeatures.push_back(extractJerseyColorFeature(playerRoi));
    }

    if(playerFeatures.empty()) return std::vector<std::pair<cv::Rect,int> >();

    // Build feature matrix for k-means input.
    cv::Mat featureMatrix((int)playerFeatures.size(), 3, CV_32F);
    for(int i = 0; i < featureMatrix.rows; i++){
        featureMatrix.at<float>(i, 0) = playerFeatures[i][0];
        featureMatrix.at<float>(i, 1) = playerFeatures[i][1];
        featureMatrix.at<float>(i, 2) = playerFeatures[i][2];
    }

    if(featureMatrix.rows < NUM_TEAMS) return std::vector<std::pair<cv::Rect,int> >();

    cv::Mat clusterLabels, clusterCenters;

    // K-means clustering — Lecture 11_1 "K-means", slide 17: "A simple clustering
    // algorithm based on a fixed number of clusters (k)." Using k=2 for two teams,
    // KMEANS_PP_CENTERS for smart initialization, 5 attempts to avoid local minima.
    cv::kmeans(featureMatrix, NUM_TEAMS, clusterLabels,
               cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
               5, cv::KMEANS_PP_CENTERS, clusterCenters);

    // Update temporal anchors with exponential moving average over the first frames.
    if(anchorFrameCount < MAX_ANCHOR_FRAMES){
        if(teamFeatureAnchors.empty()){
            for(int i = 0; i < clusterCenters.rows; i++)
                teamFeatureAnchors.push_back(clusterCenters.row(i).clone());
        } else {
            for(int i = 0; i < clusterCenters.rows; i++){
                if(teamFeatureAnchors[i].size() != clusterCenters.row(i).size() ||
                   teamFeatureAnchors[i].type() != clusterCenters.row(i).type())
                    teamFeatureAnchors[i] = clusterCenters.row(i).clone();
                else
                    teamFeatureAnchors[i] = teamFeatureAnchors[i] * 0.9f + clusterCenters.row(i) * 0.1f;
            }
        }
        anchorFrameCount++;
        if(anchorFrameCount == MAX_ANCHOR_FRAMES) teamAnchorsInitialized = true;
    }

    // Map k-means cluster indices to stable team IDs using anchor similarity.
    std::vector<int> clusterToTeamMap(NUM_TEAMS, -1);
    std::vector<bool> teamAssigned(NUM_TEAMS, false);

    for(int anchorIdx = 0; anchorIdx < NUM_TEAMS; anchorIdx++){
        float minDist = FLT_MAX;
        int bestCluster = -1;
        for(int clusterIdx = 0; clusterIdx < NUM_TEAMS; clusterIdx++){
            if(teamAssigned[clusterIdx]) continue;
            float dist = cv::norm(teamFeatureAnchors[anchorIdx] - clusterCenters.row(clusterIdx));
            if(dist < minDist){
                minDist = dist;
                bestCluster = clusterIdx;
            }
        }
        if(bestCluster != -1){
            clusterToTeamMap[bestCluster] = anchorIdx;
            teamAssigned[bestCluster] = true;
        }
    }

    // Assign team labels with confidence-based temporal smoothing.
    std::vector<std::pair<cv::Rect,int> > classifiedPlayers;
    classifiedPlayers.reserve(boxes.size());
    std::map<int, std::pair<cv::Rect,int> > currentFrameBoxes;

    for(size_t i = 0; i < boxes.size(); i++){
        int rawCluster = clusterLabels.at<int>((int)i);
        int teamLabel = clusterToTeamMap[rawCluster];

        // Confidence: ratio of distance to own cluster vs distance to other cluster.
        // High ratio means the player is close to the boundary between teams.
        float distToOwnCluster = cv::norm(featureMatrix.row(i) - clusterCenters.row(rawCluster));
        float distToOtherCluster = cv::norm(featureMatrix.row(i) - clusterCenters.row(1 - rawCluster));
        float confidenceRatio = (distToOtherCluster > 0) ? (distToOwnCluster / distToOtherCluster) : 0;

        int matchedTrackID = findClosestTrackedPlayer(boxes[i], previousFrameBoxes);

        if(matchedTrackID != -1){
            // Only inherit previous frame's label when k-means is uncertain
            // (confidence ratio > 0.7 means clusters are close for this player).
            // When k-means is confident, trust the current color evidence.
            if(confidenceRatio > 0.7 && previousFrameBoxes.at(matchedTrackID).second != teamLabel)
                teamLabel = previousFrameBoxes.at(matchedTrackID).second;
            currentFrameBoxes[matchedTrackID] = std::make_pair(boxes[i], teamLabel);
        } else {
            currentFrameBoxes[nextTrackingID++] = std::make_pair(boxes[i], teamLabel);
        }

        classifiedPlayers.push_back(std::make_pair(boxes[i], teamLabel));
    }

    previousFrameBoxes = currentFrameBoxes;
    return classifiedPlayers;
}
