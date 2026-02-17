/********************************************************************************
  Project: Sport Video Analysis
  Author: Rajmonda Bardhi (Student ID: 2071810)
  Course: Computer Vision — University of Padova
  Instructor: Prof. Stefano Ghidoni
  Notes: Original work by the author. Built with C++17 and OpenCV on the official Virtual Lab.
         No external source code beyond standard libraries and OpenCV.
********************************************************************************/
#include "classification.h"
#include <map>

static std::vector<cv::Mat> teamFeatureAnchors;
static int anchorCount=0;
static const int MAX_ANCHOR_FRAMES=10;
static bool teamAnchorsInitialized=false;
static std::map<int,std::pair<cv::Rect,int> > lastFrameBoxes;
static int nextID=0;
static const int teamsCount=2;

// avgNonGreenLab — Extract a CIELab color feature vector from player ROI,
// excluding green field pixels. CIELab is perceptually uniform, meaning
// Euclidean distance in Lab space correlates with perceived color difference
// (Lecture 11_1 "K-means", slide 7: feature vector representation using color
// information; Lab 3: HSV-based color segmentation to isolate regions).
static cv::Vec3f avgNonGreenLab(const cv::Mat &roi){
    // Focus on upper 60% of ROI — the jersey/shirt area is most discriminative
    // for team classification. Lower body (shorts, legs, feet) adds noise.
    int upperH=(int)(roi.rows*0.6);
    if(upperH<1) upperH=roi.rows;
    cv::Mat upper=roi(cv::Rect(0,0,roi.cols,upperH));
    cv::Mat hsv; cv::cvtColor(upper,hsv,cv::COLOR_BGR2HSV);
    // Mask out green field pixels and shadow pixels within the ROI
    // (Lab 3: HSV color filtering; shadows have V<50).
    cv::Mat greenM,shadowM,exclude;
    cv::inRange(hsv,cv::Scalar(40,40,40),cv::Scalar(90,255,255),greenM);
    cv::inRange(hsv,cv::Scalar(0,0,0),cv::Scalar(180,255,50),shadowM);
    cv::bitwise_or(greenM,shadowM,exclude);
    // Convert to CIELab for perceptually uniform color features.
    cv::Mat lab; cv::cvtColor(upper,lab,cv::COLOR_BGR2Lab); lab.convertTo(lab,CV_32F);
    cv::Mat r=lab.reshape(1,lab.rows*lab.cols);
    std::vector<float> Lv,Av,Bv;
    for(int i=0;i<r.rows;i++){
        if(exclude.at<uchar>(i)==0){
            cv::Vec3f p=r.at<cv::Vec3f>(i);
            Lv.push_back(p[0]); Av.push_back(p[1]); Bv.push_back(p[2]);
        }
    }
    if(Lv.empty()) return cv::Vec3f(0,0,0);
    // Use median for robustness against outlier pixels (partial occlusion, noise).
    std::sort(Lv.begin(),Lv.end()); std::sort(Av.begin(),Av.end()); std::sort(Bv.begin(),Bv.end());
    int m=(int)Lv.size()/2;
    return cv::Vec3f(Lv[m],Av[m],Bv[m]);
}

// findClosestBox — Simple nearest-neighbor tracking using Euclidean distance
// between box centers (Lecture 11_1 "K-means", slide 12: Euclidean distance
// function for comparing feature vectors — here spatial position features).
static int findClosestBox(const cv::Rect &cur,const std::map<int,std::pair<cv::Rect,int> > &last){
    int best=-1; double dmin=50.0;
    for(std::map<int,std::pair<cv::Rect,int> >::const_iterator it=last.begin();it!=last.end();++it){
        cv::Rect p=it->second.first;
        cv::Point2f c1=(cur.tl()+cur.br())*0.5f, c2=(p.tl()+p.br())*0.5f;
        double d=cv::norm(c1-c2);
        if(d<dmin){ dmin=d; best=it->first; }
    }
    return best;
}

// classifyPlayers — Assign each detected player to a team using K-means
// clustering on CIELab color features (Lecture 11_1 "K-means": partition data
// into k clusters by minimizing within-cluster sum of squares; k=2 for two teams).
// Temporal anchoring stabilizes cluster assignments across frames by maintaining
// exponential moving average of cluster centers over the first 10 frames.
std::vector<std::pair<cv::Rect,int> > classifyPlayers(const cv::Mat &frame,const std::vector<cv::Rect> &boxes){
    std::vector<cv::Vec3f> feats; feats.reserve(boxes.size());
    for(size_t i=0;i<boxes.size();i++){
        cv::Rect sb=boxes[i]&cv::Rect(0,0,frame.cols,frame.rows);
        if(sb.area()<=0){ feats.push_back(cv::Vec3f(0,0,0)); continue; }
        cv::Mat roi=frame(sb); cv::resize(roi,roi,cv::Size(32,64));
        feats.push_back(avgNonGreenLab(roi));
    }
    if(feats.empty()) return std::vector<std::pair<cv::Rect,int> >();
    cv::Mat X((int)feats.size(),3,CV_32F);
    for(int i=0;i<X.rows;i++){ X.at<float>(i,0)=feats[i][0]; X.at<float>(i,1)=feats[i][1]; X.at<float>(i,2)=feats[i][2]; }
    if(X.rows<teamsCount) return std::vector<std::pair<cv::Rect,int> >();
    cv::Mat labels,centers;
    // K-means clustering — Lecture 11_1 "K-means", slide 17: "A simple clustering
    // algorithm based on a fixed number of clusters (k)." Using k=2 for two teams,
    // KMEANS_PP_CENTERS for smart initialization, 5 attempts to avoid local minima.
    cv::kmeans(X,teamsCount,labels,cv::TermCriteria(cv::TermCriteria::EPS+cv::TermCriteria::COUNT,10,1.0),5,cv::KMEANS_PP_CENTERS,centers);
    if(anchorCount<MAX_ANCHOR_FRAMES){
        if(teamFeatureAnchors.empty()){ for(int i=0;i<centers.rows;i++) teamFeatureAnchors.push_back(centers.row(i).clone()); }
        else{
            for(int i=0;i<centers.rows;i++){
                if(teamFeatureAnchors[i].size()!=centers.row(i).size()||teamFeatureAnchors[i].type()!=centers.row(i).type())
                    teamFeatureAnchors[i]=centers.row(i).clone();
                else
                    teamFeatureAnchors[i]=teamFeatureAnchors[i]*0.9f+centers.row(i)*0.1f;
            }
        }
        anchorCount++; if(anchorCount==MAX_ANCHOR_FRAMES) teamAnchorsInitialized=true;
    }
    std::vector<int> mapLab(teamsCount,-1); std::vector<bool> used(teamsCount,false);
    for(int fixed=0;fixed<teamsCount;fixed++){
        float md=FLT_MAX; int bj=-1;
        for(int i=0;i<teamsCount;i++){
            if(used[i]) continue;
            float d=cv::norm(teamFeatureAnchors[fixed]-centers.row(i));
            if(d<md){ md=d; bj=i; }
        }
        if(bj!=-1){ mapLab[bj]=fixed; used[bj]=true; }
    }
    // Compute per-player distance to each cluster center for confidence scoring.
    std::vector<std::pair<cv::Rect,int> > out; out.reserve(boxes.size());
    std::map<int,std::pair<cv::Rect,int> > now;
    for(size_t i=0;i<boxes.size();i++){
        int raw=labels.at<int>((int)i);
        int team=mapLab[raw];
        // Confidence: ratio of distance to own cluster vs distance to other cluster.
        // High ratio means the player is close to the boundary between teams.
        float dOwn=cv::norm(X.row(i)-centers.row(raw));
        float dOther=cv::norm(X.row(i)-centers.row(1-raw));
        float confidence=(dOther>0)?(dOwn/dOther):0;
        int mid=findClosestBox(boxes[i],lastFrameBoxes);
        if(mid!=-1){
            // Only inherit previous frame's label when k-means is uncertain
            // (confidence ratio > 0.7 means clusters are close for this player).
            // When k-means is confident, trust the current color evidence.
            if(confidence>0.7 && lastFrameBoxes.at(mid).second!=team)
                team=lastFrameBoxes.at(mid).second;
            now[mid]=std::make_pair(boxes[i],team);
        }else{
            now[nextID++]=std::make_pair(boxes[i],team);
        }
        out.push_back(std::make_pair(boxes[i],team));
    }
    lastFrameBoxes=now;
    return out;
}
