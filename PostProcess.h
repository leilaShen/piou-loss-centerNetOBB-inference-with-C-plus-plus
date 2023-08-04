#pragma once
#include <torch/script.h> // One-stop header.
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

torch::Tensor ctdet_angle_decode(torch::Tensor& hm, torch::Tensor& wh, 
torch::Tensor& angle, torch::Tensor& reg, const int K = 100);

struct CtdetResult{
    cv::RotatedRect rrect;
    float confidence = 0.0f;
    CtdetResult(const cv::Point2f& p0, const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3, const float& c):confidence(c){
        std::vector<cv::Point2f> pts{p0,p1,p2,p3};
        rrect = cv::minAreaRect(pts);
    };

};

std::vector<CtdetResult> ctdet_angle_post_process(const torch::Tensor& dets, const cv::Mat& trans_mat, const float th = 0.4f);