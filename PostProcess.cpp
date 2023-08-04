#include "PostProcess.h"
#include <math.h>

torch::Tensor _nms(torch::Tensor heat, int64_t kernel = 3){
    int64_t pad = (kernel - 1) / 2;
    std::cout << "pad:" << pad << "\tkernel:" << kernel << std::endl;
    torch::Tensor hmax = at::max_pool2d(heat, {kernel, kernel}, {1, 1}, {pad, pad});
    torch::Tensor keep = (hmax == heat).toType(torch::kFloat32);
    return heat * keep;
}

torch::Tensor _gather_feat(torch::Tensor feat, torch::Tensor ind){
    int64_t dim = feat.size(2);
    ind = ind.unsqueeze(2).expand({ind.size(0), ind.size(1), dim});
    feat = feat.gather(1, ind);
    return feat;
}

void _topk(const torch::Tensor &scores,
           torch::Tensor &topk_score,
           torch::Tensor &topk_inds,
           torch::Tensor &topk_clses,
           torch::Tensor &topk_ys,
           torch::Tensor &topk_xs,
           int64_t K = 100){
	//获取特征图尺寸
    int64_t batch = scores.sizes()[0];
    int64_t cat = scores.sizes()[1];
    int64_t height = scores.sizes()[2];
    int64_t width = scores.sizes()[3];
    std::cout << "batch:" << batch << "\tcat:" << cat << "\theight:" << height << "\twidth:" << width << std::endl;
    //假设特征图大小为(1,20,128,128)　这里就是20分类的检测器

	//排序
    std::tuple<torch::Tensor, torch::Tensor> topk_score_inds =
        torch::topk(scores.view({batch, cat, -1}), K);

	//得到排序结果按默认值和假设这里tensor尺寸为(1,20,100)
    torch::Tensor topk_scores = std::get<0>(topk_score_inds);
    std::cout << "topk_scores sizes:" << topk_scores.sizes() << std::endl;
    topk_inds = std::get<1>(topk_score_inds);
    std::cout << "topk_inds sizes:" << topk_inds.sizes() << std::endl;
    
    // torch::topk得到的indexs是特征图绝对位置(这里是指拉成一维的位置)，这里除以特征图面积，得到相对偏移量
    topk_inds = topk_inds % (height * width);

	//这里得到具体x,y的偏移量
    topk_ys = (topk_inds / width).toType(torch::kInt32).toType(torch::kFloat32);
    topk_xs = (topk_inds % width).toType(torch::kInt32).toType(torch::kFloat32);
	
	//加上类别再选出topk
    std::tuple<torch::Tensor, torch::Tensor> topk_score_ind =
        torch::topk(topk_scores.view({batch, -1}), K);
        
    topk_score = std::get<0>(topk_score_ind);
    torch::Tensor topk_ind = std::get<1>(topk_score_ind);
    //这里除以Ｋ而不是除以特征图面积得到类别，是因为这个实在筛选出来的(类别数,K)的tensor上,因为topk_scores的尺寸为(1,20,100)
    topk_clses = (topk_ind / K).toType(torch::kInt32);
    //分类排序筛选出原始位置，做这步原因是这次topk的到的topk_ind不是在原始特征图上得到，需要映射回去
    topk_inds = _gather_feat(topk_inds.view({batch, -1, 1}), topk_ind).view({batch, K});
    topk_ys = _gather_feat(topk_ys.view({batch, -1, 1}), topk_ind).view({batch, K});
    topk_xs = _gather_feat(topk_xs.view({batch, -1, 1}), topk_ind).view({batch, K});
}

torch::Tensor _transpose_and_gather_feat(torch::Tensor feat, torch::Tensor ind){
    feat = feat.permute({0, 2, 3, 1}).contiguous();
    feat = feat.view({feat.size(0), -1, feat.size(3)});
    feat = _gather_feat(feat, ind);
    return feat;
}

torch::Tensor ctdet_angle_decode(torch::Tensor& hm, torch::Tensor& wh, 
torch::Tensor& angle, torch::Tensor& reg, const int K){
    int64_t batch = hm.sizes()[0];    
    auto heat = _nms(hm);    
    torch::Tensor scores,topk_inds,topk_clses,topk_ys,topk_xs;
    _topk(heat, scores, topk_inds, topk_clses, topk_ys, topk_xs);
    std::cout << "scores:" << scores[0][0] << " " << scores[0][1] << scores[0][2] << std::endl;
    std::cout << "topk_inds:" << topk_inds[0][0] << " " << topk_inds[0][1] << topk_inds[0][2] << std::endl;
    std::cout << "topk_ys:" << topk_ys[0][0]  << " " << topk_ys[0][1] << topk_ys[0][2] << std::endl;
    std::cout << "topk_ys:" << topk_xs[0][0]  << " " << topk_xs[0][1] << topk_xs[0][2] << std::endl;
    reg = _transpose_and_gather_feat(reg, topk_inds);
    reg = reg.view({batch, K, 2});
    topk_xs = topk_xs.view({batch, K, 1}) + reg.slice(2,0,1); 
    std::cout << "topk_xs dims:" << topk_xs.sizes() << std::endl;
    std::cout << "topk_xs:" << topk_xs[0][0][0] << "\t" << topk_xs[0][1][0] << "\t" << topk_xs[0][2][0] << std::endl;
    topk_ys = topk_ys.view({batch, K, 1}) + reg.slice(2,1,2);
    std::cout << "topk_ys dims:" << topk_ys.sizes() << std::endl;
    std::cout << "topk_ys:" << topk_ys[0][0][0] << "\t" << topk_ys[0][1][0] << "\t" << topk_ys[0][2][0] << std::endl;
    
    wh = _transpose_and_gather_feat(wh, topk_inds);
    wh = wh.view({batch, K, 2});
    angle = _transpose_and_gather_feat(angle, topk_inds);
    torch::Tensor clses = topk_clses.view({batch, K, 1}).toType(torch::kFloat32);
    scores = scores.view({batch, K, 1});    
    std::vector<torch::Tensor> vec_tensor = {
        topk_xs, topk_ys, wh.slice(2,0,1), wh.slice(2,1,2), angle};
    torch::Tensor bboxes = torch::cat(vec_tensor, 2);
    auto bboxes_cpu = bboxes.cpu();
    auto scores_cpu = scores.cpu();
    auto clses_cpu = clses.cpu();
    return torch::cat({bboxes_cpu, scores_cpu, clses_cpu}, 2);
}

void rotation_bboxs_to_segmentations(const torch::Tensor& rrect_tensor, cv::Point2f pts[]){
    float x = rrect_tensor[0].item().toFloat();
    float y = rrect_tensor[1].item().toFloat();
    float w = rrect_tensor[2].item().toFloat();
    float h = rrect_tensor[3].item().toFloat();
    float angle = rrect_tensor[4].item().toFloat();
    std::cout << "x:" << x << " y:" << y << " w:" << w << " h:" << h << " angle:" << angle << std::endl;
    angle = angle * 180.0f/M_PI;
    cv::RotatedRect rr = cv::RotatedRect(cv::Point2f{x +0.5f,y + 0.5f}, cv::Size(w,h), angle);    
    rr.points(pts);
    std::cout << "pts:" << pts[0] << " " << pts[1] << " " << pts[2] << " " << pts[3] << std::endl;
}

 std::vector<cv::Point2f>  affine_transform(const std::vector<cv::Point2f>& pts, const cv::Mat& trans_mat){
    cv::Mat new_pts(3, pts.size(), CV_32F);
    for (size_t i = 0; i < pts.size(); i++) {
        new_pts.at<float>(0,i) = pts[i].x;
        new_pts.at<float>(1,i) = pts[i].y;
        new_pts.at<float>(2,i) = 1.0f; 
    }
    cv::Mat trans_mat_float;
    trans_mat.convertTo(trans_mat_float, CV_32F);
    cv::Mat trans_pts = trans_mat_float * new_pts;
    //std::cout << "trans_pts:" << trans_pts.size() << std::endl;
    std::vector<cv::Point2f>  ori_pts(pts.size(), cv::Point2f(0.0f,0.0f));
    for (size_t i = 0; i < ori_pts.size(); i++) {
        ori_pts[i] = cv::Point2f{trans_pts.at<float>(0,i), trans_pts.at<float>(1,i)};
    }
    return ori_pts;
}

std::vector<CtdetResult> ctdet_angle_post_process(const torch::Tensor& dets, const cv::Mat& trans_mat, const float th){
    torch::Tensor dets_ = dets.squeeze(0).clone();        
    std::vector<CtdetResult> ret;    
    std::vector<float> confidences;
    std::vector<cv::Point2f> pts_vec;
    for(int i = 0; i < dets_.size(0); i++){           
        if (dets_[i][5].item().toFloat() > th) {
            torch::Tensor det = dets_.select(0, i);   
            std::cout << "det:" << det << std::endl;         
            cv::Point2f pts[4];
            rotation_bboxs_to_segmentations(det, pts);
            for (size_t i = 0; i < 4; i++){
                pts_vec.push_back(pts[i]);
            }
            confidences.push_back(det[5].item().toFloat());
        }
    }
    if (pts_vec.size() != 4*confidences.size()) {
        std::cerr << " pts_vec size:" << pts_vec.size() <<
        "confidences size:" << confidences.size() << " pts_vec size should be 4*confidences's size" << std::endl;
        return ret;
    }    
    //std::cout << "begin affine transform-----" << std::endl;
    if (pts_vec.size() >1){
        auto ori_pts = affine_transform(pts_vec, trans_mat);   
        for (size_t i = 0; i < ori_pts.size()/4; i++){
            CtdetResult result(ori_pts[4*i], ori_pts[4*i + 1], ori_pts[4*i + 2],ori_pts[4*i + 3], confidences[i]);
            ret.push_back(result);
        }
    }
    return ret;
}