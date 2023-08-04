#include <torch/script.h> // One-stop header.
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <memory>
#include <string>
#include <dlfcn.h>
#include "./PostProcess.h"
#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>


std::vector<char> get_the_bytes(std::string filename) {
    std::ifstream input(filename, std::ios::binary);
    std::vector<char> bytes(
        (std::istreambuf_iterator<char>(input)),
        (std::istreambuf_iterator<char>()));

    input.close();
    return bytes;
}

cv::Point2f getDir(const cv::Point2f& src_point, const float rot_rad){  
  float sn = sinf(rot_rad);  
  float cs = cosf(rot_rad);  
  cv::Point2f src_result{0.0f,0.0f};
  src_result.x = src_point.x * cs - src_point.y * sn;
  src_result.y = src_point.x * sn + src_point.y * cs;
  return src_result;
}

cv::Point2f get3rdPoint(const cv::Point2f& a, const cv::Point2f& b){
  float direct_x = a.x - b.x;
  float direct_y = a.y - b.y;
  return cv::Point2f{b.x - direct_y, b.y + direct_x};
}


cv::Mat calAffineTransform(const cv::Point& center, const int src_max_len, const cv::Size& output_size, const int inv = 0){
  int src_w = src_max_len;
  int dst_w = output_size.width;
  int dst_h = output_size.height;
  cv::Point2f src_dir = getDir(cv::Point2f{0.0f, src_w*(-0.5f)}, 0.0f);  
  cv::Point2f dst_dir{0.0f, dst_w*(-0.5f)};
  cv::Point2f src[3], dst[3];
  src[0] = cv::Point2f{center.x, center.y};
  src[1] = cv::Point2f{center.x + src_dir.x, center.y + src_dir.y};
  dst[0] = cv::Point2f{dst_w * 0.5f, dst_h * 0.5f};
  dst[1] = cv::Point2f{dst_w * 0.5f + dst_dir.x, dst_h * 0.5f + dst_dir.y};
  src[2] = get3rdPoint(src[0], src[1]);
  dst[2] = get3rdPoint(dst[0], dst[1]);  
  cv::Mat trans_mat;
  if(inv == 0){
    trans_mat = cv::getAffineTransform(src, dst);
  } else {
    trans_mat = cv::getAffineTransform(dst, src);
  }
  return trans_mat;
}

cv::Mat normalizeImg(const cv::Mat& src){
  std::vector<float> mean_value{0.773189, 0.911701, 0.857906};
  std::vector<float> std_value{0.97647, 1.137132, 1.078965};
  cv::Mat dst;
  std::vector<cv::Mat> bgr_channels(3);
  cv::split(src, bgr_channels);
  //std::vector<cv::Mat> rgb_channels(3);
  for (auto i = 0; i < bgr_channels.size(); i++) {
      bgr_channels[i].convertTo(bgr_channels[i], CV_32F);
      bgr_channels[i] = (bgr_channels[i]/255.0f - mean_value[i])/std_value[i];
  }
  cv::merge(bgr_channels, dst);
  //cv::imwrite("./img.exr", dst);
  return dst;
}

cv::Mat preprocess(const cv::Mat& img){
  int inp_width = 512;
  int inp_height = 512;
  int img_width = img.cols;
  int img_height = img.rows;
  int src_max_len = std::max(img_width, img_height);
  cv::Mat trans_mat = calAffineTransform(cv::Point{img_width/2, img_height/2}, src_max_len, cv::Size{inp_width, inp_height});
  //std::cout << "trans_mat:" << trans_mat << std::endl;
  cv::Mat dst_img;
  cv::warpAffine(img, dst_img, trans_mat, cv::Size{inp_width, inp_height});
  //cv::Mat norm_dst_img = normalizeImg(dst_img);
  //cv::imwrite("./affined_img.png", dst_img);
  //cv::Mat rgb_dst_img;
  //cv::cvtColor(norm_dst_img, rgb_dst_img, cv::COLOR_BGR2RGB);
  //std::cout << "pixel value at line 10, cols 10:" << rgb_dst_img.at<cv::Vec3f>(10,10) << std::endl;
  //return rgb_dst_img;
  //std::cout << norm_dst_img.at<cv::Vec3f>(120,120) << std::endl;
  //return norm_dst_img;
  return dst_img;
}


void testGetAffineTransform(){
  cv::Point center{968,608};
  cv::Size output_size{512,512};
  int src_max_len = 1936;
  cv::Mat trans_mat = calAffineTransform(center, src_max_len, output_size);
  std::cout << "trans mat:" << trans_mat << std::endl;
}

void testPreprocess(){
  std::string img_path = "/home/dell/leihaozhe/piou/test_folder/wuhan_small/17458_1_182499_4103175824_100010456226_RGB_2021-01-15-15-27-28-999.png";
  cv::Mat img = cv::imread(img_path, -1);
  cv::Mat dst = preprocess(img);
}

void testAffineTransform(){
  cv::Mat new_pt = (cv::Mat_<float>(3,1)<< 74.9296f , 32.60639f, 1.0f);
  cv::Mat t = (cv::Mat_<float>(2,3)  << 15.125f,   -0.0f   ,    0.f   ,
          0.0f   ,   15.125f, -360.0f);
  cv::Mat pt = t*new_pt;
  std::cout << "new_pt:" << pt << std::endl;
}

void testRotatedRect(){
  float x = 82.4433f, y = 38.1802f, w = 13.3735f, h = 13.0859f, angle = 3.27788f*180.0f/3.1415926f;
  cv::RotatedRect rr = cv::RotatedRect(cv::Point2f{x,y}, cv::Size(w,h), angle);
  cv::Point2f corners[4];
  rr.points(corners);
  cv::Point2f ori_corners[4];
  for (size_t i = 0; i < 4; i++) {
    ori_corners[i].x = corners[i].x * 4.0f;
    ori_corners[i].y = corners[i].y * 4.0f;
  }
  std::string fn = "/home/dell/shenlei/piou_inference/build/affined_img.png";
  cv::Mat img = cv::imread(fn, -1);
  for (int i = 0; i < 4; i++)
    cv::line(img, ori_corners[i], ori_corners[(i+1)%4], cv::Scalar(0,255,0), 1);
  
  cv::Point2f vertices[4] = {cv::Point2f{74.93f,32.60f}, cv::Point2f{88.18f,30.79f},cv::Point2f{89.95f,43.75f},cv::Point2f{76.71f,45.57f}};
  for (size_t i = 0; i < 4; i++) {
    vertices[i].x *= 4.0f;
    vertices[i].y *= 4.0f;
  }
  for (int i = 0; i < 4; i++)
    cv::line(img, vertices[i], vertices[(i+1)%4], cv::Scalar(255,0,0), 1);
  

  cv::imwrite("./rotated_rect.png", img);
}

void testTensor(){
  torch::Tensor a = torch::rand({1,10,7});
  std::cout << "a:" << a << std::endl;
  torch::Tensor b = a.select(1,1).squeeze(0);
  std::cout << "b:" << b << std::endl;
  std::cout << "b[2]:" <<b[2] << std::endl;
  std::cout << "b[-1]:" << b[-1] << std::endl;
  std::cout << "a:" << a << std::endl;
}

static int ind = 0;
void showResult(const cv::Mat& img, const std::vector<CtdetResult>& results){
  std::cout << "confidence:" << std::endl;
  for(const auto& result : results){
    cv::Point2f vertices[4];
    result.rrect.points(vertices);
    std::cout << result.confidence << std::endl;
    for (int i = 0; i < 4; i++)
        cv::line(img, vertices[i], vertices[(i+1)%4], cv::Scalar(0,0,255), 2);
        cv::circle(img, result.rrect.center, 2, (0,0,255),2);
        std::cout << "center:" << result.rrect.center << std::endl;
        // font = cv2.FONT_HERSHEY_SIMPLEX
        // c = (0, 0, 255)
        // txt = str(label_name)+':'+str(bbox[5])
        // cv2.putText(img, txt, (int(bbox[0]), int(bbox[1]) - 2),
        //             font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        cv::putText(img,std::to_string(result.confidence), cv::Point(result.rrect.center.x, result.rrect.center.y - 2), cv::FONT_HERSHEY_COMPLEX,
        0.5, cv::Scalar(0,0,0), 1, cv::LINE_AA);
  }
  cv::imwrite("./result_" + std::to_string(ind++) + ".png", img);
}

std::vector<std::string> getFns(const std::string& fns_file){
  //std::string path = "/home/dell/leihaozhe/piou/test_folder/wuhan/";
  std::vector<std::string> fns;
  std::ifstream file(fns_file);
  std::copy(std::istream_iterator<std::string>(file), std::istream_iterator<std::string>(), std::back_inserter(fns));
  return fns;
}



torch::Tensor mat2tensor(cv::Mat& img)
{    
    cv::cvtColor(img, img,  cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32FC3, 1.0/255.0);
    std::vector<float> mean_value{0.773189, 0.911701, 0.857906};
    std::vector<float> std_value{0.97647, 1.137132, 1.078965};
    for (int i = 0; i < 512; i ++){
      for (int j = 0; j < 512; j++){
        auto& p = img.at<cv::Vec3f>(i,j);
        p[0] = (p[0] - mean_value[0])/std_value[0];
        p[1] = (p[1] - mean_value[1])/std_value[1];
        p[1] = (p[2] - mean_value[2])/std_value[2];
      }
    }

    float *mat_data = (float*)img.data;

    torch::Tensor tensor = torch::ones({1, 512, 512 , 3});
    float *tensor_data = tensor.data_ptr<float>();

    memcpy(tensor_data, mat_data, 512 * 512 * 3 * sizeof(float));
    tensor = tensor.permute({0,3,1,2});

    return tensor;
}

void writeParamsToFile(const std::string& fn){
  cv::FileStorage fs(fn, cv::FileStorage::WRITE);
	if (!fs.isOpened()) {
		std::cerr << "open file:" << fn << " failed!" << std::endl;
    return;
	}
  fs << "model_path" << "/media/vision/Data/work/piou/src/model_test_trace_use_image.pt";
  std::string libdcn_path = "./libdcn_v2_cuda_forward_v2.so";
  fs << "libdcn_path" << libdcn_path;
  int gpu_id = 0;
  fs << "gpu_id" <<  0;
  cv::Size input_img_size{1936,1216};
  fs << "input_img_size" << input_img_size;
  cv::Size output_size{512,512};
  fs << "output_size" << output_size;
  cv::Size heatmap_size{128,128};
  fs << "heatmap_size" << heatmap_size;
  cv::Scalar mean{0.773189f, 0.911701f, 0.857906f};
  cv::Scalar std{0.97647f, 1.137132f, 1.078965f};
  fs << "mean" << mean;
  fs << "std" << std;
  fs.release();
}

int inferResult(){
  std::string model_path = "/media/vision/Data/work/piou/src/model_test_trace_use_image.pt";
  auto handle = dlopen("./libdcn_v2_cuda_forward_v2.so", RTLD_LAZY);  
  int gpu_id = 0;
  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(model_path, torch::Device(torch::DeviceType::CUDA, gpu_id));  
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    return -1;
  }

  module.eval();

  cv::Point center{968,608};
  cv::Size output_size{512,512};
  int src_max_len = 1936;
  cv::Mat trans_mat = calAffineTransform(center, src_max_len, output_size);

  std::string img_path = "/media/vision/Data/work/piou/test_folder/wuhan/";
  std::vector<std::string> fns = getFns("/media/vision/Data/work/piou/test_folder/wuhan.txt");
  std::cout << fns.size() << std::endl;
  for (size_t i = 0; i < fns.size(); i++) {
    cv::Mat img = cv::imread(img_path + fns[i], -1);
    cv::Mat dst = preprocess(img);
    //cv::Mat dst = cv::imread("../inp_image.png", -1);
    cv::Mat inp_img;
    dst.convertTo(inp_img, CV_32FC3, 1/255.0);
    cv::Scalar mean(0.773189, 0.911701, 0.857906);
    cv::subtract(inp_img, mean, inp_img);

    std::vector<cv::Mat> planes(3);
    cv::split(inp_img, planes); 
    cv::Scalar std_value(0.97647f, 1.137132f, 1.078965f);
    std::vector<cv::Mat> norm_planes(3);
    for (size_t i = 0; i < 3; i++){
      cv::divide(planes[i], std_value(i), norm_planes[i]);
    }
    cv::Mat norm_inp_img;
    cv::merge(norm_planes, norm_inp_img);
    
    std::vector<torch::jit::IValue> inputs;       
    torch::Tensor input = torch::from_blob(norm_inp_img.ptr<cv::Vec3f>(),{1,512,512,3});        
    input = input.permute({0,3,1,2}).contiguous();    
    
    
    //-------------------------load python tensor ------------------------------------------------------------
    // std::string python_tensor = "/media/vision/Data/work/piou/compare_input/single_image_for_compare.pt";
    // std::vector<char> f = get_the_bytes(python_tensor);
    // torch::IValue x = torch::pickle_load(f);
    // torch::Tensor my_tensor = x.toTensor();
    // std::cout << "python tensor size:" << my_tensor.sizes() << std::endl;
    //--------------------------------------------------------------------------------------------------------
    
    auto img_var = torch::autograd::make_variable(input, false);     
    auto img_var_cuda = img_var.to(at::kCUDA);    
   
    inputs.push_back(img_var_cuda);
    
      
    c10::IValue output = module.forward(inputs);  
    //std::cout << "output is Tuple?" << output.isTuple() << std::endl;
    auto output_tuple = output.toTuple()->elements();
    auto hm_before = output_tuple[0].toTensor().cpu();
    auto hm = hm_before.sigmoid();
    auto wh = output_tuple[1].toTensor().cpu();
    auto angle = output_tuple[2].toTensor().cpu();    
    auto reg = output_tuple[3].toTensor().cpu();
    
    //torch.cuda.synchronize();
    
    auto dets = ctdet_angle_decode(hm, wh, angle, reg);
    std::cout << "dets sizes:" << dets.sizes() << std::endl;
    std::cout << dets.slice(1,0,3) << std::endl;
    
    
    cv::Mat trans_mat_inv = calAffineTransform(center, src_max_len, cv::Size{128,128}, 1);
    auto results = ctdet_angle_post_process(dets, trans_mat_inv,0.45f);
    showResult(img, results);
  }
  
  dlclose(handle);
  std::cout << "ok\n";
  return 0;
}

int main(int argc, const char* argv[]) {   
  //writeParamsToFile("./piou.xml");   
  inferResult();
  return 0;
  
    
}