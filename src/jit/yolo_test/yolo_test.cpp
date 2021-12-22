#include <torch/script.h> // One-stop header.
#include <torch/torch.h>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <chrono>

at::Tensor xywh2xyxy(at::Tensor pred) {
    at::Tensor new_pred = torch::clone(pred);
    new_pred.index({"...", 0}) = pred.index({"...", 0}) - pred.index({"...", 2}) / 2.0f;
    new_pred.index({"...", 1}) = pred.index({"...", 1}) - pred.index({"...", 3}) / 2.0f;
    new_pred.index({"...", 2}) = pred.index({"...", 0}) + pred.index({"...", 2}) / 2.0f;
    new_pred.index({"...", 3}) = pred.index({"...", 1}) + pred.index({"...", 3}) / 2.0f;
    return new_pred;
}

cv::Mat GetSquareImage( const cv::Mat& img, int target_width = 500 )
{
    int width = img.cols,
       height = img.rows;

    cv::Mat square = cv::Mat::zeros( target_width, target_width, img.type() );

    int max_dim = ( width >= height ) ? width : height;
    float scale = ( ( float ) target_width ) / max_dim;
    cv::Rect roi;
    if ( width >= height )
    {
        roi.width = target_width;
        roi.x = 0;
        roi.height = height * scale;
        roi.y = ( target_width - roi.height ) / 2;
    }
    else
    {
        roi.y = 0;
        roi.height = target_width;
        roi.width = width * scale;
        roi.x = ( target_width - roi.width ) / 2;
    }

    cv::resize( img, square( roi ), roi.size() );

    return square;
}

at::Tensor preprocess(cv::Mat image, int image_size) {
    std::cout << "Image size: " << image.rows << " x " << image.cols << std::endl;
    int height = image.rows;
    int width = image.cols;
    auto newSize = cv::Size(0,0);

    // Pad image size:
    // Uncomment this to do just squared resizing:
    /*
    if(height < width) {
        // Width is the largest side:
        newSize = cv::Size( image_size, (image_size*height)/width ); // w,h
    }
    else {
        // Height is the largest side:
        newSize = cv::Size( (image_size*width)/height , image_size);
    } 


    std::cout << "Resizing to: " << newSize << std::endl;
    */

    // Comment the line below to do adjacent disabling:
    cv::Mat resized_image = GetSquareImage(image, image_size);
    cv::cvtColor(resized_image, resized_image, cv::COLOR_BGR2RGB);
    //cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    image.convertTo(image, CV_32FC3, 1.0f/255.0f);
    at::Tensor output = torch::from_blob(image.data, {1, image.cols, image.rows, 3});
    output = output.permute({0, 3, 1, 2});
    output = torch::nn::functional::interpolate(output, torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({image_size, image_size})).mode(torch::kNearest));
    return output;
}

void postprocess(at::Tensor pred, int image_size) {
    int NUM_CLASS = 1;
    int max_wh = 4096;
    int max_det = 300;
    int max_nms = 30000;
    float conf_thres = 0.25;
    float iou_thres = 0.45;
    auto predSize = pred.sizes();
    int batchSize = 1; // Forced batchsize of 1

    for(int i=0; i < batchSize; i++) {
        auto p = pred[i];
        auto conf = p.index({"...", 4});
        p = p.index({conf>conf_thres});
        if (p.sizes()[0] == 0) { break; }

        p.index( {torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(5, torch::indexing::None)} ) = p.index( {torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(5, torch::indexing::None)} )
        * p.index({ torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(4,5) });

        at::Tensor bboxes = xywh2xyxy(p.index( {torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(torch::indexing::None, 4)} ));
        
        // Assuming single class:
        //conf, j = x[:, 5:].max(1, keepdim=True)
        auto conf_j = torch::max(p.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(5, torch::indexing::None)}), 1);
        //auto j = torch::argmax(p.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(5, torch::indexing::None)}), 1);
        auto conf_bb = std::get<0>(conf_j);
        conf_bb = conf_bb.shaped<float, 3>({A*B, C, D});
        auto j = std::get<1>(conf_j );
        p = torch::cat((bboxes, conf_bb, j) 1)[conf.view(-1) > conf_thres]

        std::cout << conf_bb << j << std::endl;

        //x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
    }
}

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module module;
    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        module = torch::jit::load(argv[1]);
    }
        catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat image = cv::imread("/home/vijay/Documents/devmk4/radar-cnn/data/syn_walk/images/frame_40_40.png");

    {
        torch::NoGradGuard no_grad;
        at::Tensor input_tensor = preprocess(image, 416);
        std::cout << input_tensor.sizes() << std::endl;

        std::vector<torch::jit::IValue> inputs;
        //at::Tensor input_tensor = torch::ones({1,3,416,416});
        
        input_tensor = input_tensor.cuda();
        inputs.push_back(input_tensor);
        
        at::Tensor output = module.forward(inputs).toTensor();
        postprocess(output, 500);
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);    
    std::cout << duration.count() << std::endl;
}