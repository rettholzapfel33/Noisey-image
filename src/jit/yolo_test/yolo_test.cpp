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

at::Tensor preprocess(cv::Mat image, int image_size) {
    std::cout << "Image size: " << image.rows << " x " << image.cols << std::endl;
    int height = image.rows;
    int width = image.cols;
    auto newSize = cv::Size(0,0);

    // Pad image size:
    // Uncomment this to do just squared resizing:
    if(height < width) {
        // Width is the largest side:
        newSize = cv::Size( image_size, (image_size*height)/width ); // w,h
    }
    else {
        // Height is the largest side:
        newSize = cv::Size( (image_size*width)/height , image_size);
    } 
    std::cout << "Resizing to: " << newSize << std::endl;

    // Comment the line below to do adjacent disabling:
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
    image.convertTo(image, CV_32FC3, 1.0f/255.0f);

    at::Tensor output = torch::from_blob(image.data, {1, image.cols, image.rows, 3});
    output = output.permute({0, 3, 1, 2});
    output = torch::nn::functional::interpolate(output, torch::nn::functional::InterpolateFuncOptions().size(std::vector<int64_t>({image_size, image_size})).mode(torch::kNearest));
    return output;
}

// TODO: Optimize converting tensor to vector
void cvtToVector2D(at::Tensor in, std::vector<cv::Rect>& out) {
    int* temp_arr = in.data_ptr<int>();
    for(int i=0; i < in.sizes()[0]; i++) {
        int temp[4] = {-1,-1,-1,-1};
        for(int j=0; j < in.sizes()[1]; j++) {
            temp[j] = *temp_arr++;
        }
        cv::Rect _tmp( cv::Point(temp[0],temp[1]) , cv::Point(temp[2], temp[3]));
        out.push_back(_tmp);
    }
}

void cvtToVector2D(at::Tensor in, std::vector<int>& out) {
    for(int i=0; i < in.sizes()[0]; i++) { out.push_back(in[i].item<int>()); }
}

void cvtToVector2D(at::Tensor in, std::vector<float>& out) {
    for(int i=0; i < in.sizes()[0]; i++) { out.push_back(in[i].item<float>()); }
}

std::vector<std::vector<float>> postprocess(at::Tensor pred, int image_size, int orig_size[]) {
    int NUM_CLASS = 1;
    int max_wh = 4096;
    int max_det = 300;
    int max_nms = 30000;
    float conf_thres = 0.25;
    float iou_thres = 0.45;
    auto predSize = pred.sizes();
    int batchSize = 1; // Forced batchsize of 1

    std::vector<cv::Rect> final_boxes;
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<int> final_idx;
    std::vector<std::vector<float>> output;

    for(int i=0; i < batchSize; i++) {
        auto p = pred[i];
        std::cout << p.sizes() << std::endl;
        exit(1);
        auto conf = p.index({"...", 4});
        p = p.index({conf>conf_thres});
        if (p.sizes()[0] == 0) { break; }

        p.index( {torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(5, torch::indexing::None)} ) = p.index( {torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(5, torch::indexing::None)} )
        * p.index({ torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(4,5) });

        at::Tensor bboxes = xywh2xyxy(p.index( {torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(torch::indexing::None, 4)} ));

        // Assuming single class:
        auto conf_j = torch::max(p.index({torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(5, torch::indexing::None)}), 1);
        auto conf_bb = std::get<0>(conf_j);
        auto j = std::get<1>(conf_j);
        conf_bb = torch::unsqueeze(conf_bb, 1);
        j = torch::unsqueeze(j, 1);
        p = torch::cat({bboxes, conf_bb, j}, 1);
        p = p.index({conf_bb.view(-1) > conf_thres});

        int n = p.sizes()[0];
        if(n > max_nms) {
            // sort by confidence:
            p = p.index( {torch::argsort( p.index({torch::indexing::Slice(torch::indexing::None), 4}), -1, true ).index({ torch::indexing::Slice(torch::indexing::None, max_nms) }) });
        }
        
        // move to cpu:
        p = p.cpu();
        auto c = p.index({ torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(5,6) }) * max_wh;
        c = c.to(torch::kInt32);
        
        auto boxes = p.index( {torch::indexing::Slice(torch::indexing::None), torch::indexing::Slice(torch::indexing::None, 4)} ) + c;
        
        std::cout << "box: " << boxes << std::endl;

        boxes /= image_size;
        boxes.index({torch::indexing::Slice(torch::indexing::None), 0}) *= orig_size[0];
        boxes.index({torch::indexing::Slice(torch::indexing::None), 2}) *= orig_size[0];
        boxes.index({torch::indexing::Slice(torch::indexing::None), 1}) *= orig_size[1];
        boxes.index({torch::indexing::Slice(torch::indexing::None), 3}) *= orig_size[1];
        boxes = boxes.to(torch::kInt32);
        
        auto scores = p.index( {torch::indexing::Slice(torch::indexing::None), 4 } );

        cvtToVector2D(c, classIds);
        cvtToVector2D(scores, confidences);
        cvtToVector2D(boxes, final_boxes);

        cv::dnn::NMSBoxes(final_boxes, confidences, 0.0f, iou_thres, final_idx);
        // limit detections:
        /*
        // DO LATER:
        if(final_idx.size()) > max_det {

        }
        */
        
        for(int j=0; j < final_idx.size(); j++) {
            int _idx = final_idx[j];
            std::vector<float> out = {boxes[_idx][0].item<float>(), boxes[_idx][1].item<float>(), boxes[_idx][2].item<float>(), boxes[_idx][3].item<float>(), classIds[_idx], confidences[_idx]};
            output.push_back(out);
        }

        /*
        bboxes /= image_size;
        bboxes.index({torch::indexing::Slice(torch::indexing::None), 0}) *= orig_size[0];
        bboxes.index({torch::indexing::Slice(torch::indexing::None), 2}) *= orig_size[0];
        bboxes.index({torch::indexing::Slice(torch::indexing::None), 1}) *= orig_size[1];
        bboxes.index({torch::indexing::Slice(torch::indexing::None), 3}) *= orig_size[1];
        */
        //return bboxes;
        return output;
    }
}

void applyBoxes(cv::Mat image, std::vector<std::vector<float>> bboxes) {
    //cv::resize(image, image, cv::Size(416,416), cv::INTER_LINEAR);
    
    //auto intBboxes = bboxes.to(torch::kInt32);
    std::cout << bboxes << std::endl;
    std::cout << image.rows << " x " << image.cols << std::endl;
    
    for(int i=0; i < bboxes.size(); i++) {
        std::vector<float> bbox = bboxes[i];
        cv::rectangle(image, cv::Point( (int)bbox[0], (int)bbox[1]), cv::Point( (int)bbox[2], (int)bbox[3]), cv::Scalar(0,0,255), 2, 8, 0 );
    }

    cv::namedWindow("test", cv::WINDOW_AUTOSIZE);
    cv::imshow("test", image);
    cv::waitKey(-1);
    cv::destroyWindow("test");
    cv::imwrite("yolov3_cpp_out.png", image);
}

int main(int argc, const char* argv[]) {
    if (argc != 3) {
        std::cerr << "usage: example-app <path-to-exported-script-module> <path-to-image>\n";
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
    //cv::Mat image = cv::imread("/home/vijay/Documents/devmk4/radar-cnn/data/syn_walk/images/frame_40_40.png");
    cv::Mat image = cv::imread(argv[2]);
    if(image.cols == 0) {
        std::cerr << "Invalid image path!\n";
        return -1;
    }

    int orig_size[2] = {image.rows, image.cols};

    {
        torch::NoGradGuard no_grad;
        at::Tensor input_tensor = preprocess(image, 416);
        std::cout << input_tensor.sizes() << std::endl;

        std::vector<torch::jit::IValue> inputs;
        //at::Tensor input_tensor = torch::ones({1,3,416,416});
        
        input_tensor = input_tensor.cuda();
        inputs.push_back(input_tensor);
        
        at::Tensor output = module.forward(inputs).toTensor();
        std::vector<std::vector<float>> boxes = postprocess(output, 416, orig_size);
        applyBoxes(image, boxes);
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);    
    std::cout << duration.count()*0.000001 << " seconds" << std::endl;
}