#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// Configuration matches the Python export
const int INPUT_WIDTH = 192;
const int INPUT_HEIGHT = 256;
const int CHANNELS = 3;
const int BATCH_SIZE = 1;
// ImageNet mean and std
const std::vector<float> MEAN = {0.485f, 0.456f, 0.406f};
const std::vector<float> STD = {0.229f, 0.224f, 0.225f};

// COCO Keypoint names for visualization
const std::vector<std::string> KEYPOINT_NAMES = {
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear", 
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow", 
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip", 
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
};

// Skeleton connections for drawing
const std::vector<std::pair<int, int>> SKELETON = {
    {0, 1}, {0, 2}, {1, 3}, {2, 4}, {5, 6}, {5, 7}, {7, 9}, 
    {6, 8}, {8, 10}, {5, 11}, {6, 12}, {11, 12}, {11, 13}, 
    {13, 15}, {12, 14}, {14, 16}
};

void preprocess(const cv::Mat& img, std::vector<float>& input_tensor_values) {
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));

    // Convert BGR to RGB
    cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

    // Convert to float and normalize
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    // Split channels for CHW layout
    std::vector<cv::Mat> channels(3);
    cv::split(resized, channels);

    for (int c = 0; c < 3; ++c) {
        channels[c] = (channels[c] - MEAN[c]) / STD[c];
    }

    // Flatten to vector
    for (int c = 0; c < 3; ++c) {
        std::memcpy(input_tensor_values.data() + c * INPUT_WIDTH * INPUT_HEIGHT, 
                    channels[c].data, 
                    INPUT_WIDTH * INPUT_HEIGHT * sizeof(float));
    }
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <path_to_onnx_model> <path_to_image>" << std::endl;
        return -1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];

    // 1. Load Image
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "Error: Could not read image at " << image_path << std::endl;
        return -1;
    }

    // 2. Prepare Input Tensor
    size_t input_tensor_size = BATCH_SIZE * CHANNELS * INPUT_HEIGHT * INPUT_WIDTH;
    std::vector<float> input_tensor_values(input_tensor_size);
    preprocess(img, input_tensor_values);

    // 3. Setup ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ViTPoseInference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Try to enable CUDA
    try {
        OrtCUDAProviderOptions cuda_options;
        cuda_options.device_id = 0;
        session_options.AppendExecutionProvider_CUDA(cuda_options);
        std::cout << "CUDA Execution Provider enabled." << std::endl;
    } catch (const std::exception& e) {
        std::cout << "CUDA Execution Provider not available, using CPU." << std::endl;
    }

    Ort::Session session(env, model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;

    // Get input/output names
    Ort::AllocatedStringPtr input_name = session.GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr output_name = session.GetOutputNameAllocated(0, allocator);
    std::vector<const char*> input_names = {input_name.get()};
    std::vector<const char*> output_names = {output_name.get()};

    std::vector<int64_t> input_shape = {BATCH_SIZE, CHANNELS, INPUT_HEIGHT, INPUT_WIDTH};
    
    // Create Input Tensor
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_size, input_shape.data(), input_shape.size());

    // 4. Run Inference
    std::cout << "Running inference..." << std::endl;
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, 
        input_names.data(), &input_tensor, 1, 
        output_names.data(), 1);

    // 5. Process Output
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto output_tensor_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> output_shape = output_tensor_info.GetShape();

    // Output shape should be [1, 17, 64, 48] (Batch, Joints, H, W)
    int num_joints = output_shape[1];
    int heatmap_h = output_shape[2];
    int heatmap_w = output_shape[3];

    std::cout << "Output shape: [" << output_shape[0] << ", " << num_joints << ", " << heatmap_h << ", " << heatmap_w << "]" << std::endl;

    // Resize image back to original for drawing (optional, or draw on original)
    // We will draw on the original image. We need to map 64x48 back to original image dims.
    float scale_x = (float)img.cols / heatmap_w; // Note: We resized input to 192x256, but heatmaps are 48x64. 
                                                 // The model sees 192x256. 
                                                 // If we want to map back to *original* image, we should map from heatmap -> input_size -> original_size
                                                 // However, we usually just map heatmap -> original directly if the crop was the whole image.
                                                 // Simple approach: Map heatmap coordinate (0..48, 0..64) to (0..img.cols, 0..img.rows)

    std::vector<cv::Point> keypoints;
    for (int i = 0; i < num_joints; ++i) {
        // Find max in heatmap
        float max_val = -1e9;
        int max_x = 0, max_y = 0;
        
        int offset = i * heatmap_h * heatmap_w;
        for (int y = 0; y < heatmap_h; ++y) {
            for (int x = 0; x < heatmap_w; ++x) {
                float val = output_data[offset + y * heatmap_w + x];
                if (val > max_val) {
                    max_val = val;
                    max_x = x;
                    max_y = y;
                }
            }
        }

        // Map to original image coordinates
        int img_x = (int)(max_x * ((float)img.cols / heatmap_w));
        int img_y = (int)(max_y * ((float)img.rows / heatmap_h));
        keypoints.push_back(cv::Point(img_x, img_y));
        
        // Draw point
        cv::circle(img, cv::Point(img_x, img_y), 4, cv::Scalar(0, 0, 255), -1);
    }

    // Draw Skeleton
    for (const auto& limb : SKELETON) {
        cv::line(img, keypoints[limb.first], keypoints[limb.second], cv::Scalar(0, 255, 0), 2);
    }

    std::string output_path = "result.jpg";
    cv::imwrite(output_path, img);
    std::cout << "Result saved to " << output_path << std::endl;
    
    return 0;
}
