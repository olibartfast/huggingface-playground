#include "video_classification/triton_client.hpp"
#include "video_classification/videomae_image_processor.hpp"
#include "video_classification/vivit_image_processor.hpp"
#include "video_classification/timesformer_image_processor.hpp"
#include "video_classification/video_utils.hpp"
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/error/en.h>
#include <stdexcept>
#include <vector>
#include <filesystem>
#include <fstream>

namespace {
constexpr int DEFAULT_WINDOW_SIZE = 16;
constexpr int DEFAULT_BATCH_SIZE = 1;

/**
 * @brief Loads model configuration from a JSON file
 * @param config_path Path to the configuration file
 * @param config Output RapidJSON document to store the configuration
 * @throws std::runtime_error if file cannot be opened or parsed
 */
void load_config_from_file(const std::string &config_path, rapidjson::Document &config) {
  std::ifstream ifs(config_path);
  if (!ifs.is_open()) {
    throw std::runtime_error("Failed to open config file: " + config_path);
  }
  rapidjson::IStreamWrapper isw(ifs);
  config.ParseStream(isw);
  if (config.HasParseError()) {
    throw std::runtime_error("Failed to parse config file: " + config_path +
                            " - Error: " + rapidjson::GetParseError_En(config.GetParseError()) +
                            " at offset " + std::to_string(config.GetErrorOffset()));
  }
}

/**
 * @brief Creates image processor based on model type
 * @param model_type Type of model ("videomae", "vivit", or "timesformer")
 * @param config RapidJSON document with model configuration
 * @return Unique pointer to the appropriate image processor
 * @throws std::runtime_error if model type is unrecognized
 */
std::unique_ptr<ImageProcessor> create_processor(const std::string &model_type,
                                                  const rapidjson::Document &config) {
  if (model_type == "vivit") {
    return std::make_unique<VivitImageProcessor>(config);
  } else if (model_type == "timesformer") {
    return std::make_unique<TimeSformerImageProcessor>(config);
  } else if (model_type == "videomae") {
    return std::make_unique<VideoMAEImageProcessor>(config);
  } else {
    throw std::runtime_error("Unknown model type: " + model_type +
                            ". Supported types: videomae, vivit, timesformer");
  }
}
}

int main(int argc, char **argv) {
  std::string model_name = "videomae_large";
  std::string url = "http://localhost:8000";
  std::string video_path;
  std::string labels_file = "labels/kinetics400.txt";
  std::string config_file;
  std::string model_type = "videomae";  // Default model type
  int batch_size = DEFAULT_BATCH_SIZE;
  int window_size = DEFAULT_WINDOW_SIZE;

  // Parse command-line arguments
  int opt;
  while ((opt = getopt(argc, argv, "m:u:b:l:c:t:")) != -1) {
    switch (opt) {
    case 'm':
      model_name = optarg;
      break;
    case 'u':
      url = optarg;
      break;
    case 'b':
      try {
        batch_size = std::stoi(optarg);
        if (batch_size <= 0) {
          std::cerr << "Error: Batch size must be > 0\n";
          return 1;
        }
      } catch (const std::exception &e) {
        std::cerr << "Error: Invalid batch size '" << optarg << "'\n";
        return 1;
      }
      break;
    case 'l':
      labels_file = optarg;
      break;
    case 'c':
      config_file = optarg;
      break;
    case 't':
      model_type = optarg;
      break;
    default:
      std::cerr << "Usage: " << argv[0]
                << " [-m model] [-u url] [-b batch_size] [-l labels_file] "
                   "[-c config_file] [-t model_type] <video_path>\n"
                << "  -m: Model name on Triton server (default: videomae_large)\n"
                << "  -u: Triton server URL (default: http://localhost:8000)\n"
                << "  -b: Batch size (default: 1)\n"
                << "  -l: Labels file path (default: labels/kinetics400.txt)\n"
                << "  -c: Model config file path (optional)\n"
                << "  -t: Model type: videomae, vivit, or timesformer (default: videomae)\n";
      return 1;
    }
  }
  if (optind >= argc) {
    std::cerr << "Error: Video file must be specified\n";
    return 1;
  }
  video_path = argv[optind];

  // Validate video file exists and is readable
  if (!std::filesystem::exists(video_path)) {
    std::cerr << "Error: Video file does not exist: " << video_path << "\n";
    return 1;
  }
  if (!std::filesystem::is_regular_file(video_path)) {
    std::cerr << "Error: Path is not a regular file: " << video_path << "\n";
    return 1;
  }

  try {
    // Initialize Triton client
    TritonClient client(url, labels_file);

    // Get model info
    ModelInfo model_info;
    client.get_model_info(model_name, model_info);

    // Initialize processor with config
    std::unique_ptr<ImageProcessor> processor;
    rapidjson::Document config;

    if (!config_file.empty()) {
      // Load from specified config file
      load_config_from_file(config_file, config);
      if (config.HasMember("model_type") && config["model_type"].IsString()) {
        model_type = config["model_type"].GetString();
      }
      processor = create_processor(model_type, config);
    } else {
      // Try to auto-detect model type from model name or use specified type
      if (model_type == "auto") {
        if (model_name.find("vivit") != std::string::npos) {
          model_type = "vivit";
        } else if (model_name.find("timesformer") != std::string::npos) {
          model_type = "timesformer";
        } else {
          model_type = "videomae";
        }
      }

      // Try to load default config file
      std::string default_config = "configs/" + model_type + ".json";
      if (std::filesystem::exists(default_config)) {
        load_config_from_file(default_config, config);
      } else {
        // Fallback to hardcoded defaults
        std::cerr << "Warning: No config file found, using hardcoded defaults for "
                  << model_type << "\n";
        std::string config_json;
        if (model_type == "vivit") {
          config_json = R"({"shortest_edge": 256, "crop_size": 224, "rescale_factor": 0.00784313725, "offset": true, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})";
        } else if (model_type == "timesformer") {
          config_json = R"({"shortest_edge": 224, "crop_size": 224, "rescale_factor": 0.003921568627, "mean": [0.45, 0.45, 0.45], "std": [0.225, 0.225, 0.225]})";
        } else {
          config_json = R"({"image_size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]})";
        }
        config.Parse(config_json.c_str());
        if (config.HasParseError()) {
          throw std::runtime_error("Failed to parse default config JSON");
        }
      }
      processor = create_processor(model_type, config);
    }

    // Read video frames at 1 FPS
    auto frames = read_video_frames(video_path, window_size);
    frames = pad_video_frames(frames, window_size);

    // Verify frame count
    if (frames.size() != static_cast<size_t>(window_size)) {
      throw std::runtime_error("Expected " + std::to_string(window_size) +
                               " frames, got " + std::to_string(frames.size()));
    }

    // Preprocess frames
    auto pixel_values = processor->process(frames, model_info.input_c_,
                                          model_info.input_format_);

    // Validate input data size
    const size_t expected_elements = static_cast<size_t>(batch_size) *
                                     static_cast<size_t>(window_size) *
                                     static_cast<size_t>(model_info.input_c_) *
                                     static_cast<size_t>(model_info.input_h_) *
                                     static_cast<size_t>(model_info.input_w_);
    if (pixel_values.size() != expected_elements) {
      throw std::runtime_error("Invalid input data size: expected " +
                               std::to_string(expected_elements) +
                               " elements, got " +
                               std::to_string(pixel_values.size()));
    }

    // Set input shape
    std::vector<int64_t> shape = {batch_size, window_size, model_info.input_c_,
                                  model_info.input_h_, model_info.input_w_};

    // Perform inference
    auto results = client.infer(pixel_values, model_name, model_info, shape);

    // Output results
    std::cout << "Predictions for video '" << video_path << "':\n";
    for (const auto &result : results) {
      std::cout << "  " << result.label << ": " << result.probability << "\n";
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}