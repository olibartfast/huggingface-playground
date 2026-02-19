#include "video_classification/vivit_image_processor.hpp"

#include <opencv2/opencv.hpp>
#include <stdexcept>

VivitImageProcessor::VivitImageProcessor(const rapidjson::Document &config) {
  shortest_edge =
      config.HasMember("shortest_edge") && config["shortest_edge"].IsInt()
          ? config["shortest_edge"].GetInt()
          : 256;
  crop_size = config.HasMember("crop_size") && config["crop_size"].IsInt()
                  ? config["crop_size"].GetInt()
                  : 224;
  rescale_factor =
      config.HasMember("rescale_factor") && config["rescale_factor"].IsFloat()
          ? config["rescale_factor"].GetFloat()
          : 1.0f / 127.5f;
  offset = config.HasMember("offset") && config["offset"].IsBool()
               ? config["offset"].GetBool()
               : true;

  mean = {0.485f, 0.456f, 0.406f};
  if (config.HasMember("mean") && config["mean"].IsArray()) {
    mean.clear();
    for (rapidjson::SizeType i = 0; i < config["mean"].Size(); ++i) {
      if (config["mean"][i].IsFloat() || config["mean"][i].IsDouble()) {
        mean.push_back(config["mean"][i].GetFloat());
      }
    }
  }
  std = {0.229f, 0.224f, 0.225f};
  if (config.HasMember("std") && config["std"].IsArray()) {
    std.clear();
    for (rapidjson::SizeType i = 0; i < config["std"].Size(); ++i) {
      if (config["std"][i].IsFloat() || config["std"][i].IsDouble()) {
        std.push_back(config["std"][i].GetFloat());
      }
    }
  }
}

std::vector<float>
VivitImageProcessor::process(const std::vector<cv::Mat> &frames, int channels,
                             const std::string &format) {
  std::vector<float> pixel_values;
  for (const auto &frame : frames) {
    // Resize
    int height = frame.rows;
    int width = frame.cols;
    int new_height, new_width;
    if (height < width) {
      new_height = shortest_edge;
      new_width = static_cast<int>(static_cast<float>(width) / static_cast<float>(height) * static_cast<float>(new_height));
    } else {
      new_width = shortest_edge;
      new_height = static_cast<int>(static_cast<float>(height) / static_cast<float>(width) * static_cast<float>(new_width));
    }
    cv::Mat resized_frame;
    cv::resize(frame, resized_frame, cv::Size(new_width, new_height), 0, 0,
               cv::INTER_CUBIC);

    // Center crop
    int top = (new_height - crop_size) / 2;
    int left = (new_width - crop_size) / 2;
    cv::Mat cropped_frame =
        resized_frame(cv::Rect(left, top, crop_size, crop_size));

    // Convert to float
    cv::Mat float_frame;
    cropped_frame.convertTo(float_frame, CV_32F);

    // Rescale
    float_frame = float_frame * rescale_factor;
    if (offset) {
      float_frame = float_frame - 1.0;
    }

    // Normalize and convert to NCHW/NHWC
    std::vector<cv::Mat> channels_vec(static_cast<size_t>(channels));
    cv::split(float_frame, channels_vec);

    auto frame_pixels = normalize_and_convert(channels_vec, mean, std, channels, crop_size, format);
    pixel_values.insert(pixel_values.end(), frame_pixels.begin(), frame_pixels.end());
  }
  return pixel_values;
}
