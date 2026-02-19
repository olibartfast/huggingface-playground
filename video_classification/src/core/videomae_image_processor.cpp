#include "video_classification/videomae_image_processor.hpp"

#include <opencv2/opencv.hpp>

VideoMAEImageProcessor::VideoMAEImageProcessor(
    const rapidjson::Document &config) {
  image_size = config.HasMember("image_size") && config["image_size"].IsInt()
                   ? config["image_size"].GetInt()
                   : 224;
  mean = {0.485f, 0.456f, 0.406f};
  if (config.HasMember("mean") && config["mean"].IsArray()) {
    mean.clear();
    for (rapidjson::SizeType i = 0; i < config["mean"].Size() && i < 3; ++i) {
      if (config["mean"][i].IsFloat() || config["mean"][i].IsDouble()) {
        mean.push_back(config["mean"][i].GetFloat());
      }
    }
  }
  std = {0.229f, 0.224f, 0.225f};
  if (config.HasMember("std") && config["std"].IsArray()) {
    std.clear();
    for (rapidjson::SizeType i = 0; i < config["std"].Size() && i < 3; ++i) {
      if (config["std"][i].IsFloat() || config["std"][i].IsDouble()) {
        std.push_back(config["std"][i].GetFloat());
      }
    }
  }
}

std::vector<float>
VideoMAEImageProcessor::process(const std::vector<cv::Mat> &frames,
                                int channels, const std::string &format) {
  std::vector<float> pixel_values;
  for (const auto &frame : frames) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(image_size, image_size));

    cv::Mat float_frame;
    resized.convertTo(float_frame, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels_vec(static_cast<size_t>(channels));
    cv::split(float_frame, channels_vec);

    auto frame_pixels = normalize_and_convert(channels_vec, mean, std, channels, image_size, format);
    pixel_values.insert(pixel_values.end(), frame_pixels.begin(), frame_pixels.end());
  }
  return pixel_values;
}
