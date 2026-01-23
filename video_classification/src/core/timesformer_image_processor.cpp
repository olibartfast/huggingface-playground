#include "video_classification/timesformer_image_processor.hpp"

#include <opencv2/opencv.hpp>
#include <stdexcept>

TimeSformerImageProcessor::TimeSformerImageProcessor(
    const rapidjson::Document &config) {
  shortest_edge =
      config.HasMember("shortest_edge") && config["shortest_edge"].IsInt()
          ? config["shortest_edge"].GetInt()
          : 224;
  crop_size = config.HasMember("crop_size") && config["crop_size"].IsInt()
                  ? config["crop_size"].GetInt()
                  : 224;
  rescale_factor =
      config.HasMember("rescale_factor") && config["rescale_factor"].IsFloat()
          ? config["rescale_factor"].GetFloat()
          : 1.0f / 255.0f;

  mean = {0.45f, 0.45f, 0.45f};
  if (config.HasMember("mean") && config["mean"].IsArray()) {
    mean.clear();
    for (rapidjson::SizeType i = 0; i < config["mean"].Size(); ++i) {
      if (config["mean"][i].IsFloat() || config["mean"][i].IsDouble()) {
        mean.push_back(config["mean"][i].GetFloat());
      }
    }
  }
  std = {0.225f, 0.225f, 0.225f};
  if (config.HasMember("std") && config["std"].IsArray()) {
    std.clear();
    for (rapidjson::SizeType i = 0; i < config["std"].Size(); ++i) {
      if (config["std"][i].IsFloat() || config["std"][i].IsDouble()) {
        std.push_back(config["std"][i].GetFloat());
      }
    }
  }
}

std::vector<float> TimeSformerImageProcessor::process(
    const std::vector<cv::Mat> &frames, int channels,
    const std::string &format) {
  std::vector<float> pixel_values;
  for (const auto &frame : frames) {
    // Resize
    int height = frame.rows;
    int width = frame.cols;
    int new_height, new_width;
    if (height < width) {
      new_height = shortest_edge;
      new_width = static_cast<int>(static_cast<float>(width) /
                                   static_cast<float>(height) *
                                   static_cast<float>(new_height));
    } else {
      new_width = shortest_edge;
      new_height = static_cast<int>(static_cast<float>(height) /
                                    static_cast<float>(width) *
                                    static_cast<float>(new_width));
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

    // Normalize and convert to NCHW/NHWC
    std::vector<cv::Mat> channels_vec(static_cast<size_t>(channels));
    cv::split(float_frame, channels_vec);
    for (int c = 0; c < channels; ++c) {
      channels_vec[static_cast<size_t>(c)] =
          (channels_vec[static_cast<size_t>(c)] -
           mean[static_cast<size_t>(c)]) /
          std[static_cast<size_t>(c)];
      if (format == "FORMAT_NCHW" || format == "FORMAT_NONE") {
        if (channels_vec[static_cast<size_t>(c)].isContinuous()) {
          const float *ptr = channels_vec[static_cast<size_t>(c)].ptr<float>();
          pixel_values.insert(pixel_values.end(), ptr,
                              ptr + crop_size * crop_size);
        } else {
          for (int r = 0; r < channels_vec[static_cast<size_t>(c)].rows; ++r) {
            const float *ptr =
                channels_vec[static_cast<size_t>(c)].ptr<float>(r);
            pixel_values.insert(pixel_values.end(), ptr,
                                ptr +
                                    channels_vec[static_cast<size_t>(c)].cols);
          }
        }
      }
    }
    if (format == "FORMAT_NHWC") {
      for (int h = 0; h < crop_size; ++h) {
        for (int w = 0; w < crop_size; ++w) {
          for (int c = 0; c < channels; ++c) {
            pixel_values.push_back(
                channels_vec[static_cast<size_t>(c)].at<float>(h, w));
          }
        }
      }
    }
  }
  return pixel_values;
}
