#include "video_classification/image_processor.hpp"

std::vector<float> ImageProcessor::normalize_and_convert(
    const std::vector<cv::Mat> &channels_vec,
    const std::vector<float> &mean,
    const std::vector<float> &std,
    int channels,
    int size,
    const std::string &format) {

  std::vector<float> pixel_values;

  // Normalize channels
  std::vector<cv::Mat> normalized_channels(static_cast<size_t>(channels));
  for (int c = 0; c < channels; ++c) {
    normalized_channels[static_cast<size_t>(c)] =
        (channels_vec[static_cast<size_t>(c)] - mean[static_cast<size_t>(c)]) /
        std[static_cast<size_t>(c)];
  }

  // Convert to NCHW or NHWC format
  if (format == "FORMAT_NCHW" || format == "FORMAT_NONE") {
    // NCHW: channels first
    for (int c = 0; c < channels; ++c) {
      if (normalized_channels[static_cast<size_t>(c)].isContinuous()) {
        const float *ptr = normalized_channels[static_cast<size_t>(c)].ptr<float>();
        pixel_values.insert(pixel_values.end(), ptr, ptr + size * size);
      } else {
        // Fallback for non-continuous memory
        for (int r = 0; r < normalized_channels[static_cast<size_t>(c)].rows; ++r) {
          const float *ptr = normalized_channels[static_cast<size_t>(c)].ptr<float>(r);
          pixel_values.insert(pixel_values.end(), ptr,
                              ptr + normalized_channels[static_cast<size_t>(c)].cols);
        }
      }
    }
  } else if (format == "FORMAT_NHWC") {
    // NHWC: channels last
    for (int h = 0; h < size; ++h) {
      for (int w = 0; w < size; ++w) {
        for (int c = 0; c < channels; ++c) {
          pixel_values.push_back(
              normalized_channels[static_cast<size_t>(c)].at<float>(h, w));
        }
      }
    }
  }

  return pixel_values;
}
