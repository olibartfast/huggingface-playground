#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

/**
 * @brief Base class for video frame preprocessing
 *
 * Provides interface for preprocessing video frames for different model types
 */
class ImageProcessor {
public:
  virtual ~ImageProcessor() = default;

  /**
   * @brief Processes video frames into model input format
   * @param frames Vector of frames in RGB format
   * @param channels Number of color channels (typically 3 for RGB)
   * @param format Output format ("FORMAT_NCHW", "FORMAT_NHWC", or "FORMAT_NONE")
   * @return Flattened vector of preprocessed pixel values
   */
  virtual std::vector<float> process(const std::vector<cv::Mat> &frames,
                                     int channels,
                                     const std::string &format) = 0;

protected:
  /**
   * @brief Normalizes and converts image channels to NCHW or NHWC format
   *
   * @param channels_vec Split channels of the image
   * @param mean Mean values for normalization
   * @param std Standard deviation values for normalization
   * @param channels Number of channels
   * @param size Image size (width/height for square images)
   * @param format Output format ("FORMAT_NCHW", "FORMAT_NHWC", or "FORMAT_NONE")
   * @return std::vector<float> Normalized pixel values
   */
  static std::vector<float> normalize_and_convert(
      const std::vector<cv::Mat> &channels_vec,
      const std::vector<float> &mean,
      const std::vector<float> &std,
      int channels,
      int size,
      const std::string &format);
};
