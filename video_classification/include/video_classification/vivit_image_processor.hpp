#pragma once
#include "image_processor.hpp"
#include <rapidjson/document.h>
#include <string>
#include <vector>

class VivitImageProcessor : public ImageProcessor {
public:
  explicit VivitImageProcessor(const rapidjson::Document &config);

  std::vector<float> process(const std::vector<cv::Mat> &frames, int channels,
                             const std::string &format) override;

private:
  int shortest_edge;
  int crop_size;
  float rescale_factor;
  bool offset;
  std::vector<float> mean;
  std::vector<float> std;
};
