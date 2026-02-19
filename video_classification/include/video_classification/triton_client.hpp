#pragma once

#include "json_utils.hpp"
#include <http_client.h>
#include <map>
#include <memory>
#include <rapidjson/document.h>
#include <string>
#include <vector>

/**
 * @brief Holds metadata and configuration for a Triton model
 */
struct ModelInfo {
  std::string output_name_;      ///< Name of the output tensor
  std::string input_name_;       ///< Name of the input tensor
  std::string input_datatype_;   ///< Data type of input (e.g., "FP32")
  int input_c_;                  ///< Number of input channels
  int input_h_;                  ///< Input height
  int input_w_;                  ///< Input width
  std::string input_format_;     ///< Input format ("FORMAT_NCHW" or "FORMAT_NHWC")
  int type1_;                    ///< OpenCV type for single channel (e.g., CV_32FC1)
  int type3_;                    ///< OpenCV type for three channels (e.g., CV_32FC3)
  int max_batch_size_;           ///< Maximum batch size supported by model
};

/**
 * @brief Client for interacting with Triton Inference Server
 */
class TritonClient {
public:
  struct InferenceResult {
    std::string label;     ///< Human-readable label
    float probability;     ///< Prediction probability (0.0 to 1.0)
  };

  /**
   * @brief Constructs a Triton client
   * @param server_url URL of the Triton server (e.g., "http://localhost:8000")
   * @param labels_file Optional path to file containing class labels
   */
  TritonClient(const std::string &server_url,
               const std::string &labels_file = "");

  /**
   * @brief Performs inference on the given input data
   * @param input_data Preprocessed input data as float vector
   * @param model_name Name of the model on Triton server
   * @param model_info Model metadata
   * @param shape Shape of the input tensor
   * @return Vector of top predictions with labels and probabilities
   */
  std::vector<InferenceResult> infer(const std::vector<float> &input_data,
                                     const std::string &model_name,
                                     const ModelInfo &model_info,
                                     const std::vector<int64_t> &shape);

  /**
   * @brief Retrieves model metadata and configuration
   * @param model_name Name of the model on Triton server
   * @param model_info Output parameter to store model information
   */
  void get_model_info(const std::string &model_name, ModelInfo &model_info);

private:
  std::vector<InferenceResult>
  postprocess_results(const std::vector<float> &logits);
  static void parse_model_http(const rapidjson::Document &model_metadata,
                               const rapidjson::Document &model_config,
                               const size_t batch_size, ModelInfo *model_info);
  void load_labels(const std::string &labels_file);

  std::unique_ptr<triton::client::InferenceServerHttpClient> http_client_;
  std::map<std::string, std::string> id2label_;
};
