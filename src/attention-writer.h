#pragma once

#include <fstream>
#include <vector>
#include <string>

// Forward declarations to avoid including heavy headers
struct ggml_tensor;

class AttentionScoreWriter {
private:
    std::ofstream bin_file;
    void* metadata; // Use void* to avoid including json.hpp in header
    bool initialized = false;
    bool enable_attention_scores_retrieval = false;
    std::vector<ggml_tensor*> attention_tensors_per_layer;
    std::string file_path;

public:
    AttentionScoreWriter();
    ~AttentionScoreWriter();

    bool initialize(const std::string& file_path, int n_layer, int n_head);
    bool is_initialized() const;
    void reset();
    void write_batch(const std::vector<ggml_tensor*>& tensors);
    std::string get_metadata_json_string() const;

    void set_enable_attention_scores_retrieval(bool enable);
    std::vector<ggml_tensor*>& get_attention_tensors_per_layer();
    bool get_enable_attention_scores_retrieval() const;
    const std::vector<ggml_tensor*>& get_attention_tensors_per_layer() const;
};

// Global access functions
AttentionScoreWriter& get_attention_writer();
void init_attention_writer(int n_layer, int n_head);
std::string get_attention_writer_metadata_string();
