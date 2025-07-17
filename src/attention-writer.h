#pragma once

#include <fstream>
#include <vector>
#include <string>
#include <cstring>  // For strcmp

// Forward declarations to avoid including heavy headers
struct ggml_tensor;

class AttentionScoreWriter {
private:
    std::ofstream bin_file;
    void* metadata; // Use void* to avoid including json.hpp in header
    bool initialized = false;
    bool enable_attention_scores_retrieval = false;
    std::string file_path;

    // For callback-based capture
    bool capturing_enabled = false;
    int capture_counter = 0;
    int n_layer;
    // Helper method for writing tensor data
    void stream_tensor_to_disk(ggml_tensor* tensor);

public:
    AttentionScoreWriter();
    ~AttentionScoreWriter();

    bool initialize(const std::string& file_path, int n_layer, int n_head);
    bool is_initialized() const;
    void reset();
    std::string get_metadata_json_string() const;

    void set_enable_attention_scores_retrieval(bool enable);
    bool get_enable_attention_scores_retrieval() const;

    // Callback-based capture methods
    void enable_capture_for_execution();
    void capture_tensor_during_execution(ggml_tensor* tensor);
    void finish_execution_capture();
    
    // Check if we should capture this tensor (for callback filtering)
    bool should_capture_tensor(ggml_tensor* tensor) const;
    
    // Finalize any pending operations
    void finalize_streaming();

    // Legacy methods (kept for backward compatibility if needed)
    void write_batch(const std::vector<ggml_tensor*>& tensors);
    std::vector<ggml_tensor*>& get_attention_tensors_per_layer();
    const std::vector<ggml_tensor*>& get_attention_tensors_per_layer() const;

private:
    // Legacy storage (kept for backward compatibility)
    std::vector<ggml_tensor*> attention_tensors_per_layer;
};

// Global access functions
AttentionScoreWriter& get_attention_writer();
void init_attention_writer(int n_layer, int n_head);
std::string get_attention_writer_metadata_string();

// Callback function for ggml_backend_sched_set_eval_callback
extern "C" {
    bool attention_capture_callback(struct ggml_tensor * tensor, bool ask, void * user_data);
}