#include "attention-writer.h"
#include "llama-graph.h" // for ggml_tensor and other dependencies
#include "../vendor/nlohmann/json.hpp"

#include <algorithm>

using json = nlohmann::ordered_json;

// Global instance
static AttentionScoreWriter g_attention_writer;

AttentionScoreWriter::AttentionScoreWriter() : metadata(new json()) {}

AttentionScoreWriter::~AttentionScoreWriter() {
    if (bin_file.is_open()) {
        bin_file.close();
    }
    delete static_cast<json*>(metadata);
}

bool AttentionScoreWriter::initialize(const std::string& file_path, int n_layer, int n_head) {
    this->file_path = file_path;
    this->n_layer = n_layer;
    bin_file.open(this->file_path, std::ios::binary | std::ios::trunc);
    if (!bin_file.is_open()) return false;
    
    json* meta = static_cast<json*>(metadata);
    meta->clear();
    (*meta)["metadata"] = {
        {"n_layer", n_layer},
        {"n_head", n_head}
    };
    (*meta)["content"] = json::array();
    
    // Initialize callback state
    attention_tensors_per_layer.clear();
    capturing_enabled = false;
    enable_attention_scores_retrieval = false;
    initialized = true;
    
    return true;
}

bool AttentionScoreWriter::is_initialized() const {
    return initialized;
}

void AttentionScoreWriter::reset() {
    if (!initialized) return;
    
    json* meta = static_cast<json*>(metadata);
    (*meta)["content"].clear();    
    attention_tensors_per_layer.clear();
    
    // Reset callback state
    capturing_enabled = false;
    
    // Clear and reopen the binary file to remove all data
    if (bin_file.is_open()) {
        bin_file.close();
    }    
    // Reopen in truncate mode to clear existing data
    bin_file.open(this->file_path, std::ios::binary | std::ios::trunc);
}

void AttentionScoreWriter::set_enable_attention_scores_retrieval(bool enable) {
    enable_attention_scores_retrieval = enable;
}

bool AttentionScoreWriter::get_enable_attention_scores_retrieval() const {
    return enable_attention_scores_retrieval;
}

// CALLBACK-BASED CAPTURE METHODS
void AttentionScoreWriter::enable_capture_for_execution() {
    if (!initialized) return;
    capturing_enabled = true;
    capture_counter = 0;
}

void AttentionScoreWriter::capture_tensor_during_execution(ggml_tensor* tensor) {
    if (!initialized || !capturing_enabled || !tensor || !tensor->data) return;
    capture_counter++;
    // Update metadata on first tensor capture
    if (capture_counter % n_layer == 0) {
        json* meta = static_cast<json*>(metadata);
        int64_t n_query = tensor->ne[1];
        int64_t n_key = tensor->ne[0];
        
        (*meta)["content"].push_back({
            {"n_query", n_query},
            {"n_key", n_key}
        });
    }
    
    // Stream tensor data directly to disk
    stream_tensor_to_disk(tensor);
}

void AttentionScoreWriter::finish_execution_capture() {
    if (!initialized) return;
    capturing_enabled = false;
    
    // Ensure all data is written to disk
    if (bin_file.is_open()) {
        bin_file.flush();
    }
}

bool AttentionScoreWriter::should_capture_tensor(ggml_tensor* tensor) const {
    if (!initialized || !capturing_enabled || !tensor) return false;
    
    // Check if this is a softmax tensor (attention scores)
    return (strcmp(tensor->name, "attention_scores") == 0);
}

void AttentionScoreWriter::stream_tensor_to_disk(ggml_tensor* tensor) {
    if (!tensor || !tensor->data || !bin_file.is_open()) return;
    
    const int64_t n_head = tensor->ne[2];
    const int64_t n_query = tensor->ne[1];
    const int64_t n_key = tensor->ne[0];
    const size_t total_elements = n_head * n_query * n_key;
    
    // Stream in chunks to avoid large memory allocations
    const size_t chunk_size = 1024;
    std::vector<float> temp_buffer(chunk_size);
    
    for (size_t offset = 0; offset < total_elements; offset += chunk_size) {
        const size_t current_chunk = std::min(chunk_size, total_elements - offset);
        
        // Use ggml_backend_tensor_get for safe tensor data access
        ggml_backend_tensor_get(tensor, temp_buffer.data(), 
                              offset * sizeof(float), 
                              current_chunk * sizeof(float));
        
        bin_file.write(reinterpret_cast<const char*>(temp_buffer.data()), 
                      current_chunk * sizeof(float));
    }
}

void AttentionScoreWriter::finalize_streaming() {
    if (bin_file.is_open()) {
        bin_file.flush();
    }
    capturing_enabled = false;
}

// LEGACY METHODS (kept for backward compatibility)
void AttentionScoreWriter::write_batch(const std::vector<ggml_tensor*>& tensors) {
    if (!initialized) return;
    
    json* meta = static_cast<json*>(metadata);
    bool first_layer = true;
    for (size_t l = 0; l < tensors.size(); ++l) {
        ggml_tensor * attn = tensors[l];
        if (!attn || !attn->buffer) continue;
        
        int64_t n_head = attn->ne[2];
        int64_t n_query = attn->ne[1]; 
        int64_t n_key = attn->ne[0];
        
        if (first_layer) {
            (*meta)["content"].push_back({
                {"n_query", n_query},
                {"n_key", n_key}
            });
            first_layer = false;
        }

        // Stream data
        const size_t chunk_size = 1024;
        std::vector<float> temp_buffer(chunk_size);
        const size_t total_elements = n_head * n_query * n_key;
        
        for (size_t offset = 0; offset < total_elements; offset += chunk_size) {
            const size_t current_chunk = std::min(chunk_size, total_elements - offset);
            ggml_backend_tensor_get(attn, temp_buffer.data(), 
                                  offset * sizeof(float), 
                                  current_chunk * sizeof(float));
            bin_file.write(reinterpret_cast<const char*>(temp_buffer.data()), 
                          current_chunk * sizeof(float));
        }
    }
    bin_file.flush();
}

std::vector<ggml_tensor*>& AttentionScoreWriter::get_attention_tensors_per_layer() {
    return attention_tensors_per_layer;
}

const std::vector<ggml_tensor*>& AttentionScoreWriter::get_attention_tensors_per_layer() const {
    return attention_tensors_per_layer;
}

// METADATA AND GLOBAL ACCESS
std::string AttentionScoreWriter::get_metadata_json_string() const {
    const json* meta = static_cast<const json*>(metadata);
    return meta->dump();
}

// Global access functions
AttentionScoreWriter& get_attention_writer() {
    return g_attention_writer;
}

void init_attention_writer(int n_layer, int n_head) {
    g_attention_writer.initialize("cache/attention_scores.bin", n_layer, n_head);
}

std::string get_attention_writer_metadata_string() {
    return g_attention_writer.get_metadata_json_string();
}

// CALLBACK FUNCTION FOR GGML BACKEND
bool attention_capture_callback(struct ggml_tensor * tensor, bool ask, void * user_data) {
    AttentionScoreWriter* writer = static_cast<AttentionScoreWriter*>(user_data);
    if (ask) {
        // Scheduler is asking if we want to observe this tensor
        return writer->should_capture_tensor(tensor);
    } else {
        // Scheduler is providing the tensor for observation
        // Data is guaranteed to be valid at this point
        if (writer->should_capture_tensor(tensor)) {
            writer->capture_tensor_during_execution(tensor);
        }
        return true; // Continue graph execution
    }
}