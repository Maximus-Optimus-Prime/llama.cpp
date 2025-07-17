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
    bin_file.open(this->file_path, std::ios::binary | std::ios::trunc);
    if (!bin_file.is_open()) return false;
    json* meta = static_cast<json*>(metadata);
    meta->clear();
    (*meta)["metadata"] = {
        {"n_layer", n_layer},
        {"n_head", n_head}
    };
    (*meta)["content"] = json::array();
    attention_tensors_per_layer.clear();
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
    // Clear and reopen the binary file to remove all data
    if (bin_file.is_open()) {
        bin_file.close();
    }    
    // Reopen in truncate mode to clear existing data
    bin_file.open(this->file_path, std::ios::binary | std::ios::trunc);
}

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

// Global access functions
AttentionScoreWriter& get_attention_writer() {
    return g_attention_writer;
}

void init_attention_writer(int n_layer, int n_head) {
    g_attention_writer.initialize("cache/attention_scores.bin", n_layer, n_head);
}

// Add this method implementation:
std::string AttentionScoreWriter::get_metadata_json_string() const {
    const json* meta = static_cast<const json*>(metadata);
    return meta->dump(); // Return JSON as string
}

std::string get_attention_writer_metadata_string() {
    return g_attention_writer.get_metadata_json_string();
}

std::vector<ggml_tensor*>& AttentionScoreWriter::get_attention_tensors_per_layer() {
    return attention_tensors_per_layer;
}

bool AttentionScoreWriter::get_enable_attention_scores_retrieval() const {
    return enable_attention_scores_retrieval;
}

const std::vector<ggml_tensor*>& AttentionScoreWriter::get_attention_tensors_per_layer() const {
    return attention_tensors_per_layer;
}

void AttentionScoreWriter::set_enable_attention_scores_retrieval(bool enable) {
    enable_attention_scores_retrieval = enable;
}