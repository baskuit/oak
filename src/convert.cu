#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHUNK_SIZE 384
#define OUT_FEATURES 128  // match with Python side

__device__ void convert_chunk(const uint8_t* chunk, float* out) {
    // Example: Just fill with normalized values from bytes
    for (int i = 0; i < OUT_FEATURES && i < CHUNK_SIZE; ++i) {
        out[i] = chunk[i] / 255.0f;
    }

    // Zero pad if needed
    for (int i = CHUNK_SIZE; i < OUT_FEATURES; ++i) {
        out[i] = 0.0f;
    }
}

__global__ void process_chunks_kernel(const uint8_t* __restrict__ input,
                                      float* __restrict__ output,
                                      int num_chunks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_chunks) return;

    const uint8_t* chunk = input + idx * CHUNK_SIZE;
    float* out = output + idx * OUT_FEATURES;
    convert_chunk(chunk, out);
}

void process_chunks(torch::Tensor input, torch::Tensor output) {
    const int num_chunks = input.size(0) / CHUNK_SIZE;

    const int threads = 256;
    const int blocks    = (num_chunks + threads - 1) / threads;

    process_chunks_kernel<<<blocks, threads>>>(
        input.data_ptr<uint8_t>(),
        output.data_ptr<float>(),
        num_chunks
    );
}
    
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("process_chunks", &process_chunks, "Process Chunks (CUDA)");
}