#include <cuda_runtime.h>

__global__ void compute_clusters_kernel(
    const float* features, 
    const float* centroids, 
    int* assignments,      
    int n_frames,
    int k_clusters,
    int n_features
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_frames) {
        float min_dist = 1e30f;
        int best_cluster = -1;
        
        // Dynamically allocated shared memory for centroids
        // Requires shared_mem_size = k_clusters * n_features * sizeof(float)
        extern __shared__ float shared_centroids[];
        
        // Coalesced load of centroids into shared memory
        // Assuming k_clusters * n_features is smaller than blockDim.x
        for (int i = threadIdx.x; i < k_clusters * n_features; i += blockDim.x) {
            shared_centroids[i] = centroids[i];
        }
        __syncthreads();
        
        // Compute Euclidean distance to all centroids
        for (int c = 0; c < k_clusters; ++c) {
            float dist = 0.0f;
            for (int f = 0; f < n_features; ++f) {
                float diff = features[idx * n_features + f] - shared_centroids[c * n_features + f];
                dist += diff * diff;
            }
            if (dist < min_dist) {
                min_dist = dist;
                best_cluster = c;
            }
        }
        
        // Assign frame to the closest cluster
        assignments[idx] = best_cluster;
    }
}

// C-linkage wrapper to launch the kernel from host C++ code
extern "C" void launch_compute_clusters(
    const float* d_features, 
    const float* d_centroids, 
    int* d_assignments, 
    int n_frames, 
    int k_clusters, 
    int n_features,
    int threads_per_block
) {
    int blocks_per_grid = (n_frames + threads_per_block - 1) / threads_per_block;
    int shared_mem_size = k_clusters * n_features * sizeof(float);
    
    compute_clusters_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
        d_features, d_centroids, d_assignments, n_frames, k_clusters, n_features
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        // Handle error (e.g. print error string)
        // printf("CUDA Kernel Error: %s\n", cudaGetErrorString(err));
    }
    
    cudaDeviceSynchronize();
}
