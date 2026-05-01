#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cuda_runtime.h>

// External function declared in kmeans_kernel.cu
extern "C" void launch_compute_clusters(
    const float* d_features, 
    const float* d_centroids, 
    int* d_assignments, 
    int n_frames, 
    int k_clusters, 
    int n_features,
    int threads_per_block
);

void saveCentroids(float* centroids, int k, int features, const std::string& filename) {
    std::ofstream file(filename);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < features; j++) {
            file << centroids[i * features + j] << (j == features - 1 ? "" : ",");
        }
        file << "\n";
    }
    file.close();
    std::cout << "Tactical centroids saved to " << filename << std::endl;
}


int main() {
    std::string filename = "data/processed/home_team_features.csv";
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << ". Did you run parse_metrica.py?" << std::endl;
        return 1;
    }
    
    std::vector<float> h_features;
    std::string line;
    std::getline(file, line); // Skip CSV header
    
    int n_frames = 0;
    // Updated to 22 features (11 players * 2 coordinates)
    int n_features = 22; 
    
    // Parse the processed CSV
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::getline(ss, cell, ','); // Skip Frame_ID
        
        for (int i = 0; i < n_features; ++i) {
            if (std::getline(ss, cell, ',')) {
                h_features.push_back(std::stof(cell));
            }
        }
        n_frames++;
    }
    file.close();
    
    std::cout << "Loaded " << n_frames << " frames from CSV with " << n_features << " features each." << std::endl;
    
    // Define initial centroids for demonstration (e.g. 22-dimensional dummy data)
    int k_clusters = 3;
    std::vector<float> h_centroids(k_clusters * n_features, 0.0f);
    
    // Initialize mock formation centroids (spread out to simulate players)
    for (int c = 0; c < k_clusters; ++c) {
        for (int f = 0; f < n_features; ++f) {
            // Even is X, Odd is Y. Generate some dummy relative offsets.
            float val = (float)(f - 10) * 2.0f;
            h_centroids[c * n_features + f] = (c % 2 == 0) ? val : -val;
        }
    }

    // Save the initial centroids (Formations)
    saveCentroids(h_centroids.data(), k_clusters, n_features, "data/processed/tactical_centroids.csv");    
    std::vector<int> h_assignments(n_frames, -1);
    
    float *d_features = nullptr;
    float *d_centroids = nullptr;
    int *d_assignments = nullptr;
    
    // Allocate device memory
    cudaMalloc((void**)&d_features, n_frames * n_features * sizeof(float));
    cudaMalloc((void**)&d_centroids, k_clusters * n_features * sizeof(float));
    cudaMalloc((void**)&d_assignments, n_frames * sizeof(int));
    
    // Copy data from host to device
    cudaMemcpy(d_features, h_features.data(), n_frames * n_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centroids, h_centroids.data(), k_clusters * n_features * sizeof(float), cudaMemcpyHostToDevice);
    
    // Launch kernel parameters
    int threads_per_block = 256;
    std::cout << "Launching CUDA kernel with " << threads_per_block << " threads per block..." << std::endl;
    
    launch_compute_clusters(
        d_features, 
        d_centroids, 
        d_assignments, 
        n_frames, 
        k_clusters, 
        n_features,
        threads_per_block
    );
    
    // Copy results back to host
    cudaMemcpy(h_assignments.data(), d_assignments, n_frames * sizeof(int), cudaMemcpyDeviceToHost);
    
    std::cout << "Kernel execution completed successfully." << std::endl;
    std::cout << "First 5 cluster assignments: ";
    for (int i = 0; i < std::min(5, n_frames); ++i) {
        std::cout << h_assignments[i] << " ";
    }
    std::cout << std::endl;
    
    // Cleanup device memory
    cudaFree(d_features);
    cudaFree(d_centroids);
    cudaFree(d_assignments);
    
    return 0;
}
