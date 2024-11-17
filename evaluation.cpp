#define BUZZDB_MAIN_DISABLED
#include "buzzdb.cpp"
#include <iomanip>
#include <fstream>
#include <chrono>

std::vector<std::vector<float>> readGIST1M(const std::string& filename, size_t num_vectors) {
    std::vector<std::vector<float>> vectors;
    std::ifstream file(filename, std::ios::binary);
    
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    for (size_t i = 0; i < num_vectors; i++) {
        int32_t dim;
        file.read(reinterpret_cast<char*>(&dim), sizeof(int32_t));
        
        std::vector<float> vec(dim);
        file.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float));
        
        if (file.fail()) break;
        vectors.push_back(std::move(vec));
    }
    
    return vectors;
}

std::vector<std::vector<float>> generateSyntheticData(size_t num_vectors, size_t dim) {
    std::vector<std::vector<float>> vectors;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    for (size_t i = 0; i < num_vectors; i++) {
        std::vector<float> vec(dim);
        for (auto& val : vec) {
            val = dist(gen);
        }
        vectors.push_back(std::move(vec));
    }
    
    return vectors;
}

struct BenchmarkResults {
    double build_time;
    double query_time;
    double memory_usage;
    double accuracy;
    size_t num_vectors;
    size_t dimension;
};

void printResults(const std::string& title, const BenchmarkResults& results) {
    std::cout << "\n=== " << title << " ===\n"
              << "Dataset size: " << results.num_vectors << " vectors\n"
              << "Dimension: " << results.dimension << "\n"
              << "Build time: " << std::fixed << std::setprecision(3) << results.build_time << " seconds\n"
              << "Average query time: " << std::fixed << std::setprecision(6) << results.query_time * 1000 << " ms\n"
              << "Memory usage: " << results.memory_usage / 1024 / 1024 << " MB\n"
              << "Accuracy: " << results.accuracy * 100 << "%\n";
}

BenchmarkResults runBenchmark(const std::string& index_type,
                            const std::vector<std::vector<float>>& data,
                            const std::vector<std::vector<float>>& queries,
                            size_t k) {
    BufferManager buffer_manager(true);
    BenchmarkResults results;
    results.num_vectors = data.size();
    results.dimension = data[0].size();
    
    std::unique_ptr<VectorIndex> index;
    if (index_type == "hnsw") {
        auto hnsw = std::make_unique<HNSWIndex>(data[0].size(), buffer_manager);
        hnsw->setParameters(4, 16, 200);
        index = std::move(hnsw);
    } else {
        index = std::make_unique<KDTreeIndex>(data[0].size(), buffer_manager);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < data.size(); i++) {
        index->insert(data[i], i);
    }
    auto end = std::chrono::high_resolution_clock::now();
    results.build_time = std::chrono::duration<double>(end - start).count();
    
    results.memory_usage = getCurrentMemoryUsage();
    
    start = std::chrono::high_resolution_clock::now();
    double total_accuracy = 0.0;
    
    KDTreeIndex exact_index(data[0].size(), buffer_manager);
    for (size_t i = 0; i < data.size(); i++) {
        exact_index.insert(data[i], i);
    }
    
    for (const auto& query : queries) {
        auto approx_results = index->search(query, k);
        auto exact_results = exact_index.search(query, k);
        
        size_t correct = 0;
        for (const auto& approx : approx_results) {
            for (const auto& exact : exact_results) {
                if (approx.first == exact.first) {
                    correct++;
                    break;
                }
            }
        }
        total_accuracy += static_cast<double>(correct) / k;
    }
    
    end = std::chrono::high_resolution_clock::now();
    results.query_time = std::chrono::duration<double>(end - start).count() / queries.size();
    results.accuracy = total_accuracy / queries.size();
    
    return results;
}

int main() {
    try {
        std::cout << "Starting Vector Database Evaluation...\n";
        
        std::vector<size_t> sizes = {100, 500}; 
        std::vector<size_t> dims = {50, 100};
        
        for (auto size : sizes) {
            for (auto dim : dims) {
                std::cout << "\nEvaluating synthetic dataset: " << size << " vectors, " << dim << " dimensions\n";
                
                auto data = generateSyntheticData(size, dim);
                auto queries = generateSyntheticData(10, dim); 
                
                // Test HNSW
                auto hnsw_results = runBenchmark("hnsw", data, queries, 10);
                printResults("HNSW Index", hnsw_results);
                
                // Test KD-tree
                auto kdtree_results = runBenchmark("kdtree", data, queries, 10);
                printResults("KD-tree Index", kdtree_results);
            }
        }
        
        std::cout << "\nEvaluating GIST dataset subset\n";
        try {
            auto gist_data = readGIST1M("gist/gist_base.fvecs", 1000); 
            auto gist_queries = readGIST1M("gist/gist_query.fvecs", 10); 
            
            // Test HNSW
            auto hnsw_results = runBenchmark("hnsw", gist_data, gist_queries, 10);
            printResults("HNSW Index on GIST", hnsw_results);
            
            // Test KD-tree
            auto kdtree_results = runBenchmark("kdtree", gist_data, gist_queries, 10);
            printResults("KD-tree Index on GIST", kdtree_results);
            
        } catch (const std::exception& e) {
            std::cerr << "Error with GIST dataset: " << e.what() << "\n";
            std::cerr << "Skipping GIST evaluation\n";
        }
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
} 