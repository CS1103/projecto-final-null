#ifndef PROYECTO_FINAL_CSV_LOADER_H
#define PROYECTO_FINAL_CSV_LOADER_H

#include "tensor.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>

namespace utec::data {

using namespace utec::algebra;

template<typename T>
class CSVLoader {
public:
    struct Dataset {
        Tensor<T, 2> X; 
        Tensor<T, 2> Y;  
        std::vector<std::string> feature_names;
        std::vector<std::string> label_names;
    };

    static Dataset load_csv(const std::string& filename, 
                           bool has_header = true, 
                           char delimiter = ',',
                           int label_column = -1) {  
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open file: " + filename);
        }

        std::vector<std::vector<T>> data;
        std::vector<std::string> headers;
        std::string line;
        bool first_line = true;

        while (std::getline(file, line)) {
            if (first_line && has_header) {
                headers = parse_header(line, delimiter);
                first_line = false;
                continue;
            }

            auto row = parse_row(line, delimiter);
            if (!row.empty()) {
                data.push_back(row);
            }
            first_line = false;
        }

        if (data.empty()) {
            throw std::runtime_error("No data found in CSV file");
        }

        size_t n_samples = data.size();
        size_t n_features = data[0].size();
        
        int actual_label_col = (label_column == -1) ? n_features - 1 : label_column;
        
        if (actual_label_col >= static_cast<int>(n_features)) {
            throw std::runtime_error("Label column index out of range");
        }

        size_t feature_count = n_features - 1;
        Tensor<T, 2> X(n_samples, feature_count);
        Tensor<T, 2> Y(n_samples, 1);

        for (size_t i = 0; i < n_samples; ++i) {
            size_t feature_idx = 0;
            for (size_t j = 0; j < n_features; ++j) {
                if (static_cast<int>(j) == actual_label_col) {
                    Y(i, 0) = data[i][j];
                } else {
                    X(i, feature_idx++) = data[i][j];
                }
            }
        }

        Dataset result;
        result.X = std::move(X);
        result.Y = std::move(Y);
        
        if (has_header) {
            for (size_t i = 0; i < headers.size(); ++i) {
                if (static_cast<int>(i) == actual_label_col) {
                    result.label_names.push_back(headers[i]);
                } else {
                    result.feature_names.push_back(headers[i]);
                }
            }
        }

        return result;
    }

    static void save_predictions(const std::string& filename,
                               const Tensor<T, 2>& predictions,
                               const std::vector<std::string>& sample_names = {}) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot create file: " + filename);
        }

        file << "sample_id,prediction\n";

        auto shape = predictions.shape();
        for (size_t i = 0; i < shape[0]; ++i) {
            if (i < sample_names.size()) {
                file << sample_names[i];
            } else {
                file << "sample_" << i;
            }
            
            file << ",";
            
            file << predictions(i, 0) << "\n";
        }
    }

    static void save_comparison(const std::string& filename,
                              const Tensor<T, 2>& X,
                              const Tensor<T, 2>& Y_actual,
                              const Tensor<T, 2>& Y_predicted,
                              const std::vector<std::string>& feature_names = {},
                              const std::vector<std::string>& label_names = {}) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot create file: " + filename);
        }

        file << "sample_id,";
        
        auto x_shape = X.shape();
        for (size_t i = 0; i < x_shape[1]; ++i) {
            if (i < feature_names.size()) {
                file << feature_names[i];
            } else {
                file << "feature_" << i;
            }
            file << ",";
        }
        
        file << "actual,predicted,difference,correct\n";

        for (size_t i = 0; i < x_shape[0]; ++i) {
            file << "sample_" << i << ",";
            
            for (size_t j = 0; j < x_shape[1]; ++j) {
                file << X(i, j) << ",";
            }
            
            T actual = Y_actual(i, 0);
            T predicted = Y_predicted(i, 0);
            T difference = std::abs(actual - predicted);
            
            bool predicted_class = predicted > 0.5;
            bool actual_class = actual > 0.5;
            bool correct = (predicted_class == actual_class);
            
            file << actual << "," << predicted << "," << difference << "," << (correct ? 1 : 0) << "\n";
        }
    }

    static Dataset train_test_split(const Dataset& dataset, 
                                  T test_ratio = 0.2, 
                                  bool shuffle = true,
                                  unsigned seed = 42) {
        auto X_shape = dataset.X.shape();
        size_t n_samples = X_shape[0];
        size_t n_test = static_cast<size_t>(n_samples * test_ratio);
        size_t n_train = n_samples - n_test;

        std::vector<size_t> indices(n_samples);
        std::iota(indices.begin(), indices.end(), 0);

        if (shuffle) {
            std::mt19937 rng(seed);
            std::shuffle(indices.begin(), indices.end(), rng);
        }

        Tensor<T, 2> X_train(n_train, X_shape[1]);
        Tensor<T, 2> Y_train(n_train, 1);

        for (size_t i = 0; i < n_train; ++i) {
            size_t orig_idx = indices[i];
            for (size_t j = 0; j < X_shape[1]; ++j) {
                X_train(i, j) = dataset.X(orig_idx, j);
            }
            Y_train(i, 0) = dataset.Y(orig_idx, 0);
        }

        Dataset result;
        result.X = std::move(X_train);
        result.Y = std::move(Y_train);
        result.feature_names = dataset.feature_names;
        result.label_names = dataset.label_names;

        return result;
    }

private:
    static std::vector<std::string> parse_header(const std::string& line, char delimiter) {
        std::vector<std::string> headers;
        std::stringstream ss(line);
        std::string item;

        while (std::getline(ss, item, delimiter)) {
            item.erase(0, item.find_first_not_of(" \t\r\n"));
            item.erase(item.find_last_not_of(" \t\r\n") + 1);
            headers.push_back(item);
        }

        return headers;
    }

    static std::vector<T> parse_row(const std::string& line, char delimiter) {
        std::vector<T> row;
        std::stringstream ss(line);
        std::string item;

        while (std::getline(ss, item, delimiter)) {
            try {
                if constexpr (std::is_same_v<T, float>) {
                    row.push_back(std::stof(item));
                } else if constexpr (std::is_same_v<T, double>) {
                    row.push_back(std::stod(item));
                } else {
                    row.push_back(static_cast<T>(std::stod(item)));
                }
            } catch (const std::exception&) {
                throw std::runtime_error("Invalid numeric value in CSV: " + item);
            }
        }

        return row;
    }
};

template<typename T>
class DatasetGenerator {
public:
    static typename CSVLoader<T>::Dataset create_xor_dataset() {
        utec::algebra::Tensor<T, 2> X(4, 2);
        utec::algebra::Tensor<T, 2> Y(4, 1);

        X(0, 0) = 0; X(0, 1) = 0; Y(0, 0) = 0;
        X(1, 0) = 0; X(1, 1) = 1; Y(1, 0) = 1;
        X(2, 0) = 1; X(2, 1) = 0; Y(2, 0) = 1;
        X(3, 0) = 1; X(3, 1) = 1; Y(3, 0) = 0;

        typename CSVLoader<T>::Dataset dataset;
        dataset.X = std::move(X);
        dataset.Y = std::move(Y);
        dataset.feature_names = {"input1", "input2"};
        dataset.label_names = {"output"};

        return dataset;
    }

    static typename CSVLoader<T>::Dataset create_circle_dataset(size_t n_samples = 100, T noise = 0.1) {
        std::mt19937 rng(42);
        std::uniform_real_distribution<T> uniform(-1.0, 1.0);
        std::normal_distribution<T> normal(0.0, noise);

        utec::algebra::Tensor<T, 2> X(n_samples, 2);
        utec::algebra::Tensor<T, 2> Y(n_samples, 1);

        for (size_t i = 0; i < n_samples; ++i) {
            T x = uniform(rng);
            T y = uniform(rng);
            T distance = std::sqrt(x*x + y*y);
            
             
            x += normal(rng);
            y += normal(rng);
            
            X(i, 0) = x;
            X(i, 1) = y;
            Y(i, 0) = (distance < 0.7) ? 1.0 : 0.0; 
        }

        typename CSVLoader<T>::Dataset dataset;
        dataset.X = std::move(X);
        dataset.Y = std::move(Y);
        dataset.feature_names = {"x", "y"};
        dataset.label_names = {"inside_circle"};

        return dataset;
    }

    static void save_dataset_to_csv(const typename CSVLoader<T>::Dataset& dataset, 
                                   const std::string& filename) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot create file: " + filename);
        }

        for (size_t i = 0; i < dataset.feature_names.size(); ++i) {
            file << dataset.feature_names[i];
            if (i < dataset.feature_names.size() - 1) file << ",";
        }
        for (const auto& label : dataset.label_names) {
            file << "," << label;
        }
        file << "\n";

        auto X_shape = dataset.X.shape();
        for (size_t i = 0; i < X_shape[0]; ++i) {
            for (size_t j = 0; j < X_shape[1]; ++j) {
                file << dataset.X(i, j);
                if (j < X_shape[1] - 1) file << ",";
            }
            file << "," << dataset.Y(i, 0) << "\n";
        }
    }
};

} // namespace utec::data

#endif // PROYECTO_FINAL_CSV_LOADER_H
