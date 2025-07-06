//
// Created by rudri on 10/11/2020.
//

#ifndef PROYECTO_FINAL_NEURAL_NETWORK_H
#define PROYECTO_FINAL_NEURAL_NETWORK_H

#include "layers/nn_interfaces.h"
#include "layers/nn_dense.h"
#include "layers/nn_activation.h"
#include "layers/nn_loss.h"
#include "optimizers/nn_optimizer.h"
#include "csv_loader.h"
#include <vector>
#include <memory>
#include <random>
#include <iostream>
#include <iomanip>
#include <fstream>

namespace utec::neural_network {

template<typename T>
class NeuralNetwork {
private:
    std::vector<std::unique_ptr<ILayer<T>>> layers_;

public:
    void add_layer(std::unique_ptr<ILayer<T>> layer) {
        layers_.push_back(std::move(layer));
    }
    
    template<template<typename> class LossType, 
             template<typename> class OptimizerType = SGD>
    void train(const Tensor<T,2>& X, const Tensor<T,2>& Y, 
               size_t epochs, size_t batch_size, T learning_rate, bool verbose = true) {
        
        OptimizerType<T> optimizer(learning_rate);
        
        const auto& x_shape = X.shape();
        const auto& y_shape = Y.shape();
        size_t num_samples = x_shape[0];
        
        if (verbose) {
            std::cout << "Training neural network...\n";
            std::cout << "Samples: " << num_samples << ", Epochs: " << epochs 
                      << ", Batch size: " << batch_size << ", Learning rate: " << learning_rate << "\n";
            std::cout << std::string(50, '-') << "\n";
        }
        
        for (size_t epoch = 0; epoch < epochs; ++epoch) {
            T total_loss = T{0};
            size_t num_batches = 0;
            
            for (size_t batch_start = 0; batch_start < num_samples; batch_start += batch_size) {
                size_t batch_end = std::min(batch_start + batch_size, num_samples);
                size_t actual_batch_size = batch_end - batch_start;
                
                Tensor<T,2> batch_x(actual_batch_size, x_shape[1]);
                Tensor<T,2> batch_y(actual_batch_size, y_shape[1]);
                
                for (size_t i = 0; i < actual_batch_size; ++i) {
                    for (size_t j = 0; j < x_shape[1]; ++j) {
                        batch_x(i, j) = X(batch_start + i, j);
                    }
                    for (size_t j = 0; j < y_shape[1]; ++j) {
                        batch_y(i, j) = Y(batch_start + i, j);
                    }
                }
                
                Tensor<T,2> output = predict(batch_x);
                
                LossType<T> loss_fn(output, batch_y);
                total_loss += loss_fn.loss();
                
                Tensor<T,2> gradients = loss_fn.loss_gradient();
                
                for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
                    gradients = layers_[i]->backward(gradients);
                    layers_[i]->update_params(optimizer);
                }
                
                num_batches++;
            }
            
            if (verbose && (epoch % (epochs/10 + 1) == 0 || epoch == epochs - 1)) {
                T avg_loss = total_loss / num_batches;
                std::cout << "Epoch " << std::setw(4) << epoch + 1 << "/" << epochs 
                          << " - Loss: " << std::fixed << std::setprecision(6) << avg_loss << "\n";
            }
        }
        
        if (verbose) {
            std::cout << "Training completed!\n\n";
        }
    }
    
    Tensor<T,2> predict(const Tensor<T,2>& X) {
        Tensor<T,2> output = X;
        
        for (auto& layer : layers_) {
            output = layer->forward(output);
        }
        
        return output;
    }
    
    T evaluate_accuracy(const Tensor<T,2>& X, const Tensor<T,2>& Y, T threshold = 0.5) {
        Tensor<T,2> predictions = predict(X);
        auto pred_shape = predictions.shape();
        
        size_t correct = 0;
        size_t total = pred_shape[0];
        
        for (size_t i = 0; i < total; ++i) {
            T pred = (predictions(i, 0) > threshold) ? T{1} : T{0};
            T actual = Y(i, 0);
            if (std::abs(pred - actual) < T{0.1}) {
                correct++;
            }
        }
        
        return static_cast<T>(correct) / static_cast<T>(total);
    }
    
    void save_model(const std::string& filename) {
        std::ofstream file(filename);
        file << "# Neural Network Model\n";
        file << "# Format: layer_type:weights:biases\n";
        
        for (size_t i = 0; i < layers_.size(); ++i) {
            file << "# Layer " << i << "\n";
            file << "dense:placeholder:placeholder\n";
        }
    }
};

template<typename T>
struct Initializers {
    static void xavier_init(Tensor<T,2>& tensor) {
        auto shape = tensor.shape();
        std::mt19937 rng(std::random_device{}());
        T limit = std::sqrt(T{6.0} / (shape[0] + shape[1]));
        std::uniform_real_distribution<T> dist(-limit, limit);
        
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                tensor(i, j) = dist(rng);
            }
        }
    }
    
    static void zero_init(Tensor<T,2>& tensor) {
        tensor.fill(T{0});
    }
    
    static void normal_init(Tensor<T,2>& tensor, T mean = T{0}, T std = T{0.1}) {
        std::mt19937 rng(std::random_device{}());
        std::normal_distribution<T> dist(mean, std);
        
        auto shape = tensor.shape();
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                tensor(i, j) = dist(rng);
            }
        }
    }
};

} // namespace utec::neural_network

#endif //PROYECTO_FINAL_NEURAL_NETWORK_H
