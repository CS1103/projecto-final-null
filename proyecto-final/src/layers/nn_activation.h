//
// Created by rudri on 10/11/2020.
//

#ifndef PROYECTO_FINAL_NN_ACTIVATION_H
#define PROYECTO_FINAL_NN_ACTIVATION_H

#include "nn_interfaces.h"
#include <cmath>

namespace utec::neural_network {

template<typename T>
class ReLU final : public ILayer<T> {
private:
    Tensor<T,2> last_input_;

public:
    Tensor<T,2> forward(const Tensor<T,2>& x) override {
        last_input_ = x;
        auto shape = x.shape();
        Tensor<T,2> result(shape[0], shape[1]);
        
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                result(i, j) = std::max(T{0}, x(i, j));
            }
        }
        
        return result;
    }
    
    Tensor<T,2> backward(const Tensor<T,2>& gradients) override {
        auto shape = last_input_.shape();
        Tensor<T,2> result(shape[0], shape[1]);
        
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                result(i, j) = (last_input_(i, j) > T{0}) ? gradients(i, j) : T{0};
            }
        }
        
        return result;
    }
};

template<typename T>
class Sigmoid final : public ILayer<T> {
private:
    Tensor<T,2> last_output_;

public:
    Tensor<T,2> forward(const Tensor<T,2>& x) override {
        auto shape = x.shape();
        Tensor<T,2> result(shape[0], shape[1]);
        
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                T val = x(i, j);
                if (val > T{700}) val = T{700};
                if (val < T{-700}) val = T{-700};
                
                result(i, j) = T{1} / (T{1} + std::exp(-val));
            }
        }
        
        last_output_ = result;
        return result;
    }
    
    Tensor<T,2> backward(const Tensor<T,2>& gradients) override {
        auto shape = last_output_.shape();
        Tensor<T,2> result(shape[0], shape[1]);
        
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                T sigmoid_val = last_output_(i, j);
                result(i, j) = gradients(i, j) * sigmoid_val * (T{1} - sigmoid_val);
            }
        }
        
        return result;
    }
};

} // namespace utec::neural_network

#endif //PROYECTO_FINAL_NN_ACTIVATION_H
