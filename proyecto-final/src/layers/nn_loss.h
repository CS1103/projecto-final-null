//
// Created by rudri on 10/11/2020.
//

#ifndef PROYECTO_FINAL_NN_LOSS_H
#define PROYECTO_FINAL_NN_LOSS_H

#include "nn_interfaces.h"
#include <cmath>

namespace utec::neural_network {

template<typename T>
class MSELoss final : public ILoss<T, 2> {
private:
    Tensor<T,2> y_prediction_;
    Tensor<T,2> y_true_;

public:
    MSELoss(const Tensor<T,2>& y_prediction, const Tensor<T,2>& y_true) 
        : y_prediction_(y_prediction), y_true_(y_true) {}
    
    T loss() const override {
        T total_loss = T{0};
        size_t count = 0;
        
        auto shape = y_prediction_.shape();
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                T diff = y_prediction_(i, j) - y_true_(i, j);
                total_loss += diff * diff;
                count++;
            }
        }
        
        return total_loss / static_cast<T>(count);
    }
    
    Tensor<T,2> loss_gradient() const override {
        auto shape = y_prediction_.shape();
        Tensor<T,2> gradient(shape[0], shape[1]);
        T scale = T{2} / static_cast<T>(y_prediction_.size());
        
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                gradient(i, j) = scale * (y_prediction_(i, j) - y_true_(i, j));
            }
        }
        
        return gradient;
    }
};

template<typename T>
class BCELoss final : public ILoss<T, 2> {
private:
    Tensor<T,2> y_prediction_;
    Tensor<T,2> y_true_;
    const T epsilon_ = T{1e-15}; 
public:
    BCELoss(const Tensor<T,2>& y_prediction, const Tensor<T,2>& y_true) 
        : y_prediction_(y_prediction), y_true_(y_true) {}
    
    T loss() const override {
        T total_loss = T{0};
        size_t count = 0;
        
        auto shape = y_prediction_.shape();
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                T pred = std::max(epsilon_, std::min(T{1} - epsilon_, y_prediction_(i, j)));
                T true_val = y_true_(i, j);
                
                total_loss += -(true_val * std::log(pred) + (T{1} - true_val) * std::log(T{1} - pred));
                count++;
            }
        }
        
        return total_loss / static_cast<T>(count);
    }
    
    Tensor<T,2> loss_gradient() const override {
        auto shape = y_prediction_.shape();
        Tensor<T,2> gradient(shape[0], shape[1]);
        T scale = T{1} / static_cast<T>(y_prediction_.size());
        
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                T pred = std::max(epsilon_, std::min(T{1} - epsilon_, y_prediction_(i, j)));
                T true_val = y_true_(i, j);
                
                gradient(i, j) = scale * ((pred - true_val) / (pred * (T{1} - pred)));
            }
        }
        
        return gradient;
    }
};

} // namespace utec::neural_network

#endif //PROYECTO_FINAL_NN_LOSS_H
