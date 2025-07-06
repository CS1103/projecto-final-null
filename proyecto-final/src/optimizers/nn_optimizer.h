//
// Created by rudri on 10/11/2020.
//

#ifndef PROYECTO_FINAL_NN_OPTIMIZER_H
#define PROYECTO_FINAL_NN_OPTIMIZER_H

#include "../layers/nn_interfaces.h"
#include <cmath>

namespace utec::neural_network {

template<typename T>
class SGD final : public IOptimizer<T> {
private:
    T learning_rate_;

public:
    explicit SGD(T learning_rate = T{0.01}) : learning_rate_(learning_rate) {}
    
    void update(Tensor<T,2>& params, const Tensor<T,2>& gradients) override {
        auto shape = params.shape();
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                params(i, j) -= learning_rate_ * gradients(i, j);
            }
        }
    }
};

template<typename T>
class Adam final : public IOptimizer<T> {
private:
    T learning_rate_;
    T beta1_;
    T beta2_;
    T epsilon_;
    int t_; 
    
    mutable Tensor<T,2> m_; 
    mutable Tensor<T,2> v_; 
    mutable bool initialized_;

public:
    explicit Adam(T learning_rate = T{0.001}, T beta1 = T{0.9}, T beta2 = T{0.999}, T epsilon = T{1e-8})
        : learning_rate_(learning_rate), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), 
          t_(0), initialized_(false) {}
    
    void update(Tensor<T,2>& params, const Tensor<T,2>& gradients) override {
        if (!initialized_) {
            auto shape = params.shape();
            m_ = Tensor<T,2>(shape[0], shape[1]);
            v_ = Tensor<T,2>(shape[0], shape[1]);
            m_.fill(T{0});
            v_.fill(T{0});
            initialized_ = true;
        }
        
        t_++;
        
        auto shape = params.shape();
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                m_(i, j) = beta1_ * m_(i, j) + (T{1} - beta1_) * gradients(i, j);
            }
        }
        
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                v_(i, j) = beta2_ * v_(i, j) + (T{1} - beta2_) * gradients(i, j) * gradients(i, j);
            }
        }
        
        T beta1_t = std::pow(beta1_, t_);
        T beta2_t = std::pow(beta2_, t_);
        
        for (size_t i = 0; i < shape[0]; ++i) {
            for (size_t j = 0; j < shape[1]; ++j) {
                T m_hat = m_(i, j) / (T{1} - beta1_t);
                T v_hat = v_(i, j) / (T{1} - beta2_t);
                params(i, j) -= learning_rate_ * m_hat / (std::sqrt(v_hat) + epsilon_);
            }
        }
    }
    
    void step() override {
    }
};

} // namespace utec::neural_network

#endif //PROYECTO_FINAL_NN_OPTIMIZER_H
