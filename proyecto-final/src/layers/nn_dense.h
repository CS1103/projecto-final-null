//
// Created by rudri on 10/11/2020.
//

#ifndef PROYECTO_FINAL_NN_DENSE_H
#define PROYECTO_FINAL_NN_DENSE_H

#include "nn_interfaces.h"

namespace utec::neural_network {

template<typename T>
class Dense final : public ILayer<T> {
private:
    Tensor<T,2> weights_;
    Tensor<T,2> biases_;
    Tensor<T,2> last_input_;
    Tensor<T,2> weight_gradients_;
    Tensor<T,2> bias_gradients_;
    size_t in_features_;
    size_t out_features_;

public:
    template<typename InitWFun, typename InitBFun>
    Dense(size_t in_features, size_t out_features, InitWFun init_w_fun, InitBFun init_b_fun)
        : in_features_(in_features), out_features_(out_features) {
        
        weights_ = Tensor<T,2>(in_features, out_features);
        biases_ = Tensor<T,2>(1, out_features);
        
        weight_gradients_ = Tensor<T,2>(in_features, out_features);
        bias_gradients_ = Tensor<T,2>(1, out_features);
        
        init_w_fun(weights_);
        init_b_fun(biases_);
    }
    
    Tensor<T,2> forward(const Tensor<T,2>& x) override {
        last_input_ = x;      
        Tensor<T,2> output = x.matmul(weights_);
        
        const auto& output_shape = output.shape();
        for (size_t i = 0; i < output_shape[0]; ++i) {
            for (size_t j = 0; j < output_shape[1]; ++j) {
                output(i, j) += biases_(0, j);
            }
        }
        
        return output;
    }
    
    Tensor<T,2> backward(const Tensor<T,2>& output_gradients) override {
        weight_gradients_ = last_input_.transpose().matmul(output_gradients);
        
        const auto& grad_shape = output_gradients.shape();
        bias_gradients_.fill(T{0});
        
        for (size_t i = 0; i < grad_shape[0]; ++i) {
            for (size_t j = 0; j < grad_shape[1]; ++j) {
                bias_gradients_(0, j) += output_gradients(i, j);
            }
        }
        
        Tensor<T,2> input_gradients = output_gradients.matmul(weights_.transpose());
        
        return input_gradients;
    }
    
    void update_params(IOptimizer<T>& optimizer) override {
        optimizer.update(weights_, weight_gradients_);
        optimizer.update(biases_, bias_gradients_);
        optimizer.step();
    }
    
    const Tensor<T,2>& weights() const { return weights_; }
    const Tensor<T,2>& biases() const { return biases_; }
};

} // namespace utec::neural_network

#endif //PROYECTO_FINAL_NN_DENSE_H
