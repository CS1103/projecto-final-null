#include "../src/neural_network.h"
#include <cassert>
#include <iostream>
#include <cmath>

using namespace utec::algebra;
using namespace utec::neural_network;

template<typename T>
void simple_init(Tensor<T, 2>& tensor) {
    auto shape = tensor.shape();
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            tensor(i, j) = T{0.1};
        }
    }
}

template<typename T>
void zero_init(Tensor<T, 2>& tensor) {
    tensor.fill(T{0});
}

void test_activation_functions() {
    std::cout << "Testing activation functions..." << std::endl;
    
    ReLU<float> relu;
    Tensor<float, 2> input(2, 2);
    input = {-1, 2, 0, -3};
    
    auto relu_output = relu.forward(input);
    assert(relu_output(0, 0) == 0); 
    assert(relu_output(0, 1) == 2); 
    assert(relu_output(1, 0) == 0); 
    assert(relu_output(1, 1) == 0); 
    
    Tensor<float, 2> grad(2, 2);
    grad.fill(1.0f);
    auto relu_grad = relu.backward(grad);
    assert(relu_grad(0, 0) == 0); 
    assert(relu_grad(0, 1) == 1); 
    
    Sigmoid<float> sigmoid;
    Tensor<float, 2> sigmoid_input(1, 1);
    sigmoid_input = {0};
    
    auto sigmoid_output = sigmoid.forward(sigmoid_input);
    assert(std::abs(sigmoid_output(0, 0) - 0.5f) < 1e-6f); 
    
    std::cout << "✓ Activation functions tests passed" << std::endl;
}

void test_dense_layer() {
    std::cout << "Testing dense layer..." << std::endl;
    
    Dense<float> dense(2, 3, simple_init<float>, zero_init<float>);
    
    Tensor<float, 2> input(1, 2);
    input = {1, 2};
    
    auto output = dense.forward(input);
    auto shape = output.shape();
    assert(shape[0] == 1);
    assert(shape[1] == 3);
    
    for (size_t i = 0; i < shape[1]; ++i) {
        assert(std::abs(output(0, i) - 0.3f) < 1e-6f);
    }
    
    std::cout << "✓ Dense layer tests passed" << std::endl;
}

void test_loss_functions() {
    std::cout << "Testing loss functions..." << std::endl;
    
    Tensor<float, 2> pred(2, 1);
    pred = {1, 2};
    
    Tensor<float, 2> true_val(2, 1);
    true_val = {1.5, 1.5};
    
    MSELoss<float> mse_loss(pred, true_val);
    float loss = mse_loss.loss();
    
    assert(std::abs(loss - 0.25f) < 1e-6f);
    
    auto grad = mse_loss.loss_gradient();
    assert(std::abs(grad(0, 0) - (-1.0f)) < 1e-6f);
    assert(std::abs(grad(1, 0) - 1.0f) < 1e-6f);   
    
    std::cout << "✓ Loss functions tests passed" << std::endl;
}

void test_optimizers() {
    std::cout << "Testing optimizers..." << std::endl;
    
    // Test SGD
    SGD<float> sgd(0.1f);
    
    Tensor<float, 2> params(2, 2);
    params = {1, 2, 3, 4};
    
    Tensor<float, 2> grads(2, 2);
    grads = {0.1, 0.2, 0.3, 0.4};
    
    sgd.update(params, grads);
    
    assert(std::abs(params(0, 0) - 0.99f) < 1e-6f);
    assert(std::abs(params(0, 1) - 1.98f) < 1e-6f);
    assert(std::abs(params(1, 0) - 2.97f) < 1e-6f); 
    assert(std::abs(params(1, 1) - 3.96f) < 1e-6f); 
    
    std::cout << "✓ Optimizers tests passed" << std::endl;
}

void test_simple_network() {
    std::cout << "Testing simple network..." << std::endl;
    
    NeuralNetwork<float> network;
    
    network.add_layer(std::make_unique<Dense<float>>(2, 2, simple_init<float>, zero_init<float>));
    network.add_layer(std::make_unique<ReLU<float>>());
    network.add_layer(std::make_unique<Dense<float>>(2, 1, simple_init<float>, zero_init<float>));
    
    Tensor<float, 2> input(1, 2);
    input = {1, 1};
    
    auto output = network.predict(input);
    auto shape = output.shape();
    assert(shape[0] == 1);
    assert(shape[1] == 1);
    

    assert(std::abs(output(0, 0) - 0.04f) < 1e-6f);
    
    std::cout << "✓ Simple network tests passed" << std::endl;
}

int main() {
    std::cout << "Running Neural Network Tests..." << std::endl;
    std::cout << "===============================" << std::endl;
    
    try {
        test_activation_functions();
        test_dense_layer();
        test_loss_functions();
        test_optimizers();
        test_simple_network();
        
        std::cout << std::endl;
        std::cout << "All neural network tests passed! ✓" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
