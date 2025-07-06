#include "../src/tensor.h"
#include <cassert>
#include <iostream>
#include <cmath>

using namespace utec::algebra;

void test_tensor_creation() {
    std::cout << "Testing tensor creation..." << std::endl;
    
    Tensor<float, 2> t(3, 4);
    auto shape = t.shape();
    assert(shape[0] == 3);
    assert(shape[1] == 4);
    
    Tensor<int, 3> t3(2, 3, 4);
    auto shape3 = t3.shape();
    assert(shape3[0] == 2);
    assert(shape3[1] == 3);
    assert(shape3[2] == 4);
    
    std::cout << "✓ Tensor creation tests passed" << std::endl;
}

void test_tensor_operations() {
    std::cout << "Testing tensor operations..." << std::endl;
    
    Tensor<float, 2> t(2, 2);
    t.fill(5.0f);
    
    assert(t(0, 0) == 5.0f);
    assert(t(1, 1) == 5.0f);
    
    t = {1, 2, 3, 4};
    assert(t(0, 0) == 1);
    assert(t(0, 1) == 2);
    assert(t(1, 0) == 3);
    assert(t(1, 1) == 4);
    
    Tensor<float, 2> t2(2, 2);
    t2 = {2, 2, 2, 2};
    
    auto result = t + t2;
    assert(result(0, 0) == 3);
    assert(result(1, 1) == 6);
    
    std::cout << "✓ Tensor operations tests passed" << std::endl;
}

void test_matrix_operations() {
    std::cout << "Testing matrix operations..." << std::endl;
    
    Tensor<float, 2> A(2, 3);
    A = {1, 2, 3, 4, 5, 6};
    
    Tensor<float, 2> B(3, 2);
    B = {1, 2, 3, 4, 5, 6};
    
    auto C = matrix_product(A, B);
    auto shape = C.shape();
    assert(shape[0] == 2);
    assert(shape[1] == 2);
    
    assert(C(0, 0) == 22); 
    assert(C(0, 1) == 28); 
    
    auto AT = transpose_2d(A);
    auto at_shape = AT.shape();
    assert(at_shape[0] == 3);
    assert(at_shape[1] == 2);
    assert(AT(0, 0) == 1);
    assert(AT(1, 0) == 2);
    assert(AT(2, 1) == 6);
    
    std::cout << "✓ Matrix operations tests passed" << std::endl;
}

void test_error_handling() {
    std::cout << "Testing error handling..." << std::endl;
    
    try {
        Tensor<int, 2> t(2, 2, 2); 
        assert(false); 
    } catch (const std::runtime_error& e) {
        // Expected
    }
    
    try {
        Tensor<int, 2> t(2, 2);
        t = {1, 2, 3};
        assert(false); 
    } catch (const std::runtime_error& e) {
    }
    
    std::cout << "✓ Error handling tests passed" << std::endl;
}

int main() {
    std::cout << "Running Tensor Tests..." << std::endl;
    std::cout << "========================" << std::endl;
    
    try {
        test_tensor_creation();
        test_tensor_operations();
        test_matrix_operations();
        test_error_handling();
        
        std::cout << std::endl;
        std::cout << "All tensor tests passed! ✓" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
