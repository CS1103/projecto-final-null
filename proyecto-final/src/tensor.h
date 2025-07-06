#ifndef PROYECTO_FINAL_TENSOR_H
#define PROYECTO_FINAL_TENSOR_H

#pragma once

#include <array>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <initializer_list>
#include <numeric>
#include <functional>
#include <type_traits>

namespace utec::algebra {

template <typename T, size_t N>
class Tensor {
public:
    Tensor() : total_size_(0) {
        dimensions_.fill(0);
    }

    template <typename... Args>
    Tensor(Args... dims) {
        if (sizeof...(Args) != N) {
            throw std::runtime_error("Number of dimensions do not match with " + std::to_string(N) + 
                                    ". Got " + std::to_string(sizeof...(Args)) + " arguments");
        }
        std::array<size_t, N> dims_array{ static_cast<size_t>(dims)... };
        for (size_t i = 0; i < N; ++i) {
            dimensions_[i] = dims_array[i];
        }
        total_size_ = std::accumulate(dimensions_.begin(), dimensions_.end(), size_t{1}, std::multiplies<>());
        data_.resize(total_size_);
    }

    const std::array<size_t, N>& shape() const {
        return dimensions_;
    }

    void fill(const T& value) {
        std::fill(data_.begin(), data_.end(), value);
    }

    Tensor<T, N>& operator=(std::initializer_list<T> list) {
        if (list.size() != total_size_) {
            throw std::runtime_error("Data size does not match tensor size");
        }
        std::copy(list.begin(), list.end(), data_.begin());
        return *this;
    }

    template <typename... Args>
    void reshape(Args... new_dims) {
        constexpr size_t arg_count = sizeof...(Args);
        if (arg_count != N) {
            throw std::runtime_error("Number of dimensions do not match with " + std::to_string(N));
        }

        std::array<size_t, N> new_shape;
        size_t idx = 0;
        for (auto val : std::initializer_list<size_t>{static_cast<size_t>(new_dims)...}) {
            new_shape[idx++] = val;
        }

        size_t new_total = std::accumulate(new_shape.begin(), new_shape.end(), size_t{1}, std::multiplies<>());
        if (new_total > total_size_) {
            data_.resize(new_total, T{});
        }
        dimensions_ = new_shape;
        total_size_ = new_total;
    }

    template <typename... Indices>
    T& operator()(Indices... idxs) {
        static_assert(sizeof...(idxs) == N, "Número incorrecto de índices.");
        std::array<size_t, N> indices{ static_cast<size_t>(idxs)... };
        return data_[get_flat_index(indices)];
    }

    template <typename... Indices>
    const T& operator()(Indices... idxs) const {
        static_assert(sizeof...(idxs) == N, "Número incorrecto de índices.");
        std::array<size_t, N> indices{ static_cast<size_t>(idxs)... };
        return data_[get_flat_index(indices)];
    }

    Tensor<T, N> operator+(const Tensor<T, N>& other) const {
        return broadcast_operation(other, std::plus<>());
    }

    Tensor<T, N> operator-(const Tensor<T, N>& other) const {
        return broadcast_operation(other, std::minus<>());
    }

    Tensor<T, N> operator*(const Tensor<T, N>& other) const {
        return broadcast_operation(other, std::multiplies<>());
    }

    Tensor<T, N> operator+(const T& scalar) const {
        Tensor<T, N> result = *this;
        for (auto& val : result.data_) val += scalar;
        return result;
    }

    Tensor<T, N> operator-(const T& scalar) const {
        Tensor<T, N> result = *this;
        for (auto& val : result.data_) val -= scalar;
        return result;
    }

    Tensor<T, N> operator*(const T& scalar) const {
        Tensor<T, N> result = *this;
        for (auto& val : result.data_) val *= scalar;
        return result;
    }

    Tensor<T, N> operator/(const T& scalar) const {
        Tensor<T, N> result = *this;
        for (auto& val : result.data_) val /= scalar;
        return result;
    }

    auto begin() { return data_.begin(); }
    auto end() { return data_.end(); }
    auto begin() const { return data_.begin(); }
    auto end() const { return data_.end(); }
    auto cbegin() const { return data_.cbegin(); }
    auto cend() const { return data_.cend(); }

    // Método para compatibilidad con neural network
    size_t size() const { return total_size_; }
    
    // Acceso por índice lineal (para compatibilidad)
    T& operator[](size_t index) { return data_[index]; }
    const T& operator[](size_t index) const { return data_[index]; }
    
    // Multiplicación matricial (solo para tensores 2D)
    template<size_t M = N>
    typename std::enable_if<M == 2, Tensor<T, 2>>::type 
    matmul(const Tensor<T, 2>& other) const {
        return matrix_product(*this, other);
    }
    
    // Transposición (solo para tensores 2D)
    template<size_t M = N>
    typename std::enable_if<M == 2, Tensor<T, 2>>::type 
    transpose() const {
        return transpose_2d(*this);
    }

    friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        if constexpr (N == 1) {
            for (size_t i = 0; i < t.total_size_; ++i) {
                os << t.data_[i] << " ";
            }
            os << "\n";
        } else if constexpr (N == 2) {
            size_t rows = t.dimensions_[0];
            size_t cols = t.dimensions_[1];
            os << "{\n";
            if (rows == 1) {
                for (size_t j = 0; j < cols; ++j) {
                    os << t.data_[j] << " ";
                }
                os << "\n";
            } else {
                for (size_t i = 0; i < rows; ++i) {
                    for (size_t j = 0; j < cols; ++j) {
                        os << t.data_[i * cols + j] << " ";
                    }
                    os << "\n";
                }
            }
            os << "}\n";
        } else if constexpr (N == 3) {
            size_t d0 = t.dimensions_[0];
            size_t d1 = t.dimensions_[1];
            size_t d2 = t.dimensions_[2];
            os << "{\n";
            size_t idx = 0;
            for (size_t i = 0; i < d0; ++i) {
                os << "{\n";
                for (size_t j = 0; j < d1; ++j) {
                    for (size_t k = 0; k < d2; ++k) {
                        os << t.data_[idx++] << " ";
                    }
                    os << "\n";
                }
                os << "}\n";
            }
            os << "}\n";
        } else {
            std::array<size_t, N> indices{};
            t.print_recursive(os, 0, indices);
        }
        return os;
    }

private:
    std::array<size_t, N> dimensions_{};
    std::vector<T> data_;
    size_t total_size_ = 0;

    size_t get_flat_index(const std::array<size_t, N>& indices) const {
        size_t index = 0;
        size_t stride = 1;
        for (int i = N - 1; i >= 0; --i) {
            index += indices[i] * stride;
            stride *= dimensions_[i];
        }
        return index;
    }

    bool is_broadcast_compatible(const std::array<size_t, N>& other_shape) const {
        for (size_t i = 0; i < N; ++i) {
            if (dimensions_[i] != other_shape[i] && dimensions_[i] != 1 && other_shape[i] != 1) {
                return false;
            }
        }
        return true;
    }

    size_t get_broadcast_index(const std::array<size_t, N>& index, const std::array<size_t, N>& shape) const {
        size_t flat = 0, stride = 1;
        for (int i = N - 1; i >= 0; --i) {
            size_t idx = (shape[i] == 1) ? 0 : index[i];
            flat += idx * stride;
            stride *= shape[i];
        }
        return flat;
    }

    template <typename BinaryOp>
    Tensor<T, N> broadcast_operation(const Tensor<T, N>& other, BinaryOp op) const {
        if (!is_broadcast_compatible(other.shape()) && !other.is_broadcast_compatible(this->shape())) {
            throw std::runtime_error("Shapes do not match and they are not compatible for broadcasting");
        }

        Tensor<T, N> result = *this;
        std::array<size_t, N> index{};
        auto shape = result.shape();

        for (size_t i = 0; i < result.data_.size(); ++i) {
            size_t temp = i;
            for (int d = N - 1; d >= 0; --d) {
                index[d] = temp % shape[d];
                temp /= shape[d];
            }

            size_t idx1 = get_broadcast_index(index, this->shape());
            size_t idx2 = get_broadcast_index(index, other.shape());
            result.data_[i] = op(this->data_[idx1], other.data_[idx2]);
        }

        return result;
    }

    void print_recursive(std::ostream& os, size_t dim, std::array<size_t, N>& indices) const {
        if (dim == N - 1) {
            os << "{ ";
            for (size_t i = 0; i < dimensions_[dim]; ++i) {
                indices[dim] = i;
                os << data_[get_flat_index(indices)] << " ";
            }
            os << "}";
        } else {
            os << "{\n";
            for (size_t i = 0; i < dimensions_[dim]; ++i) {
                indices[dim] = i;
                print_recursive(os, dim + 1, indices);
                if (i < dimensions_[dim] - 1) os << ",\n";
            }
            os << "\n}";
        }
    }
};

template <typename T, size_t N>
Tensor<T, N> operator+(const T& scalar, const Tensor<T, N>& tensor) {
    return tensor + scalar;
}

template <typename T>
Tensor<T, 2> transpose_2d(const Tensor<T, 2>& input) {
    const auto& shape = input.shape();
    Tensor<T, 2> result(shape[1], shape[0]);

    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            result(j, i) = input(i, j);
        }
    }
    return result;
}

template <typename T, size_t N>
Tensor<T, N> transpose_2d(const Tensor<T, N>& t) {
    static_assert(N >= 2, "Tensor must have at least 2 dimensions");

    std::array<size_t, N> new_dims = t.shape();
    std::swap(new_dims[N - 2], new_dims[N - 1]);

    Tensor<T, N> result(new_dims);

    std::array<size_t, N> idx;
    for (size_t i = 0; i < t.total_size_; ++i) {
        size_t remaining = i;
        for (int d = N - 1; d >= 0; --d) {
            idx[d] = remaining % t.shape()[d];
            remaining /= t.shape()[d];
        }

        std::array<size_t, N> new_idx = idx;
        std::swap(new_idx[N - 2], new_idx[N - 1]);
        result(new_idx) = t(idx);
    }

    return result;
}

template <typename T>
Tensor<T, 2> matrix_product(const Tensor<T, 2>& A, const Tensor<T, 2>& B) {
    const auto& shapeA = A.shape();
    const auto& shapeB = B.shape();

    if (shapeA[1] != shapeB[0]) {
        throw std::runtime_error("Matrix dimensions are incompatible for multiplication");
    }

    size_t m = shapeA[0], n = shapeA[1], p = shapeB[1];
    Tensor<T, 2> result(m, p);

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j) {
            T sum{};
            for (size_t k = 0; k < n; ++k) {
                sum += A(i, k) * B(k, j);
            }
            result(i, j) = sum;
        }
    }

    return result;
}

template <typename T>
Tensor<T, 3> matrix_product(const Tensor<T, 3>& A, const Tensor<T, 3>& B) {
    const auto& shapeA = A.shape();
    const auto& shapeB = B.shape();

    size_t B1 = shapeA[0], M = shapeA[1], N1 = shapeA[2];
    size_t B2 = shapeB[0], N2 = shapeB[1], P = shapeB[2];

    if (N1 != N2) {
        throw std::runtime_error("Matrix dimensions are incompatible for multiplication");
    }
    if (B1 != B2) {
        throw std::runtime_error("Matrix dimensions are compatible for multiplication BUT Batch dimensions do not match");
    }

    Tensor<T, 3> result(B1, M, P);
    for (size_t b = 0; b < B1; ++b) {
        for (size_t i = 0; i < M; ++i) {
            for (size_t j = 0; j < P; ++j) {
                T sum{};
                for (size_t k = 0; k < N1; ++k) {
                    sum += A(b, i, k) * B(b, k, j);
                }
                result(b, i, j) = sum;
            }
        }
    }

    return result;
}

}  // namespace utec::algebra

#endif // PROYECTO_FINAL_TENSOR_H
