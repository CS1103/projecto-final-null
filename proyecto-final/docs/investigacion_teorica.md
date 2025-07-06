# Investigación Teórica - Redes Neuronales

## **CS2013 Programación III** · Proyecto Final

---

## 1. Fundamentos Matemáticos

### 1.1 Álgebra Lineal

Las redes neuronales dependen fuertemente del álgebra lineal para sus operaciones fundamentales:

#### Multiplicación Matricial
```
Z = X * W + b
```
Donde:
- `X`: Matriz de entrada (batch_size × input_features)
- `W`: Matriz de pesos (input_features × output_features)  
- `b`: Vector de bias (1 × output_features)
- `Z`: Salida lineal (batch_size × output_features)

**Implementación en nuestro proyecto:**
```cpp
template <typename T>
Tensor<T, 2> matrix_product(const Tensor<T, 2>& A, const Tensor<T, 2>& B) {
    // Validación de dimensiones
    if (shapeA[1] != shapeB[0]) {
        throw std::runtime_error("Matrix dimensions are incompatible");
    }
    
    // Multiplicación optimizada O(n³)
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < p; ++j) {
            T sum{};
            for (size_t k = 0; k < n; ++k) {
                sum += A(i, k) * B(k, j);
            }
            result(i, j) = sum;
        }
    }
}
```

#### Broadcasting
Permite operaciones entre tensores de diferentes dimensiones siguiendo reglas específicas:
- Dimensiones se alinean desde la derecha
- Dimensiones de tamaño 1 se expanden automáticamente
- Dimensiones faltantes se tratan como tamaño 1

### 1.2 Cálculo Diferencial

#### Gradientes y Derivadas Parciales

Para una función de pérdida `L(θ)`, el gradiente es:
```
∇L = [∂L/∂θ₁, ∂L/∂θ₂, ..., ∂L/∂θₙ]
```

#### Regla de la Cadena
Fundamental para backpropagation:
```
∂L/∂x = (∂L/∂y) × (∂y/∂x)
```

**Ejemplo en una capa densa:**
```cpp
// Forward: y = f(Wx + b)
// Backward: ∂L/∂W = x^T × ∂L/∂y
Tensor<T,2> weight_gradients = last_input_.transpose().matmul(output_gradients);

// ∂L/∂x = ∂L/∂y × W^T  
Tensor<T,2> input_gradients = output_gradients.matmul(weights_.transpose());
```

---

## 2. Arquitecturas de Redes Neuronales

### 2.1 Perceptrón Multicapa (MLP)

#### Estructura:
- **Capa de entrada**: Recibe los datos de entrada
- **Capas ocultas**: Transforman los datos mediante funciones no lineales
- **Capa de salida**: Produce la predicción final

#### Representación matemática:
```
h₁ = f₁(W₁x + b₁)
h₂ = f₂(W₂h₁ + b₂)
...
y = fₙ(Wₙhₙ₋₁ + bₙ)
```

**Implementación en nuestro proyecto:**
```cpp
class Dense final : public ILayer<T> {
    Tensor<T,2> forward(const Tensor<T,2>& x) override {
        // Multiplicación matricial + bias
        Tensor<T,2> output = x.matmul(weights_);
        // Agregar bias a cada muestra
        for (size_t i = 0; i < batch_size; ++i) {
            for (size_t j = 0; j < output_features; ++j) {
                output(i, j) += biases_(0, j);
            }
        }
        return output;
    }
};
```

### 2.2 Funciones de Activación

#### ReLU (Rectified Linear Unit)
```
f(x) = max(0, x)
f'(x) = 1 if x > 0, else 0
```

**Ventajas:**
- Computacionalmente eficiente
- Mitiga el problema del gradiente que desaparece
- Introduce no-linealidad sparse

**Implementación:**
```cpp
Tensor<T,2> forward(const Tensor<T,2>& x) override {
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            result(i, j) = std::max(T{0}, x(i, j));
        }
    }
    return result;
}
```

#### Sigmoid
```
f(x) = 1 / (1 + e^(-x))
f'(x) = f(x) × (1 - f(x))
```

**Características:**
- Salida entre [0, 1]
- Diferenciable en todo punto
- Útil para clasificación binaria

---

## 3. Algoritmos de Entrenamiento

### 3.1 Forward Propagation

**Proceso:**
1. Los datos fluyen desde la entrada hacia la salida
2. Cada capa aplica su transformación: `output = activation(weights × input + bias)`
3. Se calcula la predicción final

```cpp
Tensor<T,2> predict(const Tensor<T,2>& X) {
    Tensor<T,2> output = X;
    for (auto& layer : layers_) {
        output = layer->forward(output);  // Propagación hacia adelante
    }
    return output;
}
```

### 3.2 Backpropagation

**Algoritmo:**
1. Calcular error en la salida: `δₗ = ∇L × f'(zₗ)`
2. Propagar error hacia atrás: `δₗ₋₁ = (Wₗ^T × δₗ) ⊙ f'(zₗ₋₁)`
3. Calcular gradientes: `∇W = δₗ × aₗ₋₁^T`, `∇b = δₗ`

**Implementación:**
```cpp
// Calcular gradientes de la función de pérdida
LossType<T> loss_fn(output, batch_y);
Tensor<T,2> gradients = loss_fn.loss_gradient();

// Propagar gradientes hacia atrás
for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
    gradients = layers_[i]->backward(gradients);
    layers_[i]->update_params(optimizer);
}
```

### 3.3 Optimizadores

#### Stochastic Gradient Descent (SGD)
```
θₜ₊₁ = θₜ - η × ∇L(θₜ)
```

**Implementación:**
```cpp
void update(Tensor<T,2>& params, const Tensor<T,2>& gradients) override {
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            params(i, j) -= learning_rate_ * gradients(i, j);
        }
    }
}
```

#### Adam Optimizer
Combina momentum y adaptive learning rate:
```
mₜ = β₁ × mₜ₋₁ + (1-β₁) × ∇L
vₜ = β₂ × vₜ₋₁ + (1-β₂) × (∇L)²
m̂ₜ = mₜ / (1-β₁ᵗ)
v̂ₜ = vₜ / (1-β₂ᵗ)
θₜ₊₁ = θₜ - η × m̂ₜ / (√v̂ₜ + ε)
```

**Ventajas:**
- Convergencia más rápida
- Adaptativo a cada parámetro
- Robusto a hiperparámetros

---

## 4. Funciones de Pérdida

### 4.1 Mean Squared Error (MSE)
Para problemas de regresión:
```
MSE = (1/n) × Σ(yᵢ - ŷᵢ)²
∇MSE = (2/n) × (ŷ - y)
```

### 4.2 Binary Cross-Entropy (BCE)
Para clasificación binaria:
```
BCE = -(1/n) × Σ[yᵢ×log(ŷᵢ) + (1-yᵢ)×log(1-ŷᵢ)]
∇BCE = (1/n) × (ŷ - y) / [ŷ × (1-ŷ)]
```

---

## 5. Consideraciones de Implementación en C++

### 5.1 Templates para Flexibilidad de Tipos
```cpp
template<typename T, size_t N>
class Tensor {
    // Soporte para float, double, int, etc.
    // Dimensionalidad fija en tiempo de compilación
};
```

### 5.2 Gestión de Memoria
- **RAII**: Destrucción automática de recursos
- **std::vector**: Gestión dinámica segura
- **std::unique_ptr**: Ownership claro de capas

### 5.3 Patrones de Diseño

#### Strategy Pattern
```cpp
template<typename T>
struct IOptimizer {
    virtual void update(Tensor<T,2>& params, const Tensor<T,2>& gradients) = 0;
};
```

#### Template Method Pattern
```cpp
template<typename T>
struct ILayer {
    virtual Tensor<T,2> forward(const Tensor<T,2>& x) = 0;
    virtual Tensor<T,2> backward(const Tensor<T,2>& gradients) = 0;
};
```

---

## 6. Comparación con Frameworks Existentes

### TensorFlow/PyTorch vs Nuestra Implementación

| Aspecto | TensorFlow/PyTorch | Nuestro Proyecto |
|---------|-------------------|------------------|
| **Performance** | GPU optimizado, CUDA | CPU, single-thread |
| **Escalabilidad** | Grandes datasets | Datasets pequeños-medianos |
| **Flexibilidad** | APIs complejas | Implementación educativa |
| **Comprensión** | Black-box para principiantes | Transparencia total |
| **Producción** | Listo para producción | Propósito académico |

### Ventajas de Implementación Propia

1. **Comprensión profunda**: Conocimiento completo de cada operación
2. **Control total**: Sin dependencias externas complejas
3. **Optimización específica**: Adaptado a casos de uso específicos
4. **Aprendizaje**: Dominio de conceptos fundamentales

---

## 7. Limitaciones y Mejoras Futuras

### Limitaciones Actuales
- **Sin paralelización**: Single-threaded
- **Sin optimización BLAS**: Multiplicaciones matriciales básicas
- **Tipos de capas limitados**: Solo Dense y activaciones básicas
- **Sin regularización**: L1/L2, Dropout, BatchNorm

### Mejoras Propuestas

1. **Optimización de rendimiento:**
   ```cpp
   // Integración con OpenBLAS
   cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
               m, n, k, 1.0, A.data(), k, B.data(), n, 0.0, C.data(), n);
   ```

2. **Paralelización:**
   ```cpp
   #pragma omp parallel for
   for (size_t i = 0; i < batch_size; ++i) {
       // Procesamiento paralelo de muestras
   }
   ```

3. **Más tipos de capas:**
   - Convolucionales (CNN)
   - Recurrentes (RNN, LSTM)
   - Normalization (BatchNorm, LayerNorm)

---

## 8. Conclusiones Teóricas

### Aprendizajes Clave
1. **Matemáticas fundamentales**: Las redes neuronales son álgebra lineal + cálculo
2. **Importancia del diseño**: Arquitectura modular permite extensibilidad
3. **Trade-offs de optimización**: Precisión vs velocidad vs memoria
4. **Complejidad algorítmica**: O(n³) para multiplicaciones, O(n) para activaciones

### Impacto Educativo
- **Base sólida** para entender frameworks modernos
- **Apreciación de optimizaciones** en librerías profesionales
- **Comprensión de limitaciones** computacionales
- **Preparación para investigación** en deep learning

---

## Referencias Bibliográficas

[1] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

[2] Nielsen, M. (2015). *Neural Networks and Deep Learning*. Determination Press.

[3] Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature*, 323(6088), 533-536.

[4] Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. *arXiv preprint arXiv:1412.6980*.

[5] LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*, 521(7553), 436-444.

[6] Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks. *Proceedings of the thirteenth international conference on artificial intelligence and statistics*, 249-256.
