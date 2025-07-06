# Manual de Usuario - Red Neuronal
## Proyecto Final CS2013 Programación III

## Descripción
Este proyecto implementa una **red neuronal multicapa completa desde cero** en C++, utilizando una librería de tensores personalizada y un sistema de carga/procesamiento de datos CSV para casos de uso reales.

## 🚀 Características Principales

### 🧮 Librería de Tensores Avanzada
- **Soporte completo para tensores N-dimensionales**
- **Operaciones matemáticas optimizadas** (+, -, *, /)
- **Multiplicación matricial eficiente** con validación de dimensiones
- **Transposición y reshape** de tensores
- **Broadcasting inteligente** para operaciones elemento a elemento
- **Indexación multidimensional** con verificación de bounds
- **Inicialización Xavier/He** para pesos de red neuronal

### 🧠 Red Neuronal Completa
- **Capas densas (fully connected)** con inicialización configurable
- **Funciones de activación**: ReLU, Sigmoid, Tanh
- **Funciones de pérdida**: MSE (Mean Squared Error), BCE (Binary Cross-Entropy)
- **Optimizadores avanzados**: SGD, Adam con momentum
- **Entrenamiento por lotes** con progreso en tiempo real
- **Evaluación de precisión** automática
- **Soporte para datasets personalizados**

### 📊 Sistema CSV Integrado
- **Carga automática de datasets** desde archivos CSV
- **Generación de predicciones** en formato CSV
- **Comparación detallada** entre valores esperados y predicciones
- **Workflow completo**: Input → Entrenamiento → Output → Análisis

## 🛠️ Instalación y Compilación

### 📋 Requisitos del Sistema
- **CMake 3.18 o superior**
- **Compilador C++17 compatible:**
  - GCC 11+ (Linux/Windows)
  - MSVC 2019+ (Windows)
  - Clang 12+ (macOS/Linux)
- **Git** (para clonar el repositorio)

### 🪟 Compilación en Windows (Recomendado)

#### Paso 1: Preparar el entorno
```powershell
# Navegar al directorio del proyecto
cd e:\projecto-final-null\proyecto-final

# Crear directorio de build (si no existe)
mkdir build
cd build
```

#### Paso 2: Configurar con CMake
```powershell
# Configurar el proyecto
cmake ..
```

#### Paso 3: Compilar
```powershell
# Compilar en modo Debug (recomendado para desarrollo)
cmake --build . --config Debug

# O compilar en modo Release (optimizado)
cmake --build . --config Release
```

#### Paso 4: Ejecutar el programa
```powershell
# Ejecutar el demo principal (desde directorio build)
.\bin\Debug\neural_net_demo.exe

# O en modo Release
.\bin\Release\neural_net_demo.exe
```

### 🐧 Compilación en Linux/macOS

```bash
# Navegar al proyecto
cd proyecto-final

# Crear y entrar al directorio build
mkdir build && cd build

# Configurar y compilar
cmake ..
make -j$(nproc)

# Ejecutar
./bin/neural_net_demo
```

## 🎮 Demos Incluidos

El programa incluye **4 demos completos** que demuestran todas las capacidades del sistema:

### 1️⃣ **Demo: Tensor Operations**
- Demuestra operaciones básicas con tensores
- Multiplicación matricial, transposición
- Operaciones elemento a elemento
- **Propósito**: Verificar que la librería de tensores funciona correctamente

### 2️⃣ **Demo: XOR Problem (usando CSV)**
- Resuelve el problema XOR clásico
- **Carga datos desde `input_data.csv`**
- Entrena red neuronal 2→4→1 con **1000 épocas**
- **Precisión esperada**: ~100%
- **Tiempo**: ~80-100ms

### 3️⃣ **Demo: Circle Classification (usando CSV)**
- Clasificación binaria con el mismo dataset CSV
- Red neuronal 2→4→1 con **1000 épocas**
- Muestra progreso de entrenamiento detallado
- **Propósito**: Demostrar versatilidad del mismo dataset

### 4️⃣ **Demo: CSV Workflow Completo**
- **Flujo realista completo**: Input → Entrenamiento → Output
- **Lee**: `input_data.csv` (datos de entrada)
- **Entrena**: Red neuronal 2→6→1 con **1000 épocas**
- **Genera**:
  - `output_predictions.csv` (predicciones)
  - `comparison_results.csv` (comparación detallada)
- **Evalúa**: Precisión del modelo

## 📁 Archivos CSV del Sistema

### 📥 **Input: `input_data.csv`**
```csv
feature_1,feature_2,label
0.1,0.2,1
0.8,0.9,0
0.3,0.1,1
...
```
- **Ubicación**: Raíz del proyecto
- **Formato**: 10 muestras, 2 características, 1 etiqueta binaria
- **Usado por**: Todos los demos (datasets unificado)

### 📤 **Output: `output_predictions.csv`**
```csv
sample_id,prediction
sample_0,0.954554
sample_1,0.00557967
sample_2,0.980913
...
```
- **Ubicación**: Raíz del proyecto
- **Contenido**: Predicciones reales de la red neuronal entrenada

### 📊 **Analysis: `comparison_results.csv`**
```csv
sample_id,feature_1,feature_2,expected,predicted,difference,correct
sample_0,0.1,0.2,1.0,0.954554,0.045446,1
sample_1,0.8,0.9,0.0,0.00557967,0.00557967,1
...
```
- **Ubicación**: Raíz del proyecto
- **Contenido**: Análisis detallado de predicciones vs valores esperados

## 💻 Ejemplos de Código

### 1️⃣ **Crear y Manipular Tensores**

```cpp
#include "src/tensor.h"
using namespace utec::algebra;

// Crear tensor 2D (matriz 3x3)
Tensor<float, 2> matrix(3, 3);

// Inicializar con valores
matrix = {1, 2, 3, 4, 5, 6, 7, 8, 9};

// Operaciones básicas
auto doubled = matrix * 2.0f;              // Multiplicación por escalar
auto transposed = matrix.transpose();       // Transposición
auto sum_result = matrix + doubled;         // Suma elemento a elemento

// Multiplicación matricial
Tensor<float, 2> other(3, 2);
other = {1, 2, 3, 4, 5, 6};
auto product = matrix.matmul(other);        // Resultado: 3x2

// Acceso a elementos
float value = matrix(1, 2);                 // Fila 1, Columna 2
std::cout << "Matrix:\n" << matrix;         // Impresión formateada
```

### 2️⃣ **Crear Red Neuronal Completa**

```cpp
#include "src/neural_network.h"
using namespace utec::neural_network;
using namespace utec::data;

// Crear red neuronal
NeuralNetwork<float> network;

// Agregar capas con inicialización Xavier
network.add_layer(std::make_unique<Dense<float>>(
    2, 4,                                    // 2 entradas → 4 neuronas
    Initializers<float>::xavier_init,        // Pesos Xavier
    Initializers<float>::zero_init           // Bias en cero
));
network.add_layer(std::make_unique<ReLU<float>>());
network.add_layer(std::make_unique<Dense<float>>(4, 1, 
    Initializers<float>::xavier_init, 
    Initializers<float>::zero_init
));
network.add_layer(std::make_unique<Sigmoid<float>>());

// Datos de entrenamiento (problema XOR)
Tensor<float, 2> X(4, 2);
X = {0, 0,    // XOR: [0,0] → 0
     0, 1,    //      [0,1] → 1  
     1, 0,    //      [1,0] → 1
     1, 1};   //      [1,1] → 0

Tensor<float, 2> Y(4, 1);
Y = {0, 1, 1, 0};

// Entrenar la red
network.train<MSELoss, SGD>(
    X, Y,           // Datos de entrada y salida
    1000,           // Épocas
    4,              // Tamaño de lote
    0.1f,           // Tasa de aprendizaje
    true            // Mostrar progreso
);

// Hacer predicciones
auto predictions = network.predict(X);
std::cout << "Predicciones:\n" << predictions;

// Evaluar precisión
float accuracy = network.evaluate_accuracy(X, Y);
std::cout << "Precisión: " << (accuracy * 100) << "%\n";
```

### 3️⃣ **Cargar Datos desde CSV**

```cpp
#include "src/csv_loader.h"
using namespace utec::data;

// Cargar dataset desde CSV
auto dataset = CSVLoader<float>::load_csv(
    "input_data.csv",    // Archivo
    true,                // Tiene header
    ',',                 // Delimitador
    -1                   // Última columna como label
);

std::cout << "Muestras: " << dataset.X.shape()[0] << std::endl;
std::cout << "Características: " << dataset.X.shape()[1] << std::endl;

// Entrenar con datos CSV
NeuralNetwork<float> network;
// ... configurar capas ...
network.train<BCELoss, SGD>(dataset.X, dataset.Y, 1000, 10, 0.01f);

// Generar predicciones
auto predictions = network.predict(dataset.X);

// Guardar resultados
CSVLoader<float>::save_predictions("output.csv", predictions);
CSVLoader<float>::save_comparison("comparison.csv", 
    dataset.X, dataset.Y, predictions,
    dataset.feature_names, dataset.label_names
);
```

### 4️⃣ **Configurar Diferentes Optimizadores**

```cpp
// SGD (Stochastic Gradient Descent)
network.train<MSELoss, SGD>(X, Y, epochs, batch_size, learning_rate);

// Adam (con momentum adaptativo)
network.train<BCELoss, Adam>(X, Y, epochs, batch_size, learning_rate);

// Diferentes funciones de pérdida
network.train<MSELoss, SGD>(X, Y, 1000, 4, 0.1f);     // Para regresión
network.train<BCELoss, Adam>(X, Y, 1000, 4, 0.001f);  // Para clasificación binaria
```

## 🏗️ Arquitectura del Proyecto

```
e:\projecto-final-null\proyecto-final/
├── 📁 src/                           # Código fuente principal
│   ├── 🧮 tensor.h                   # Librería de tensores N-dimensionales
│   ├── 🧠 neural_network.h           # Clase principal de red neuronal
│   ├── 📊 csv_loader.h               # Sistema de carga/guardado CSV
│   ├── 📁 layers/                    # Capas de la red neuronal
│   │   ├── 🔗 nn_interfaces.h        # Interfaces base (Layer, etc.)
│   │   ├── 🔢 nn_dense.h             # Capa densa (fully connected)
│   │   ├── ⚡ nn_activation.h        # Funciones de activación
│   │   └── 📉 nn_loss.h              # Funciones de pérdida
│   └── 📁 optimizers/                # Algoritmos de optimización
│       └── 🎯 nn_optimizer.h         # SGD, Adam, etc.
├── 📁 build/                         # Archivos de compilación
│   ├── 📁 bin/Debug/                 # Ejecutables Debug
│   └── 📁 bin/Release/               # Ejecutables Release
├── 📁 tests/                         # Pruebas unitarias
│   ├── 🧪 test_tensor.cpp            # Tests de tensores
│   └── 🧪 test_neural_network.cpp    # Tests de red neuronal
├── 📁 docs/                          # Documentación
│   ├── 📖 manual_usuario.md          # Este manual
│   └── 📚 investigacion_teorica.md   # Fundamentos teóricos
├── 🚀 main.cpp                       # Demo principal (4 demos)
├── ⚙️ CMakeLists.txt                 # Configuración de build
├── 📥 input_data.csv                 # Dataset de entrada
├── 📤 output_predictions.csv         # Predicciones generadas (se crea luego de ejecutar)
└── 📊 comparison_results.csv         # Análisis de resultados (se crea luego de ejecutar)
```

### 🔧 Componentes Principales

#### **1. Tensor Engine (`tensor.h`)**
- **Motor de cálculo matricial** de alto rendimiento
- **Soporte N-dimensional** con verificación de tipos
- **Operaciones vectorizadas** para eficiencia

#### **2. Neural Network Core (`neural_network.h`)**
- **Arquitectura modular** con capas intercambiables
- **Forward/Backward propagation** optimizado
- **Training loop** con métricas en tiempo real

#### **3. CSV Data Pipeline (`csv_loader.h`)**
- **Parser CSV robusto** con manejo de errores
- **Conversión automática** a tensores
- **Export de resultados** en múltiples formatos

#### **4. Layer System (`layers/`)**
- **Interface uniforme** para todas las capas
- **Dense layers** con inicialización configurable
- **Activation functions** diferenciables

#### **5. Optimization Engine (`optimizers/`)**
- **SGD** con momentum opcional
- **Adam** con decay adaptativo
- **Extensible** para nuevos optimizadores

## 🧪 Testing y Validación

### 🏃‍♂️ Ejecutar Tests (Opcional)

```powershell
# Compilar con tests habilitados
cmake .. -DBUILD_TESTS=ON
cmake --build . --config Debug

# Ejecutar tests individuales
.\test_tensor.exe
.\test_neural_network.exe
```

### ✅ Verificación de Funcionamiento

#### **Indicadores de Éxito:**
1. **Compilación exitosa** sin errores ni warnings
2. **Todos los demos ejecutan** sin excepciones
3. **Archivos CSV generados** en la raíz del proyecto
4. **Precisión del XOR ≥ 95%** (debe ser ~100%)
5. **Tiempos de entrenamiento < 200ms** por demo

#### **Salida Esperada:**
```
========================================
   Neural Network Demo - Proyecto Final
   CS2013 Programación III
========================================

=== Demo: Tensor Operations ===
Matrix A (2x3): ...
✅ Operaciones tensoriales correctas

=== Demo: XOR Problem (using input_data.csv) ===
Loaded dataset from CSV: 10 samples, 2 features
Training network with 1000 epochs...
Accuracy: 100%
✅ XOR resuelto perfectamente

=== Demo: Circle Classification (using input_data.csv) ===
Training time: 76 ms
Final accuracy: 50-90%
✅ Clasificación completada

=== Demo: CSV Workflow ===
✅ Saved output_predictions.csv
✅ Saved comparison_results.csv
📊 Model accuracy: 100%
✅ CSV workflow completed successfully!

========================================
   All demos completed successfully!
========================================
```

## 🚨 Solución de Problemas

### ❌ **Error: "cmake: command not found"**
**Solución**: Instalar CMake desde https://cmake.org/download/

### ❌ **Error: "Compilador C++17 no encontrado"**
**Solución**: 
- Windows: Instalar Visual Studio 2019+
- Linux: `sudo apt install g++-11`

### ❌ **Error: "neural_net_demo.exe no encontrado"**
**Verificar**:
```powershell
dir bin\Debug\       # Debe mostrar neural_net_demo.exe
```

### ❌ **Error: "input_data.csv not found"**
**Verificar**:
```powershell
dir ..\*.csv         # Desde directorio build
```

### ❌ **Warning: Precisión baja (<50%)**
**Posibles causas**:
- Dataset muy pequeño (normal con 10 muestras)
- Learning rate muy alto/bajo
- Pocas épocas de entrenamiento

## 📊 Métricas de Rendimiento

### ⏱️ **Tiempos Esperados** (Windows, CPU moderno):
- **Demo Tensor Operations**: < 5ms
- **Demo XOR Problem**: 80-100ms
- **Demo Circle Classification**: 70-80ms  
- **Demo CSV Workflow**: 50-100ms
- **Total**: < 300ms

### 🎯 **Precisión Esperada**:
- **XOR Problem**: 95-100%
- **Circle Classification**: 50-90% (depende del dataset)
- **CSV Workflow**: 80-100%

### 💾 **Uso de Memoria**: < 50MB
### 🗃️ **Archivos Generados**: 3 CSV (~1-5KB cada uno)

## ⚠️ Limitaciones Conocidas

### 🚀 **Rendimiento**
- **Implementación didáctica**: Optimizada para claridad del código, no velocidad máxima
- **CPU solo**: No incluye aceleración GPU (CUDA/OpenCL)
- **Threading**: Sin paralelización explícita del entrenamiento

### 💾 **Memoria**
- **Datasets pequeños**: Optimizado para datasets de demostración (<1000 muestras)
- **Carga completa**: Todos los datos se cargan en memoria
- **Sin streaming**: No hay soporte para datasets que no quepan en RAM

### 🔧 **Funcionalidad**
- **Capas limitadas**: Solo Dense + Activaciones básicas
- **Formatos**: Solo CSV para datos, no hay JSON/XML/binario
- **Persistencia**: No incluye guardado/carga de modelos entrenados
- **Regularización**: Sin Dropout, Batch Normalization, etc.

## 🚀 Extensiones Futuras

### 🧠 **Más Tipos de Capas**
```cpp
// Propuestas para futuras versiones:
network.add_layer(std::make_unique<Convolutional2D<float>>(32, 3, 3));
network.add_layer(std::make_unique<LSTM<float>>(128));
network.add_layer(std::make_unique<Dropout<float>>(0.5f));
network.add_layer(std::make_unique<BatchNormalization<float>>());
```

### ⚡ **Optimizaciones de Rendimiento**
- **BLAS integration**: Usar librerías optimizadas (Intel MKL, OpenBLAS)
- **SIMD**: Vectorización explícita con instrucciones AVX/SSE
- **GPU support**: Implementación CUDA para entrenamiento masivo
- **Parallel training**: Multi-threading para lotes grandes

### 📊 **Más Métricas y Funcionalidades**
```cpp
// Métricas avanzadas
float f1_score = network.evaluate_f1(X, Y);
float precision = network.evaluate_precision(X, Y);
float recall = network.evaluate_recall(X, Y);

// Callbacks de entrenamiento
network.add_callback(std::make_unique<EarlyStopping>(patience=10));
network.add_callback(std::make_unique<LearningRateScheduler>());
```

### 💾 **Persistencia de Modelos**
```cpp
// Guardar modelo entrenado
network.save("mi_modelo.nn");

// Cargar modelo previamente entrenado
auto network = NeuralNetwork<float>::load("mi_modelo.nn");
```

### 🌐 **Soporte de Formatos**
- **JSON/XML**: Para configuración de arquitecturas
- **HDF5/NPZ**: Para datasets científicos grandes
- **ONNX**: Para interoperabilidad con otros frameworks

## 👥 Información del Proyecto

### 📚 **Curso**: CS2013 Programación III
### 🎯 **Objetivo**: Implementación completa de red neuronal desde cero
### 💻 **Lenguaje**: C++17 moderno
### 🛠️ **Build System**: CMake 3.18+
### 📅 **Versión**: 2025.01 (Proyecto Final)

### 🔗 **Componentes Implementados**:
- ✅ **Librería de Tensores N-dimensionales**
- ✅ **Red Neuronal Multicapa**
- ✅ **Sistema de Optimización (SGD/Adam)**
- ✅ **Pipeline CSV Completo**
- ✅ **4 Demos Funcionales**
- ✅ **Documentación Completa**

---

## 📞 Soporte

Si encuentras problemas durante la compilación o ejecución:

1. **Verificar requisitos**: CMake 3.18+, C++17
2. **Limpiar build**: `rm -rf build && mkdir build`
3. **Verificar archivos CSV**: Deben estar en la raíz del proyecto
4. **Revisar logs de compilación**: Buscar errores específicos

**¡El proyecto está diseñado para funcionar out-of-the-box en sistemas Windows modernos!** 🎉
