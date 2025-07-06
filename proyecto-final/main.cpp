#include "src/neural_network.h"
#include <iostream>
#include <chrono>

using namespace utec::neural_network;
using namespace utec::data;
using namespace utec::algebra;

// Forward declarations
void demo_tensor_operations();
void demo_xor_problem();
void demo_circle_classification();
void demo_csv_workflow();

int main() {
    std::cout << "========================================\n";
    std::cout << "   Neural Network Demo - Proyecto Final\n";
    std::cout << "   CS2013 Programación III\n";
    std::cout << "========================================\n\n";

    try {
        // Demo 1: Tensor Operations
        demo_tensor_operations();
        
        // Demo 2: XOR Problem
        demo_xor_problem();
        
        // Demo 3: Circle Classification
        demo_circle_classification();
        
        // Demo 4: CSV Workflow
        demo_csv_workflow();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "========================================\n";
    std::cout << "   All demos completed successfully!\n";
    std::cout << "========================================\n";
    
    return 0;
}

void demo_tensor_operations() {
    std::cout << "=== Demo: Tensor Operations ===\n";
    
    // Create tensors
    utec::algebra::Tensor<float, 2> A(2, 3);
    A = {1, 2, 3, 4, 5, 6};
    
    utec::algebra::Tensor<float, 2> B(3, 2);
    B = {1, 2, 3, 4, 5, 6};
    
    std::cout << "Matrix A (2x3):\n" << A;
    std::cout << "Matrix B (3x2):\n" << B;
    
    // Matrix multiplication
    auto C = A.matmul(B);
    std::cout << "A * B:\n" << C;
    
    // Transpose
    auto AT = A.transpose();
    std::cout << "A transpose:\n" << AT;
    
    // Element-wise operations
    utec::algebra::Tensor<float, 2> D(2, 3);
    D.fill(2.0f);
    auto E = A + D;
    std::cout << "A + D (filled with 2.0):\n" << E;
    
    std::cout << "\n";
}

void demo_xor_problem() {
    std::cout << "=== Demo: XOR Problem (using input_data.csv) ===\n";
    
    try {
        // Load data from CSV instead of generating XOR dataset
        std::cout << "Loading data from input_data.csv...\n";
        auto dataset = CSVLoader<float>::load_csv("../input_data.csv", true, ',', -1);
        
        std::cout << "Loaded dataset from CSV:\n";
        std::cout << "Samples: " << dataset.X.shape()[0] << ", Features: " << dataset.X.shape()[1] << std::endl;
        std::cout << "Training data (from CSV):\n";
        std::cout << "Inputs (X):\n" << dataset.X;
        std::cout << "Expected outputs (Y):\n" << dataset.Y;
        
        // Create neural network
        std::cout << "Creating neural network...\n";
        NeuralNetwork<float> network;        
        // Add layers: 2 -> 4 -> 1
        std::cout << "Adding layers...\n";
        network.add_layer(std::make_unique<Dense<float>>(2, 4, Initializers<float>::xavier_init, Initializers<float>::zero_init));
        network.add_layer(std::make_unique<ReLU<float>>());
        network.add_layer(std::make_unique<Dense<float>>(4, 1, Initializers<float>::xavier_init, Initializers<float>::zero_init));
        network.add_layer(std::make_unique<Sigmoid<float>>());
        
        // Training parameters
        const size_t epochs = 1000;
        const size_t batch_size = 4;
        const float learning_rate = 0.1f;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Train the network
        std::cout << "Training network with " << epochs << " epochs...\n";
        network.train<MSELoss, SGD>(dataset.X, dataset.Y, epochs, batch_size, learning_rate, false);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // Test the network
        auto predictions = network.predict(dataset.X);
        
        std::cout << "Final predictions:\n";
        for (size_t i = 0; i < dataset.X.shape()[0]; ++i) {
            std::cout << "Input: [" << dataset.X(i, 0) << ", " << dataset.X(i, 1) 
                      << "] -> Output: " << predictions(i, 0) 
                      << " (Expected: " << dataset.Y(i, 0) << ")\n";
        }
        
        float accuracy = network.evaluate_accuracy(dataset.X, dataset.Y);
        std::cout << "Training time: " << duration.count() << " ms\n";
        std::cout << "Accuracy: " << (accuracy * 100) << "%\n\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error in XOR demo: " << e.what() << std::endl;
        std::cerr << "Falling back to generated XOR dataset...\n";
        
        // Fallback to generated XOR dataset
        auto dataset = DatasetGenerator<float>::create_xor_dataset();
        
        NeuralNetwork<float> network;
        network.add_layer(std::make_unique<Dense<float>>(2, 4, Initializers<float>::xavier_init, Initializers<float>::zero_init));
        network.add_layer(std::make_unique<ReLU<float>>());
        network.add_layer(std::make_unique<Dense<float>>(4, 1, Initializers<float>::xavier_init, Initializers<float>::zero_init));
        network.add_layer(std::make_unique<Sigmoid<float>>());
        
        network.train<MSELoss, SGD>(dataset.X, dataset.Y, 1000, 4, 0.1f, false);
        auto predictions = network.predict(dataset.X);
        
        float accuracy = network.evaluate_accuracy(dataset.X, dataset.Y);
        std::cout << "Fallback XOR demo completed! Accuracy: " << (accuracy * 100) << "%\n\n";
    }
}

void demo_circle_classification() {
    std::cout << "=== Demo: Circle Classification (using input_data.csv) ===\n";
    
    try {
        // Load data from CSV instead of generating circle dataset
        std::cout << "Loading data from input_data.csv...\n";
        auto dataset = CSVLoader<float>::load_csv("../input_data.csv", true, ',', -1);
        
        std::cout << "Loaded dataset from CSV:\n";
        std::cout << "Samples: " << dataset.X.shape()[0] << ", Features: " << dataset.X.shape()[1] << std::endl;
        std::cout << "Features: [x, y] coordinates from CSV\n";
        std::cout << "Labels: Binary classification from CSV\n";
        
        // Create a simpler network
        NeuralNetwork<float> network;
        
        // Add layers: 2 -> 4 -> 1 (simplified from 2 -> 8 -> 4 -> 1)
        network.add_layer(std::make_unique<Dense<float>>(2, 4, Initializers<float>::xavier_init, Initializers<float>::zero_init));
        network.add_layer(std::make_unique<ReLU<float>>());
        network.add_layer(std::make_unique<Dense<float>>(4, 1, Initializers<float>::xavier_init, Initializers<float>::zero_init));
        network.add_layer(std::make_unique<Sigmoid<float>>());
        
        // Training parameters
        const size_t epochs = 1000;
        const size_t batch_size = 10;
        const float learning_rate = 0.001f;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::cout << "Starting training with " << epochs << " epochs...\n";
        
        // Train with SGD optimizer (more stable than Adam)
        network.train<BCELoss, SGD>(dataset.X, dataset.Y, epochs, batch_size, learning_rate, true);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        // Evaluate performance
        float accuracy = network.evaluate_accuracy(dataset.X, dataset.Y);
        
        std::cout << "Training time: " << duration.count() << " ms\n";
        std::cout << "Final accuracy: " << (accuracy * 100) << "%\n";
        
        // Show some predictions
        std::cout << "Sample predictions:\n";
        auto predictions = network.predict(dataset.X);
        size_t max_samples = std::min(static_cast<size_t>(5), dataset.X.shape()[0]);
        for (size_t i = 0; i < max_samples; ++i) {
            std::cout << "Point (" << dataset.X(i, 0) << ", " << dataset.X(i, 1) 
                      << ") -> Prediction: " << predictions(i, 0) 
                      << " (Actual: " << dataset.Y(i, 0) << ")\n";
        }
        std::cout << "Circle classification demo completed!\n\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error in Circle Classification demo: " << e.what() << std::endl;
        std::cerr << "Falling back to generated circular dataset...\n";
        
        // Fallback to generated circle dataset
        auto dataset = DatasetGenerator<float>::create_circle_dataset(100, 0.1f);
        
        NeuralNetwork<float> network;
        network.add_layer(std::make_unique<Dense<float>>(2, 4, Initializers<float>::xavier_init, Initializers<float>::zero_init));
        network.add_layer(std::make_unique<ReLU<float>>());
        network.add_layer(std::make_unique<Dense<float>>(4, 1, Initializers<float>::xavier_init, Initializers<float>::zero_init));
        network.add_layer(std::make_unique<Sigmoid<float>>());
        
        network.train<BCELoss, SGD>(dataset.X, dataset.Y, 1000, 10, 0.001f, true);
        float accuracy = network.evaluate_accuracy(dataset.X, dataset.Y);
        
        std::cout << "Fallback Circle Classification completed! Accuracy: " << (accuracy * 100) << "%\n\n";
    }
}

void demo_csv_workflow() {
    std::cout << "=== Demo: CSV Workflow ===\n";
    
    try {
        // PASO 1: Simular carga de datos reales
        std::cout << "1. Loading existing CSV data (simulating real-world scenario)...\n";
        
        // Intentar cargar el archivo existente desde la raíz
        std::cout << "   - Looking for input_data.csv in project root...\n";
        
        // Cargar datos de entrada desde la raíz
        auto input_data = CSVLoader<float>::load_csv("../input_data.csv", true, ',', -1);
        std::cout << "   ✅ Loaded input data: " << input_data.X.shape()[0] << " samples, " 
                  << input_data.X.shape()[1] << " features\n";
        
        // PASO 2: Entrenar red neuronal
        std::cout << "2. Training neural network with loaded data...\n";
        
        NeuralNetwork<float> network;
        std::cout << "   - Creating network layers...\n";
        network.add_layer(std::make_unique<Dense<float>>(2, 6, Initializers<float>::xavier_init, Initializers<float>::zero_init));
        network.add_layer(std::make_unique<ReLU<float>>());
        network.add_layer(std::make_unique<Dense<float>>(6, 1, Initializers<float>::xavier_init, Initializers<float>::zero_init));
        network.add_layer(std::make_unique<Sigmoid<float>>());
        
        std::cout << "   - Starting training with 1000 epochs...\n";
        network.train<BCELoss, SGD>(input_data.X, input_data.Y, 1000, 2, 0.01f, false);
        std::cout << "   - Training completed!\n";
        
        // PASO 3: Generar predicciones REALES de la red neuronal entrenada
        std::cout << "3. Generating real neural network predictions...\n";
        
        // Usar predicciones reales de la red neuronal entrenada
        auto predictions = network.predict(input_data.X);
        std::cout << "   - Generated real predictions from trained network!\n";
        
        std::cout << "   - Saving to CSV files...\n";
        
        // Guardar predicciones simples en la raíz
        CSVLoader<float>::save_predictions("../output_predictions.csv", predictions);
        std::cout << "   ✅ Saved output_predictions.csv\n";
        
        // Guardar comparación detallada en la raíz
        CSVLoader<float>::save_comparison("../comparison_results.csv", 
                                        input_data.X, 
                                        input_data.Y, 
                                        predictions,
                                        input_data.feature_names,
                                        input_data.label_names);
        std::cout << "   ✅ Saved comparison_results.csv\n";
        
        // PASO 4: Evaluar resultados
        float accuracy = network.evaluate_accuracy(input_data.X, input_data.Y);
        std::cout << "   📊 Model accuracy: " << (accuracy * 100) << "%\n";
        
        std::cout << "\n🎉 CSV workflow completed successfully!\n";
        std::cout << "📥 Input: input_data.csv (project root)\n";
        std::cout << "📤 Output: output_predictions.csv (project root)\n";
        std::cout << "📊 Comparison: comparison_results.csv (project root)\n";
        
    } catch (const std::exception& e) {
        std::cout << "❌ CSV workflow failed: " << e.what() << "\n";
        std::cout << "Note: This is expected if running without proper file system access.\n";
    }
    
    std::cout << "\n";
}
