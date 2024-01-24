#include "../inc/node.hpp"
#include <cmath>
#include <iostream>
#include <thread>
#include <chrono>

namespace {
    
template <typename T = int>
void Print(const std::vector<T>& data, std::ostream& ostream = std::cout) {
    static_assert(std::is_arithmetic<T>::value, 
        "Non-arithmetic type selected for method ::Print!");
    ostream << "[";
    for (const auto& i : data) {
        ostream << i;
        if (&i < &data[data.size() - 1]) { ostream << ", "; }
    }
    ostream << "]\n";
}
} // namespace

namespace yrgo {
namespace machine_learning {

NeuralNetwork::NeuralNetwork() 
    : input_nodes(4) 
    , hidden_nodes(10) 
    , output_nodes(1)
    , learning_rate(0.01) {
        InitializeWeights();
        output.resize(output_nodes, 0);
        input.resize(input_nodes, 0);
        hidden_outputs.resize(hidden_nodes, 0);
        weights_input_hidden.resize(input_nodes, std::vector<double>(hidden_nodes));
        weights_hidden_output.resize(hidden_nodes, std::vector<double>(output_nodes));

}

NeuralNetwork::~NeuralNetwork() {

    // Release neural network resources if needed
    // Add additional logic here for neural network resource cleanup if any
}

void NeuralNetwork::InitializeGPIO() {
}

std::vector<double> NeuralNetwork::ReadButtonStates() {
    std::vector<double> buttonStates;
    return buttonStates;
}

void NeuralNetwork::ControlLED(bool state) {
}

void NeuralNetwork::InitializeWeights() {
    // Initialize weights and biases with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0, 1.0); // Adjust the range as needed

    // Initialize weights for input-hidden layer
    weights_input_hidden.resize(input_nodes, std::vector<double>(hidden_nodes));
    for (int i = 0; i < input_nodes; ++i) {
        for (int j = 0; j < hidden_nodes; ++j) {
            weights_input_hidden[i][j] = dist(gen);
        }
    }

    // Initialize weights for hidden-output layer
    weights_hidden_output.resize(hidden_nodes, std::vector<double>(output_nodes));
    for (int i = 0; i < hidden_nodes; ++i) {
        for (int j = 0; j < output_nodes; ++j) {
            weights_hidden_output[i][j] = dist(gen);
        }
    }

    // Initialize biases for hidden and output layers
    bias_hidden.resize(hidden_nodes);
    bias_output.resize(output_nodes);
    for (int i = 0; i < hidden_nodes; ++i) {
        bias_hidden[i] = dist(gen);
    }
    for (int i = 0; i < output_nodes; ++i) {
        bias_output[i] = dist(gen);
    }

    // std::cout << "Bias hidden ";
    // Print(bias_hidden);
    // std::this_thread::sleep_for(std::chrono::milliseconds(5000));

    // Other necessary variable initialization for the neural network
}

double NeuralNetwork::ReLU(double x) {
    // ReLU activation function
    return std::max(0.0, x);
}

double NeuralNetwork::TanH(double x) {
    return std::tanh(x);
}

double NeuralNetwork::TanHDelta(double x) {
    return 1 - std::pow(std::tanh(x), 2); // 1 - tanh^2
}

void NeuralNetwork::ForwardPropagation(const std::vector<double> &input) {
    // Calculate inputs to the hidden layer
    std::vector<double> hidden_inputs(hidden_nodes, 0);
    for (int i = 0; i < hidden_nodes; ++i) {
        hidden_inputs[i] = bias_hidden[i];
        // std::cout << "Hidden inputs[" << i << "] = " << hidden_inputs[i] << "\n";
        for (int j = 0; j < input_nodes; ++j) {
            hidden_inputs[i] += input[j] * weights_input_hidden[j][i];
        }
    }

    // Apply TanH activation function to the hidden layer
    hidden_outputs.resize(hidden_nodes, 0);
    for (int i = 0; i < hidden_nodes; ++i) {
        hidden_outputs[i] = TanH(hidden_inputs[i]);
    }

    // std::cout <<  "Hidden output: ";
    // Print(hidden_outputs);

    // Calculate inputs to the output layer
    std::vector<double> final_inputs(output_nodes, 0);
    for (int i = 0; i < output_nodes; ++i) {
        final_inputs[i] = bias_output[i];
        for (int j = 0; j < hidden_nodes; ++j) {
            final_inputs[i] += hidden_outputs[j] * weights_hidden_output[j][i];
        }
    }

    output.resize(output_nodes, 0);
    for (int i = 0; i < output_nodes; ++i) {
        output[i] = ReLU(final_inputs[i]);
    }
    
    // std::cout <<  "Current output: ";
    // Print(output);
}

void NeuralNetwork::BackPropagation(double target) {
    // Calculate output layer errors
    std::vector<double> output_errors(output_nodes); // Adjusted to use resize
    for (int i = 0; i < output_nodes; ++i) {
        // Calculate the error for each neuron in the output layerk
        double output_error = target - output[i]; // Adjust based on actual output storage
        auto relu_delta = output[i] > 0 ? 1 : 0;
        output_errors[i] = output_error * relu_delta;

        // std::cout << "Output  errors: ";
        // Print(output_errors);

        // Update biases of the output layer
        bias_output[i] += learning_rate * output_error;

        // Update weights between hidden and output layers
        for (int j = 0; j < hidden_nodes; ++j) {
            double weight_delta = learning_rate * output_errors[i] * hidden_outputs[j];
            weights_hidden_output[j][i] += weight_delta;
        }
    }

    // Calculate hidden layer errors
    std::vector<double> hidden_errors(hidden_nodes);
    for (int i = 0; i < hidden_nodes; ++i) {
        // Calculate the error for each neuron in the hidden layer
        // Error calculation depends on the activation function and output layer errors
        // Here, it's based on the weighted sum of output errors and weights from hidden to output
        double weighted_output_errors = 0.0;
        for (int j = 0; j < output_nodes; ++j) {
            weighted_output_errors += output_errors[j] * weights_hidden_output[i][j];
        }
        // Derivative of TanH activation function
        hidden_errors[i] = weighted_output_errors * TanHDelta(hidden_outputs[i]);

        // Update biases of the hidden layer
        bias_hidden[i] += learning_rate * hidden_errors[i];
        
        for (int j = 0; j < input_nodes; ++j) {
            double weight_delta = learning_rate * hidden_errors[i] * input[j];
            weights_input_hidden[j][i] += weight_delta;
        }
    }
}

void NeuralNetwork::TrainNetwork(const std::vector<std::vector<double>> &inputs, const std::vector<double> &targets, double epochs, double learning_rate) {
    // Training loop for the specified number of epochs
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Iterate over each input-target pair for training
        for (size_t idx = 0; idx < inputs.size(); ++idx) {
            // Perform forward propagation for the current input
            ForwardPropagation(inputs[idx]);

            // Calculate output layer errors and update weights using backpropagation
            BackPropagation(targets[idx]);

            // Optionally, perform weight updates after processing a batch of inputs
            // Here, weight updates are done after processing each input for simplicity
        }
    }
}

double NeuralNetwork::Predict(const std::vector<double> &input) {
    // Perform forward propagation to predict the output based on the given input
    ForwardPropagation(input);

    // Apply ReLU activation function to the output layer
    return ReLU(output[0]);
}

void NeuralNetwork::PredictAndControlLED() {
    // Initial training to set the neural network to predict button press patterns
    const std::vector<std::vector<double>> trainingData = {
        {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 0, 1, 1},
        {0, 1, 0, 0}, {0, 1, 0, 1}, {0, 1, 1, 0}, {0, 1, 1, 1},
        {1, 0, 0, 0}, {1, 0, 0, 1}, {1, 0, 1, 0}, {1, 0, 1, 1},
        {1, 1, 0, 0}, {1, 1, 0, 1}, {1, 1, 1, 0}, {1, 1, 1, 1}
    };

    const std::vector<double> labels = {0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0};
    const double epochs = 10000;
    const double learningRate = 0.1;

    TrainNetwork(trainingData, labels, epochs, learningRate);

    // Vi testar alla trainingData-uppsättningar och skriver ut i terminalen.
    PrintPredictions(trainingData, 2);
}

void NeuralNetwork::PrintPredictions(const std::vector<std::vector<double>>& input_sets,
                                     const std::size_t num_decimals,
                                     std::ostream& ostream) {
    if (input_sets.size() == 0) { return; }
    ostream << std::fixed << std::setprecision(num_decimals);
    ostream << "--------------------------------------------------------------------------------";
    for (const auto& input: input_sets) {
        ostream << "\nInput:\t";
        Print<double>(input, ostream);
        ostream << "Predicted:\t";
        ostream << "[" << Predict(input) << "]\n";
    }
    ostream << "--------------------------------------------------------------------------------\n\n";
}

} // namespace machine_learning
} // namespace yrgo
