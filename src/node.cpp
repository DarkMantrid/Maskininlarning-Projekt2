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

NeuralNetwork::NeuralNetwork(const std::size_t num_inputs, 
                             const std::size_t num_hidden, 
                             const std::size_t num_outputs,
                             const double learning_rate) 
    : input_nodes(num_inputs) 
    , hidden_nodes(num_hidden) 
    , output_nodes(num_outputs)
    , learning_rate(learning_rate) {
        InitializeWeights();
        output.resize(output_nodes, 0);
        input.resize(input_nodes, 0);
        hidden_outputs.resize(hidden_nodes, 0);
        hidden_errors.resize(hidden_nodes, 0);
        output_errors.resize(output_nodes, 0);

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
    std::uniform_real_distribution<> dist(0, 1); // Adjust the range as needed

    bias_hidden.resize(hidden_nodes);
    weights_input_hidden.resize(hidden_nodes, std::vector<double>(input_nodes));
    for (int i = 0; i < hidden_nodes; ++i) {
         bias_hidden[i] = dist(gen);
        for (int j = 0; j < input_nodes; ++j) {
            weights_input_hidden[i][j] = dist(gen);
        }
    }

    bias_output.resize(output_nodes);
    weights_hidden_output.resize(output_nodes, std::vector<double>(hidden_nodes));
    for (int i = 0; i < output_nodes; ++i) {
        bias_output[i] = dist(gen);
        for (int j = 0; j < hidden_nodes; ++j) {
            weights_hidden_output[i][j] = dist(gen);
        }
    }
}

double NeuralNetwork::ReLU(double x) {
    // ReLU activation function
    return std::max(0.0, x);
}

double NeuralNetwork::ReLUDelta(double x) {
    return x > 0 ? 1 : 0;
}

double NeuralNetwork::TanH(double x) {
    return std::tanh(x);
}

double NeuralNetwork::TanHDelta(double x) {
    return 1 - std::pow(std::tanh(x), 2); // 1 - tanh^2
}

void NeuralNetwork::ForwardPropagation(const std::vector<double> &input) {
    std::vector<double> hidden_inputs(hidden_nodes, 0);
    for (int i = 0; i < hidden_nodes; ++i) {
        hidden_inputs[i] = bias_hidden[i];
        for (int j = 0; j < input_nodes; ++j) {
            hidden_inputs[i] += input[j] * weights_input_hidden[i][j];
        }
        hidden_outputs[i] = TanH(hidden_inputs[i]);
    }

    std::vector<double> final_inputs(output_nodes, 0);
    for (int i = 0; i < output_nodes; ++i) {
        final_inputs[i] = bias_output[i];
        for (int j = 0; j < hidden_nodes; ++j) {
            final_inputs[i] += hidden_outputs[j] * weights_hidden_output[i][j];
        }
        output[i] = ReLU(final_inputs[i]);
    }
}

void NeuralNetwork::BackPropagation(double target) { 
    for (int i = 0; i < output_nodes; ++i) {
        output_errors[i] = (target - output[i]) * ReLUDelta(output[i]);
    }

    for (int i = 0; i < hidden_nodes; ++i) {
        double weighted_output_errors = 0.0;
        for (int j = 0; j < output_nodes; ++j) {
            weighted_output_errors += output_errors[j] * weights_hidden_output[j][i];
        }
        hidden_errors[i] = weighted_output_errors * TanHDelta(hidden_outputs[i]);
    }
}

void NeuralNetwork::Optimize(const std::vector<double> &input) {
    for (std::size_t i{}; i < hidden_nodes; ++i) {
        bias_hidden[i] += hidden_errors[i] * learning_rate;
        for (std::size_t j{}; j < input_nodes && j < input.size(); ++j) {
            weights_input_hidden[i][j] += hidden_errors[i] * learning_rate * input[j];
        }
    }

    for (std::size_t i{}; i < output_nodes; ++i) {
        bias_output[i] += output_errors[i] * learning_rate;
        for (std::size_t j{}; j < hidden_nodes; ++j) {
            weights_hidden_output[i][j] += output_errors[i] * learning_rate * hidden_outputs[j];
        }
    }
}

void NeuralNetwork::TrainNetwork(const std::vector<std::vector<double>> &inputs, const std::vector<double> &targets, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t idx = 0; idx < inputs.size(); ++idx) {
            ForwardPropagation(inputs[idx]);
            BackPropagation(targets[idx]);
            Optimize(inputs[idx]);
        }
    }
}

double NeuralNetwork::Predict(const std::vector<double> &input) {
    // Perform forward propagation to predict the output based on the given input
    ForwardPropagation(input);

    // Apply ReLU activation function to the output layer
    return ReLU(output[0]);
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
