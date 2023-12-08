#pragma once

#include <vector>
#include <gpiod.h>
#include <iostream>
#include <random>
#include <cmath>
#include <chrono>
#include <thread>

namespace yrgo {
namespace machine_learning {

class Node {
public:
    Node();
    ~Node();

    void InitializeGPIO();
    void PredictAndControlLED();

private:
    struct gpiod_chip *chip;
    struct gpiod_line *ledLine;
    std::vector<struct gpiod_line *> buttonLines;

    // Neural network parameters
    int input_nodes;
    int hidden_nodes;
    int output_nodes;
    std::vector<std::vector<double>> weights_input_hidden;
    std::vector<std::vector<double>> weights_hidden_output;
    std::vector<double> bias_hidden;
    std::vector<double> bias_output;

    std::vector<double> hidden_outputs; // Store outputs of hidden layer neurons
    double learning_rate; // Learning rate for weight updates
    std::vector<int> input; // Input data used in backpropagation
    std::vector<double> output; // Actual output from the network during forward pass

    // Neural network methods
    std::vector<int> ReadButtonStates();
    void ControlLED(bool state);
    void TrainNetwork(const std::vector<std::vector<int>> &inputs, const std::vector<int> &targets, int epochs, double learning_rate);
    double ReLU(double x);
    void InitializeWeights();
    void ForwardPropagation(const std::vector<int> &input);
    void BackPropagation(int target);
    int Predict(const std::vector<int> &input);

    // Additional members for neural network functionality
    void SetOutput(const std::vector<double> &new_output) { output = new_output; }
    void SetHiddenOutputs(const std::vector<double> &new_hidden_outputs) { hidden_outputs = new_hidden_outputs; }
};

// Implement the methods inline or define them in a corresponding .cpp file
// This might depend on the complexity and size of the methods

inline Node::Node() : input_nodes(4), hidden_nodes(4), output_nodes(1) {
    InitializeWeights();
    InitializeGPIO();
    output.resize(output_nodes);
    hidden_outputs.resize(hidden_nodes);
}

inline Node::~Node() {
    // Release GPIO resources
    if (chip) {
        gpiod_chip_close(chip);
        chip = nullptr;
    }
    if (ledLine) {
        gpiod_line_release(ledLine);
        ledLine = nullptr;
    }
    for (auto line : buttonLines) {
        if (line) {
            gpiod_line_release(line);
        }
    }
    buttonLines.clear();

    // Release neural network resources if needed
    // Add additional logic here for neural network resource cleanup if any
}

// Implement the methods inline or define them in a corresponding .cpp file
// This might depend on the complexity and size of the methods

} // namespace machine_learning
} // namespace yrgo
