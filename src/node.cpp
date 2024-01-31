#include "../inc/node.hpp"
#include "../inc/gpio_utils.hpp"
#include <cmath>
#include <iostream>
#include <thread>
#include <chrono>

namespace {

/**********************************************************************
 * @brief Template function to print elements of a vector to an output stream.
 * @tparam T The type of elements in the vector. Default is int.
 * @param data The vector containing elements to be printed.
 * @param ostream The output stream where the elements will be printed. 
 *                Default is std::cout.
 *********************************************************************/    
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

/**********************************************************************
 * @brief Default constructor for NeuralNetwork class
 * @param num_inputs, num_hidden, num_outputs, learning_rate
 **********************************************************************/
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

/**********************************************************************
 * @brief Destructor for NeuralNetwork class
 **********************************************************************/
NeuralNetwork::~NeuralNetwork() {
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
    // Release neural network resources
    buttonLines.clear();
}

/**********************************************************************
 * @brief Initializes the GPIO chip and lines
 **********************************************************************/
void NeuralNetwork::InitializeGPIO() {
    // Open GPIO chip
    chip = gpiod_chip_open("/dev/gpiochip0");
    if (!chip) {
        std::cerr << "Error opening GPIO chip" << std::endl;
        // Handle error or throw an exception
    }

    // Open and configure the LED line
    constexpr int ledGPIO = 17;
    ledLine = gpiod_chip_get_line(chip, ledGPIO);
    if (!ledLine) {
        std::cerr << "Error getting LED line" << std::endl;
        // Handle error or throw an exception
    }
    if (gpiod_line_request_output(ledLine, "LED", 0) < 0) {
        std::cerr << "Error requesting LED line" << std::endl;
        // Handle error or throw an exception
    }

    // GPIO numbers for buttons
    std::vector<int> buttonGPIOLines = {22, 23, 24, 25};

    // Loop through the button lines
    for (std::vector<gpiod_line*>::size_type buttonNum = 0; buttonNum < buttonGPIOLines.size(); ++buttonNum) {
        struct gpiod_line *buttonLine = gpiod_chip_get_line(chip, buttonGPIOLines[buttonNum]);
        if (!buttonLine) {
            std::cerr << "Error getting button line " << buttonNum << std::endl;
            // Handle error or throw an exception
        }
        if (gpiod_line_request_input(buttonLine, "Button") < 0) {
            std::cerr << "Error requesting button line " << buttonNum << std::endl;
            // Handle error or throw an exception
        }
        buttonLines.push_back(buttonLine);
    }
}

/**********************************************************************
 * @brief Reads the states of the buttons
 * @return Vector containing the states of the buttons
 **********************************************************************/
std::vector<double> NeuralNetwork::ReadButtonStates() {
    std::vector<double> buttonStates;

    for (const auto &buttonLine : buttonLines) {
        int state = gpiod_line_get_value(buttonLine);
        buttonStates.push_back(state);
    }

    return buttonStates;
}

/**********************************************************************
 * @brief Controls the LED based on the given state
 * @param state The state to set the LED to
 **********************************************************************/
void NeuralNetwork::ControlLED(bool state) {
    // Set LED state based on prediction
    int value = state ? 1 : 0;
    if (gpiod_line_set_value(ledLine, value) < 0) {
        std::cerr << "Error setting LED state" << std::endl;
        // Handle error or throw an exception
    }
}

/**********************************************************************
 * @brief Initializes the weights and biases of the neural network
 **********************************************************************/
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

/**********************************************************************
 * @brief Rectified Linear Unit (ReLU) activation function
 * @param x Input value
 * @return Output value after applying ReLU
 **********************************************************************/
double NeuralNetwork::ReLU(double x) {
    // ReLU activation function
    return std::max(0.0, x);
}

/**********************************************************************
 * @brief Derivative of the Rectified Linear Unit (ReLU) activation function
 * @param x Input value
 * @return Derivative of ReLU with respect to the input value
 **********************************************************************/
double NeuralNetwork::ReLUDelta(double x) {
    return x > 0 ? 1 : 0;
}

/**********************************************************************
 * @brief Hyperbolic Tangent (TanH) activation function
 * @param x Input value
 * @return Output value after applying TanH
 **********************************************************************/
double NeuralNetwork::TanH(double x) {
    return std::tanh(x);
}

/**********************************************************************
 * @brief Derivative of the Hyperbolic Tangent (TanH) activation function
 * @param x Input value
 * @return Derivative of TanH with respect to the input value
 **********************************************************************/
double NeuralNetwork::TanHDelta(double x) {
    return 1 - std::pow(std::tanh(x), 2); // 1 - tanh^2
}

/**********************************************************************
 * @brief Performs forward propagation in the neural network
 * @param input Vector of input data
 **********************************************************************/
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

/**********************************************************************
 * @brief Performs backpropagation in the neural network
 * @param target Target output value
 **********************************************************************/
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

/**********************************************************************
 * @brief Optimizes the neural network parameters using the provided input
 * @param input Input vector for optimization
 **********************************************************************/
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

/**********************************************************************
 * @brief Trains the neural network
 * @param inputs Vector of input data
 * @param targets Vector of target outputs
 * @param epochs Number of training epochs
 **********************************************************************/
void NeuralNetwork::TrainNetwork(const std::vector<std::vector<double>> &inputs, const std::vector<double> &targets, int epochs) {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (size_t idx = 0; idx < inputs.size(); ++idx) {
            ForwardPropagation(inputs[idx]);
            BackPropagation(targets[idx]);
            Optimize(inputs[idx]);
        }
    }
}

/**********************************************************************
 * @brief Predicts the output based on the input data
 * @param input Vector of input data
 * @return Predicted output value
 **********************************************************************/
double NeuralNetwork::Predict(const std::vector<double> &input) {
    // Perform forward propagation to predict the output based on the given input
    ForwardPropagation(input);

    // Apply ReLU activation function to the output layer
    return ReLU(output[0]);
}

/********************************************************************************
 * @brief Predicts and controls the LEDs.
 *******************************************************************************/
void NeuralNetwork::PredictAndControlLED() {
    // Read the states of the buttons
    std::vector<double> buttonStates = ReadButtonStates();

    // Predict LED state based on button inputs
    double prediction = Predict(buttonStates);

    // Control LED based on the predicted state
    ControlLED(prediction > 0.5);
}

/********************************************************************************
 * @brief Performs predictions with all input sets and prints the output.
 * @param input_sets   Reference to vector holding all input sets to predict with.
 * @param num_decimals The number of decimals to print (default = 0).
 * @param ostream      Reference to output stream (default = terminal print).
 ********************************************************************************/
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
