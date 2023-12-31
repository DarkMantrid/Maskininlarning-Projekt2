#include "../inc/node.hpp"


namespace yrgo {
namespace machine_learning {

NeuralNetwork::NeuralNetwork() : input_nodes(4), hidden_nodes(4), output_nodes(1) {
    InitializeWeights();
    InitializeGPIO();
}

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
    buttonLines.clear();

    // Release neural network resources if needed
    // Add additional logic here for neural network resource cleanup if any
}

void NeuralNetwork::InitializeGPIO() {
    // Open GPIO chip
    chip = gpiod_chip_open("/dev/gpiochip0");
    if (!chip) {
        std::cerr << "Error opening GPIO chip" << std::endl;
        // Handle error or throw an exception
    }

    // Open and configure the LED line
    constexpr int ledGPIO = 25; // Replace with the actual GPIO pin for the LED
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
    std::vector<int> buttonGPIOLines = {17, 18, 19}; // Replace with your actual GPIO numbers for buttons

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

std::vector<int> NeuralNetwork::ReadButtonStates() {
    std::vector<int> buttonStates;

    for (const auto &buttonLine : buttonLines) {
        int state = gpiod_line_get_value(buttonLine);
        buttonStates.push_back(state);
    }

    return buttonStates;
}

void NeuralNetwork::ControlLED(bool state) {
    // Set LED state based on prediction
    int value = state ? 1 : 0;
    if (gpiod_line_set_value(ledLine, value) < 0) {
        std::cerr << "Error setting LED state" << std::endl;
        // Handle error or throw an exception
    }
}

void NeuralNetwork::InitializeWeights() {
    // Initialize weights and biases with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1.0, 1.0); // Adjust the range as needed

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

    // Other necessary variable initialization for the neural network
}

double NeuralNetwork::ReLU(double x) {
    // ReLU activation function
    return std::max(0.0, x);
}

void NeuralNetwork::ForwardPropagation(const std::vector<int> &input) {
    // Calculate inputs to the hidden layer
    std::vector<double> hidden_inputs(hidden_nodes);
    for (int i = 0; i < hidden_nodes; ++i) {
        hidden_inputs[i] = bias_hidden[i];
        for (int j = 0; j < input_nodes; ++j) {
            hidden_inputs[i] += input[j] * weights_input_hidden[j][i];
        }
    }

    // Apply ReLU activation function to the hidden layer
    std::vector<double> hidden_outputs(hidden_nodes);
    for (int i = 0; i < hidden_nodes; ++i) {
        hidden_outputs[i] = ReLU(hidden_inputs[i]);
    }

    // Calculate inputs to the output layer
    std::vector<double> final_inputs(output_nodes);
    for (int i = 0; i < output_nodes; ++i) {
        final_inputs[i] = bias_output[i];
        for (int j = 0; j < hidden_nodes; ++j) {
            final_inputs[i] += hidden_outputs[j] * weights_hidden_output[j][i];
        }
    }

    // Apply ReLU activation function to the output layer (if needed)
    // For classification tasks, the output layer might not use ReLU

    // Output layer activations or outputs (depends on the task)
    // Modify this part according to your network's output requirements
    for (int i = 0; i < output_nodes; ++i) {
        // Calculate the output of the output layer neurons here
        // Assign the calculated output to your output structure or variable
    }
}

void NeuralNetwork::BackPropagation(int target) {
    // Calculate output layer errors
    std::vector<double> output_errors(output_nodes);
    for (int i = 0; i < output_nodes; ++i) {
        // Calculate the error for each neuron in the output layer
        // Error calculation depends on the task (classification, regression, etc.)
        // Adjust this calculation based on your network's output requirements
        // Example for a simple squared error loss function
        // Replace this with the appropriate error calculation for your task
        double output_error = target - output[i]; // Adjust based on actual output storage
        output_errors[i] = output_error;

        // Update biases of the output layer
        bias_output[i] += learning_rate * output_error;

        // Update weights between hidden and output layers
        for (int j = 0; j < hidden_nodes; ++j) {
            double weight_delta = learning_rate * output_error * hidden_outputs[j];
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
        // Derivative of ReLU activation function
        double derivative = (hidden_outputs[i] > 0) ? 1 : 0;
        hidden_errors[i] = weighted_output_errors * derivative;

        // Update biases of the hidden layer
        bias_hidden[i] += learning_rate * hidden_errors[i];

        // Update weights between input and hidden layers
        for (int j = 0; j < input_nodes; ++j) {
            double weight_delta = learning_rate * hidden_errors[i] * input[j];
            weights_input_hidden[j][i] += weight_delta;
        }
    }
}
void NeuralNetwork::TrainNetwork(const std::vector<std::vector<int>> &inputs, const std::vector<int> &targets, int epochs, double learning_rate) {
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


int NeuralNetwork::Predict(const std::vector<int> &input) {
    // Perform forward propagation to predict the output based on the given input
    ForwardPropagation(input);

    // Interpret the network's output to make a prediction
    // You might need to adjust this logic based on your specific problem
    // For instance, if you're dealing with binary classification, 
    // return 0 or 1 based on the output value thresholded at 0.5
    // Adjust this return statement as per your network's output representation

    // Example: binary classification
    return (output[0] > 0.5) ? 1 : 0;
}

void NeuralNetwork::PredictAndControlLED() {
    // Initial training to set the neural network to predict button press patterns
    TrainNetwork({{0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 0, 1, 1}}, {0, 1, 1, 0}, 1000, 0.1);

    while (true) {
        std::vector<int> buttonStates = ReadButtonStates();

        // Make a prediction based on current button states
        int prediction = Predict(buttonStates);

        // Control LED based on the prediction
        ControlLED(prediction == 1);

        // Optional delay to avoid rapid predictions
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}


} // namespace machine_learning
} // namespace yrgo
