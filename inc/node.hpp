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


/**********************************************************************
 * @brief Class representing a NeuralNetwork in a machine learning system
 **********************************************************************/
class NeuralNetwork {
public:
    /**********************************************************************
     * @brief Default constructor for NeuralNetwork class
     **********************************************************************/
    NeuralNetwork();

    /**********************************************************************
     * @brief Destructor for NeuralNetwork class
     **********************************************************************/
    ~NeuralNetwork();

    /**********************************************************************
     * @brief Initializes the GPIO chip and lines
     **********************************************************************/
    void InitializeGPIO();

    /**********************************************************************
     * @brief Predicts and controls the LED based on the neural network's 
     *        output
     **********************************************************************/
    void PredictAndControlLED();

    /**********************************************************************
     * @brief Trains the neural network using backpropagation
     * @param inputs Vector of input data
     * @param targets Vector of target outputs
     * @param epochs Number of training epochs
     * @param learning_rate Learning rate for weight updates
     **********************************************************************/
    void TrainNetwork(const std::vector<std::vector<int>> &inputs, const std::vector<int> &targets, int epochs, double learning_rate);

    /**********************************************************************
     * @brief Predicts the output based on the input data
     * @param input Vector of input data
     * @return Predicted output value
     **********************************************************************/
    int Predict(const std::vector<int> &input);


    // Setter method for GPIO
    void SetGPIO(gpiod_chip* chip, gpiod_line* ledLine, const std::vector<gpiod_line*>& buttonLines) {
        this->chip = chip;
        this->ledLine = ledLine;
        this->buttonLines = buttonLines;
    }


private:
    struct gpiod_chip *chip; /**< Pointer to the GPIO chip */
    struct gpiod_line *ledLine; /**< Pointer to the LED line */
    std::vector<struct gpiod_line *> buttonLines; /**< Vector of pointers to button lines */

    // Neural network parameters
    uint8_t input_nodes; /**< Number of input nodes */
    uint8_t hidden_nodes; /**< Number of hidden layer nodes */
    uint8_t output_nodes; /**< Number of output nodes */
    std::vector<std::vector<double>> weights_input_hidden; /**< Weights between input and hidden layers */
    std::vector<std::vector<double>> weights_hidden_output; /**< Weights between hidden and output layers */
    std::vector<double> bias_hidden; /**< Biases of hidden layer nodes */
    std::vector<double> bias_output; /**< Biases of output layer nodes */
    std::vector<double> hidden_outputs; /**< Outputs of hidden layer neurons */
    double learning_rate; /**< Learning rate for weight updates */
    std::vector<int> input; /**< Input data used in backpropagation */
    std::vector<double> output; /**< Actual output from the network during forward pass */

    // Neural network methods

    /**********************************************************************
     * @brief Reads the states of the buttons
     * @return Vector containing the states of the buttons
     **********************************************************************/
    std::vector<int> ReadButtonStates();

    /**********************************************************************
     * @brief Controls the LED based on the given state
     * @param state The state to set the LED to
     **********************************************************************/
    void ControlLED(bool state);


    /**********************************************************************
     * @brief Rectified Linear Unit (ReLU) activation function
     * @param x Input value
     * @return Output value after applying ReLU
     **********************************************************************/
    double ReLU(double x);

    /**********************************************************************
     * @brief Initializes the weights and biases of the neural network
     **********************************************************************/
    void InitializeWeights();

    /**********************************************************************
     * @brief Performs forward propagation in the neural network
     * @param input Vector of input data
     **********************************************************************/
    void ForwardPropagation(const std::vector<int> &input);

    /**********************************************************************
     * @brief Performs backpropagation in the neural network
     * @param target Target output value
     **********************************************************************/
    void BackPropagation(int target);


};

} // namespace machine_learning
} // namespace yrgo
