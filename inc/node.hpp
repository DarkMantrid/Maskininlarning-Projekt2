#pragma once

#include <vector>
#include <gpiod.h>
#include <iostream>
#include <random>
#include <cmath>
#include <chrono>
#include <thread>
#include <iomanip>
#include <type_traits>


namespace yrgo {
namespace machine_learning {


/**********************************************************************
 * @brief Class representing a NeuralNetwork in a machine learning system
 **********************************************************************/
class NeuralNetwork {
public:

    NeuralNetwork() = delete;

    /**********************************************************************
     * @brief Default constructor for NeuralNetwork class
     **********************************************************************/
    NeuralNetwork(const std::size_t num_inputs, 
                  const std::size_t num_hidden, 
                  const std::size_t num_outputs,
                  const double learning_rate = 0.01);

    /**********************************************************************
     * @brief Destructor for NeuralNetwork class
     **********************************************************************/
    ~NeuralNetwork();

    /**********************************************************************
     * @brief Initializes the GPIO chip and lines
     **********************************************************************/
    void InitializeGPIO();

    /**********************************************************************
     * @brief Trains the neural network using backpropagation
     * @param inputs Vector of input data
     * @param targets Vector of target outputs
     * @param epochs Number of training epochs
     * @param learning_rate Learning rate for weight updates
     **********************************************************************/
    void TrainNetwork(const std::vector<std::vector<double>> &inputs, const std::vector<double> &targets, int epochs);

    /**********************************************************************
     * @brief Predicts the output based on the input data
     * @param input Vector of input data
     * @return Predicted output value
     **********************************************************************/
    double Predict(const std::vector<double> &input);


    // Setter method for GPIO
    void SetGPIO(gpiod_chip* chip, gpiod_line* ledLine, const std::vector<gpiod_line*>& buttonLines) {
        this->chip = chip;
        this->ledLine = ledLine;
        this->buttonLines = buttonLines;
    }


    /********************************************************************************
     * @brief Performs predictions with all input sets and prints the output.
     * 
     * @param input_sets   Reference to vector holding all input sets to predict with.
     * @param num_decimals The number of decimals to print (default = 0).
     * @param ostream      Reference to output stream (default = terminal print).
     ********************************************************************************/
    void PrintPredictions(const std::vector<std::vector<double>>& input_sets,
                          const std::size_t num_decimals = 0,
                          std::ostream& ostream = std::cout);


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
    std::vector<double> input; /**< Input data used in backpropagation */
    std::vector<double> output; /**< Actual output from the network during forward pass */
    std::vector<double> output_errors;
    std::vector<double> hidden_errors;

    // Neural network methods

    /**********************************************************************
     * @brief Reads the states of the buttons
     * @return Vector containing the states of the buttons
     **********************************************************************/
    std::vector<double> ReadButtonStates();

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

    double ReLUDelta(double x);

    double TanH(double x);

    double TanHDelta(double x);

    /**********************************************************************
     * @brief Initializes the weights and biases of the neural network
     **********************************************************************/
    void InitializeWeights();

    /**********************************************************************
     * @brief Performs forward propagation in the neural network
     * @param input Vector of input data
     **********************************************************************/
    void ForwardPropagation(const std::vector<double> &input);

    /**********************************************************************
     * @brief Performs backpropagation in the neural network
     * @param target Target output value
     **********************************************************************/
    void BackPropagation(double target);

    /**********************************************************************
     * @brief Optimizes the neural network parameters using the provided input
     * @param input Input vector for optimization
     **********************************************************************/
    void Optimize(const std::vector<double> &input);


};

} // namespace machine_learning
} // namespace yrgo
