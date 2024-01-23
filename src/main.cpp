#include "../inc/node.hpp"
#include "../inc/gpio_utils.hpp"

/**********************************************************************
 * @brief Main function controlling the machine learning system
 * @return Exit status of the program
 **********************************************************************/
int main() {
    yrgo::machine_learning::NeuralNetwork neuralNetwork; // Create an instance of the NeuralNetwork class
    neuralNetwork.PredictAndControlLED();
    return 0; // Return 0 to indicate successful execution
}
