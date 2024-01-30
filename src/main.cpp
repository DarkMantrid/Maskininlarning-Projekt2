#include "../inc/node.hpp"
#include "../inc/gpio_utils.hpp"

/**********************************************************************
 * @brief Main function controlling the machine learning system
 * @return Exit status of the program
 **********************************************************************/
int main() {
    yrgo::machine_learning::NeuralNetwork neuralNetwork{4, 10, 1, 0.01}; // Create an instance of the NeuralNetwork class

    const std::vector<std::vector<double>> trainingData = {
        {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 0, 1, 1},
        {0, 1, 0, 0}, {0, 1, 0, 1}, {0, 1, 1, 0}, {0, 1, 1, 1},
        {1, 0, 0, 0}, {1, 0, 0, 1}, {1, 0, 1, 0}, {1, 0, 1, 1},
        {1, 1, 0, 0}, {1, 1, 0, 1}, {1, 1, 1, 0}, {1, 1, 1, 1}
    };
    const std::vector<double> labels = {0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0};
    neuralNetwork.TrainNetwork(trainingData, labels, 15000);
    neuralNetwork.PrintPredictions(trainingData, 2);
    // Initialize GPIO pins for buttons and LED
    neuralNetwork.InitializeGPIO();

    // Run the system continuously
    while (true) {
        // Predict LED state based on button inputs and control LED
        neuralNetwork.PredictAndControlLED();
        // Introduce a delay to avoid tight loop
        usleep(50000); // 50ms delay
    } 

    return 0; // Return 0 to indicate successful execution
}

