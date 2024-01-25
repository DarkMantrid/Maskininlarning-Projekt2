#include "../inc/node.hpp"
#include "../inc/gpio_utils.hpp"

/**********************************************************************
 * @brief Main function controlling the machine learning system
 * @return Exit status of the program
 **********************************************************************/
int main() {
    yrgo::machine_learning::NeuralNetwork neuralNetwork{4, 10, 1, 0.01};
    const std::vector<std::vector<double>> trainingData = {
        {0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 0, 1, 1},
        {0, 1, 0, 0}, {0, 1, 0, 1}, {0, 1, 1, 0}, {0, 1, 1, 1},
        {1, 0, 0, 0}, {1, 0, 0, 1}, {1, 0, 1, 0}, {1, 0, 1, 1},
        {1, 1, 0, 0}, {1, 1, 0, 1}, {1, 1, 1, 0}, {1, 1, 1, 1}
    };

    const std::vector<double> labels = {0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0};
    neuralNetwork.TrainNetwork(trainingData, labels, 10000);
    neuralNetwork.PrintPredictions(trainingData, 2);
    return 0; 
}