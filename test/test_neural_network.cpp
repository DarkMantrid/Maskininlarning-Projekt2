#include "gtest/gtest.h"
#include "../inc/node.hpp"  

class NeuralNetworkTest : public ::testing::Test {
protected:
    yrgo::machine_learning::NeuralNetwork neuralNetwork;  

public:
    NeuralNetworkTest() : neuralNetwork(4, 10, 1, 0.01) {}

    // Constructor to initialize neuralNetwork with provided parameters
    NeuralNetworkTest(const std::size_t num_inputs, 
                      const std::size_t num_hidden, 
                      const std::size_t num_outputs,
                      const double learning_rate)
        : neuralNetwork(num_inputs, num_hidden, num_outputs, learning_rate) {}

    // No need to implement TearDown() if there are no resources to clean up

};

// Test case to check the prediction accuracy of the NeuralNetwork class
TEST_F(NeuralNetworkTest, PredictionAccuracyWithinTolerance) {
    // Define test input data and expected output
    std::vector<std::vector<double>> input_sets = {{0, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 0, 1, 1},
                                                   {0, 1, 0, 0}, {0, 1, 0, 1}, {0, 1, 1, 0}, {0, 1, 1, 1},
                                                   {1, 0, 0, 0}, {1, 0, 0, 1}, {1, 0, 1, 0}, {1, 0, 1, 1},
                                                   {1, 1, 0, 0}, {1, 1, 0, 1}, {1, 1, 1, 0}, {1, 1, 1, 1}};
    std::vector<double> expected_outputs = {0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0};

    // Train the neural network
    neuralNetwork.TrainNetwork(input_sets, expected_outputs, 15000);

    // Validate predictions
    for (size_t i = 0; i < input_sets.size(); ++i) {
        double prediction = neuralNetwork.Predict(input_sets[i]);
        EXPECT_NEAR(prediction, expected_outputs[i], 0.01); // Tolerance for prediction accuracy
    }
}

// Entry point for running all the tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}