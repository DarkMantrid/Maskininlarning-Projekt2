#include "../inc/node.hpp"
#include "../inc/gpio_utils.hpp"

/**********************************************************************
 * @brief Main function controlling the machine learning system
 * @return Exit status of the program
 **********************************************************************/
int main() {
    yrgo::machine_learning::Node neuralNetwork; // Create an instance of the Node class

    // Initialize GPIO pins for buttons and LED
    neuralNetwork.InitializeGPIO();

    // Run the system continuously
    while (true) {
        // Predict LED state based on button inputs and control LED
        neuralNetwork.PredictAndControlLED();

        // Optional delay to avoid rapid predictions (adjust as needed)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0; // Return 0 to indicate successful execution
}
