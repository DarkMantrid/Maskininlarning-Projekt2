#include "../inc/node.hpp"
#include "../inc/gpio_utils.hpp"


int main() {
    yrgo::machine_learning::Node neuralNetwork;

    // Initialize GPIO pins for buttons and LED
    neuralNetwork.InitializeGPIO();

    // Run the system
    while (true) {
        // Predict LED state based on button inputs and control LED
        neuralNetwork.PredictAndControlLED();

        // Optional delay to avoid rapid predictions (adjust as needed)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    return 0;
}
