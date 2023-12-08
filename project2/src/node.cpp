#include "../inc/node.hpp"
#include "../inc/gpio_utils.hpp"
#include <iostream>
#include <chrono>
#include <thread>

namespace yrgo {
namespace machine_learning {

void Node::InitializeGPIO() {
    // Connect to GPIO chip
    chip = gpiod_chip_open("/dev/gpiochip0");
    if (!chip) {
        std::cerr << "Error opening GPIO chip" << std::endl;
        // Handle error
    }

    // Open LED line
    constexpr int ledGPIO = 25; // Replace with actual GPIO pin for my led
    ledLine = gpiod_chip_get_line(chip, ledGPIO);
    if (!ledLine) {
        std::cerr << "Error getting LED line" << std::endl;
        // Handle error
    }

    if (gpiod_line_request_output(ledLine, "LED", 0) < 0) {
        std::cerr << "Error requesting LED line" << std::endl;
        // Handle error
    }

    // GPIO numbers for buttons
    std::vector<int> buttonGPIOLines = {17, 18, 19}; // Replace with your actual GPIO numbers for buttons

    // Open and request lines for buttons
    for (int buttonNum = 0; buttonNum < buttonGPIOLines.size(); ++buttonNum) {
        struct gpiod_line *buttonLine = gpiod_chip_get_line(chip, buttonGPIOLines[buttonNum]);
        if (!buttonLine) {
            std::cerr << "Error getting button line " << buttonNum << std::endl;
            // Handle error
        }

        if (gpiod_line_request_input(buttonLine, "Button") < 0) {
            std::cerr << "Error requesting button line " << buttonNum << std::endl;
            // Handle error
        }

        buttonLines.push_back(buttonLine);
    }
}


std::vector<int> Node::ReadButtonStates() {
    std::vector<int> buttonStates;

    for (const auto& buttonLine : buttonLines) {
        int state = gpiod_line_get_value(buttonLine);
        buttonStates.push_back(state);
    }

    return buttonStates;
}

void Node::ControlLED(bool state) {
    // Set LED state based on prediction
    int value = state ? 1 : 0;
    if (gpiod_line_set_value(ledLine, value) < 0) {
        std::cerr << "Error setting LED state" << std::endl;
        // Handle error
    }
}

void Node::PredictAndControlLED() {
    // Read button states
    std::vector<int> buttonStates = ReadButtonStates();

    // Implement neural network prediction logic based on buttonStates
    // For the sake of example, a dummy prediction is made here
    bool predictedState = (buttonStates.size() % 2 != 0);

    // Control LED based on prediction
    ControlLED(predictedState);
}

} // namespace machine_learning
} // namespace yrgo
