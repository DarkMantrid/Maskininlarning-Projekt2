#include "../inc/node.hpp"
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
    ledLine = gpiod_chip_get_line(chip, /* LED GPIO Line Number */);
    if (!ledLine) {
        std::cerr << "Error getting LED line" << std::endl;
        // Handle error
    }

    // Request LED line for output
    if (gpiod_line_request_output(ledLine, "LED", 0) < 0) {
        std::cerr << "Error requesting LED line" << std::endl;
        // Handle error
    }

    // Open and request lines for buttons (adjust the button GPIO line numbers)
    for (int buttonNum = 0; buttonNum < /* Number of buttons */; ++buttonNum) {
        struct gpiod_line *buttonLine = gpiod_chip_get_line(chip, /* Button GPIO Line Number */);
        if (!buttonLine) {
            std::cerr << "Error getting button line" << std::endl;
            // Handle error
        }

        if (gpiod_line_request_input(buttonLine, "Button") < 0) {
            std::cerr << "Error requesting button line" << std::endl;
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
