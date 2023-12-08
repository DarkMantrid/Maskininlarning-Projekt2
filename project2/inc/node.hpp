#pragma once

#include <vector>
#include <gpiod.h> // Include libgpiod headers

namespace yrgo {
namespace machine_learning {

class Node {
public:
    // Existing methods for Node class

    // New method to handle prediction based on button inputs and LED control
    void PredictAndControlLED();

    // Methods for GPIO initialization, button reading, and LED control
    void InitializeGPIO(); // Initialize GPIO pins
    std::vector<int> ReadButtonStates(); // Read button states
    void ControlLED(bool state); // Control LED based on prediction

private:
    // Add GPIO related members (GPIO chip, line pointers, etc.) here
    struct gpiod_chip *chip;
    struct gpiod_line *ledLine;
    std::vector<struct gpiod_line*> buttonLines;
};

} // namespace machine_learning
} // namespace yrgo
