#include "../inc/gpio_utils.hpp"

namespace GPIOUtils {

/**********************************************************************
 * @brief Gets the GPIO chip
 * @return Pointer to the GPIO chip
 **********************************************************************/
gpiod_chip* get_gpiod_chip0() {
    static gpiod_chip* chip0 = nullptr;
    if (!chip0) chip0 = gpiod_chip_open("/dev/gpiochip0"); // Opens chip 0 at startup.
    return chip0;
}

/**********************************************************************
 * @brief Creates a new GPIO line
 * @param pin GPIO pin number
 * @param direction Direction of the GPIO line
 * @return Pointer to the created GPIO line
 **********************************************************************/
gpiod_line* gpiod_line_new(const uint8_t pin, const GPIODLineDirection direction) {
    gpiod_line* self = gpiod_chip_get_line(get_gpiod_chip0(), pin);
    if (direction == GPIODLineDirection::In) {
        gpiod_line_request_input(self, "");
    } else {
        gpiod_line_request_output(self, "", 0);
    }
    return self;
}

/**********************************************************************
 * @brief Toggles the state of a GPIO line
 * @param self Pointer to the GPIO line
 **********************************************************************/
void gpiod_line_toggle(gpiod_line* self) {
    gpiod_line_set_value(self, !gpiod_line_get_value(self));
}

/**********************************************************************
 * @brief Blinks a GPIO line at a specified speed
 * @param self Pointer to the GPIO line
 * @param blink_speed_ms Blink speed in milliseconds
 **********************************************************************/
void gpiod_line_blink(gpiod_line* self, const uint16_t blink_speed_ms) {
    gpiod_line_toggle(self);
    delay_ms(blink_speed_ms);
}

/**********************************************************************
 * @brief Detects an event on a GPIO line based on edge detection
 * @param self Pointer to the GPIO line
 * @param edge Edge to detect (rising, falling, or both)
 * @param previous_input Pointer to previous input state
 * @return True if an event is detected, false otherwise
 **********************************************************************/
bool gpiod_line_event_detected(gpiod_line* self, const GPIODLineEdge edge, uint8_t* previous_input) {
    delay_ms(50);
    const uint8_t old_val = *previous_input;
    const uint8_t new_val = gpiod_line_get_value(self);
    *previous_input = new_val;

    if (old_val == new_val) {
        return false;
    } else {
        if (edge == GPIODLineEdge::RISING_EDGE) {
            return new_val && !old_val ? true : false;
        } else if (edge == GPIODLineEdge::FALLING_EDGE) {
            return !new_val && old_val ? true : false;
        } else {
            return true;
        }
    }
}

/**********************************************************************
 * @brief Delays execution for a specified time in milliseconds
 * @param delay_time_ms Time to delay in milliseconds
 **********************************************************************/
void delay_ms(const uint16_t delay_time_ms) {
    usleep(delay_time_ms * 1000);
}

} // namespace GPIOUtils