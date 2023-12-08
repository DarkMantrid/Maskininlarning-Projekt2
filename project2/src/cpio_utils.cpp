#include "../inc/gpio_utils.hpp"

namespace GPIOUtils {

gpiod_chip* get_gpiod_chip0() {
    static gpiod_chip* chip0 = nullptr;
    if (!chip0) chip0 = gpiod_chip_open("/dev/gpiochip0"); // Opens chip 0 at startup.
    return chip0;
}

gpiod_line* gpiod_line_new(const uint8_t pin, const GPIODLineDirection direction) {
    gpiod_line* self = gpiod_chip_get_line(get_gpiod_chip0(), pin);
    if (direction == GPIOD_LINE_DIRECTION_IN) {
        gpiod_line_request_input(self, "");
    } else {
        gpiod_line_request_output(self, "", 0);
    }
    return self;
}

void gpiod_line_toggle(gpiod_line* self) {
    gpiod_line_set_value(self, !gpiod_line_get_value(self));
}

void gpiod_line_blink(gpiod_line* self, const uint16_t blink_speed_ms) {
    gpiod_line_toggle(self);
    delay_ms(blink_speed_ms);
}

bool gpiod_line_event_detected(gpiod_line* self, const GPIODLineEdge edge, uint8_t* previous_input) {
    delay_ms(50);
    const uint8_t old_val = *previous_input;
    const uint8_t new_val = gpiod_line_get_value(self);
    *previous_input = new_val;

    if (old_val == new_val) {
        return false;
    } else {
        if (edge == GPIOD_LINE_EDGE_RISING) {
            return new_val && !old_val ? true : false;
        } else if (edge == GPIOD_LINE_EDGE_FALLING) {
            return !new_val && old_val ? true : false;
        } else {
            return true;
        }
    }
}

void delay_ms(const uint16_t delay_time_ms) {
    usleep(delay_time_ms * 1000);
}

} // namespace GPIOUtils
