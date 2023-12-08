#pragma once

#include <cstdint>
#include <gpiod.h>
#include <unistd.h>

namespace GPIOUtils {

enum GPIODLineDirection { GPIOD_LINE_DIRECTION_IN, GPIOD_LINE_DIRECTION_OUT };

enum GPIODLineEdge { GPIOD_LINE_EDGE_RISING, GPIOD_LINE_EDGE_FALLING, GPIOD_LINE_EDGE_BOTH };

gpiod_chip* get_gpiod_chip0();

gpiod_line* gpiod_line_new(const uint8_t pin, const GPIODLineDirection direction);

void gpiod_line_toggle(gpiod_line* self);

void gpiod_line_blink(gpiod_line* self, const uint16_t blink_speed_ms);

bool gpiod_line_event_detected(gpiod_line* self, const GPIODLineEdge edge, uint8_t* previous_input);

void delay_ms(const uint16_t delay_time_ms);

} // namespace GPIOUtils
