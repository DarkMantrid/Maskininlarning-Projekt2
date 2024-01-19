#pragma once

#include <cstdint>
#include <gpiod.h>
#include <unistd.h>

namespace GPIOUtils {

/**********************************************************************
 * @brief Enum representing GPIO line direction
 **********************************************************************/
enum GPIODLineDirection { In, Out };

/**********************************************************************
 * @brief Enum representing GPIO line edge
 **********************************************************************/
enum GPIODLineEdge { RISING_EDGE, FALLING_EDGE, BOTH_EDGE };

/**********************************************************************
 * @brief Gets the GPIO chip
 * @return Pointer to the GPIO chip
 **********************************************************************/
gpiod_chip* get_gpiod_chip0();

/**********************************************************************
 * @brief Creates a new GPIO line
 * @param pin GPIO pin number
 * @param direction Direction of the GPIO line
 * @return Pointer to the created GPIO line
 **********************************************************************/
gpiod_line* gpiod_line_new(const uint8_t pin, const GPIODLineDirection direction);

/**********************************************************************
 * @brief Toggles the state of a GPIO line
 * @param self Pointer to the GPIO line
 **********************************************************************/
void gpiod_line_toggle(gpiod_line* self);

/**********************************************************************
 * @brief Blinks a GPIO line at a specified speed
 * @param self Pointer to the GPIO line
 * @param blink_speed_ms Blink speed in milliseconds
 **********************************************************************/
void gpiod_line_blink(gpiod_line* self, const uint16_t blink_speed_ms);

/**********************************************************************
 * @brief Detects an event on a GPIO line based on edge detection
 * @param self Pointer to the GPIO line
 * @param edge Edge to detect (rising, falling, or both)
 * @param previous_input Pointer to previous input state
 * @return True if an event is detected, false otherwise
 **********************************************************************/
bool gpiod_line_event_detected(gpiod_line* self, const GPIODLineEdge edge, uint8_t* previous_input);

/**********************************************************************
 * @brief Delays execution for a specified time in milliseconds
 * @param delay_time_ms Time to delay in milliseconds
 **********************************************************************/
void delay_ms(const uint16_t delay_time_ms);

} // namespace GPIOUtils
