import numpy as np
import matplotlib.pyplot as plt


def partial_pressure_vapor(T: float, relative_humidity:float) -> float:
    T_celsius = T - 273.15
    p_sat_kPa = 0.61121 * np.exp((18.678 - T_celsius/234.5) * (T_celsius / (257.14 + T_celsius)))
    saturation_pressure = p_sat_kPa * 1000 #Convert to [Pa]
    partial_pressure_vapor = relative_humidity * saturation_pressure
    return partial_pressure_vapor

input_relative = 0.832
output_relative = 1

temperatures_celsius = np.linspace(23,10, 200)
temperatures_kelvin = temperatures_celsius + 273.15

temperature_offsets_celsius = np.array([1, 2, 4, 5, 10, 13])

humidity_input = np.array([
    0.622 * (
        partial_pressure_vapor(T, input_relative) / (101325 - partial_pressure_vapor(T, input_relative))
    )
    for T in temperatures_kelvin
])

humidity_output_zero_offset = np.array([
    0.622 * (
        partial_pressure_vapor(T, output_relative) / (101325 - partial_pressure_vapor(T, output_relative))
    )
    for T in temperatures_kelvin
])

plt.figure()
cmap = plt.cm.viridis
plt.plot(temperatures_celsius, humidity_input, label="Input RH = 0.832", color=cmap(0.15))
plt.plot(temperatures_celsius, humidity_output_zero_offset, label="Output RH = 1.0", color=cmap(0.85))
plt.xlabel("Temperature [°C]")
plt.ylabel("Specific Humidity [g/kg]")
plt.title("Specific Humidity vs Temperature")
plt.grid(True)
plt.legend()
plt.gca().invert_xaxis()
plt.xlim(temperatures_celsius.max(), temperatures_celsius.min())
plt.figure()

colors = cmap(np.linspace(0.1, 0.9, len(temperature_offsets_celsius)))
for offset, color in zip(temperature_offsets_celsius, colors):
    output_temperatures_kelvin = temperatures_kelvin - offset
    humidity_output_offset = np.array([
        0.622 * (
            partial_pressure_vapor(T, output_relative) / (101325 - partial_pressure_vapor(T, output_relative))
        )
        for T in output_temperatures_kelvin
    ])
    humidity_difference = humidity_input - humidity_output_offset
    plt.plot(temperatures_celsius, humidity_difference, label=f"Offset = {offset:.0f}°C", color=color)

plt.xlabel("Input temperature [°C]")
plt.ylabel("Specific humidity difference [g/kg]")
plt.title("Specific humidity difference vs temperature for multiple output offsets")
plt.grid(True)
plt.legend()
plt.gca().invert_xaxis()
plt.xlim(temperatures_celsius.max(), temperatures_celsius.min())

input_temperature_constant_celsius = 23
input_temperature_constant_kelvin = input_temperature_constant_celsius + 273.15

humidity_input_constant = 0.622 * (
    partial_pressure_vapor(input_temperature_constant_kelvin, input_relative)
    / (101325 - partial_pressure_vapor(input_temperature_constant_kelvin, input_relative))
)

# create a continuous offset range for a smooth curve
offsets_continuous = np.linspace(temperature_offsets_celsius.min(), temperature_offsets_celsius.max(), 400)
humidity_difference_vs_offset = []
for offset in offsets_continuous:
    output_temperature_kelvin = input_temperature_constant_kelvin - offset
    humidity_output_constant = 0.622 * (
        partial_pressure_vapor(output_temperature_kelvin, output_relative)
        / (101325 - partial_pressure_vapor(output_temperature_kelvin, output_relative))
    )
    humidity_difference_vs_offset.append((humidity_input_constant - humidity_output_constant) * 1000)

plt.figure()
plt.plot(offsets_continuous, humidity_difference_vs_offset, color=cmap(0.5))
plt.xlabel("Temperature Drop [°C]")
plt.ylabel("Specific Humidity Change [g/kg]")
plt.title("$\omega_{in}$ - $\omega_{out}$ vs Temperature Drop from $T_{in} = 23\degree$C")
plt.grid(True)
plt.xlim(temperature_offsets_celsius.min(), temperature_offsets_celsius.max())

plt.show()