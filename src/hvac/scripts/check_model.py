import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from models import HeatExchangerModel


model = HeatExchangerModel(
    model = "heater",
    volume_flow_rate = 0.5,   # [m³/s]
    cross_section_area = 0.5, # [m²]
    num_segments = 3,
)
