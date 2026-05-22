"""Energy and carbon estimation utilities inspired by Green Algorithms.

Examples
--------
>>> from core.energy import estimate_energy_and_carbon
>>> estimate_energy_and_carbon(runtime_seconds=10.0, average_power_watts=50.0)[0] > 0
True

Notes
-----
Author: vibe coding of Warith Harchaoui on top of Andrej Karpathy.
"""

from __future__ import annotations

from typing import Tuple

from .config import get_settings


def estimate_energy_and_carbon(
    runtime_seconds: float,
    average_power_watts: float,
    memory_gb: float = 0.0,
) -> Tuple[float, float]:
    """Estimate energy usage and carbon footprint for a single completion.

    Parameters
    ----------
    runtime_seconds:
        End-to-end runtime in seconds.
    average_power_watts:
        Average device power draw in watts.
    memory_gb:
        Approximate RAM footprint in gigabytes.

    Returns
    -------
    tuple of float
        Energy in kWh and carbon in grams CO2e.
    """
    settings = get_settings()
    total_power = average_power_watts + memory_gb * settings.memory_power_watts_per_gb
    energy_kwh = runtime_seconds * total_power * settings.power_usage_effectiveness / 3600.0 / 1000.0
    carbon_g = energy_kwh * settings.grid_carbon_intensity
    return energy_kwh, carbon_g
