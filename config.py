"""
Copyright © 2025 David Mulnix. All rights reserved.

Licensing:
Permission is hereby granted to use, modify, and distribute this software for non-commercial research and educational purposes only, provided that proper attribution is given.

Commercial use—including but not limited to use in proprietary software, paid services, or for-profit research—requires prior written permission from the author.

This software is provided "as is", without warranty of any kind, express or implied. The author retains all intellectual property rights and reserves the right to grant commercial licenses at their discretion.

Disclaimer:
THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE, AND NON-INFRINGEMENT. 

IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

THE ENTIRE RISK AS TO THE QUALITY AND PERFORMANCE OF THE SOFTWARE IS WITH YOU. SHOULD THE SOFTWARE PROVE DEFECTIVE, YOU ASSUME THE COST OF ALL NECESSARY SERVICING, REPAIR, OR CORRECTION.

Scientific Disclaimer:

This simulator is intended for research and educational purposes only. While it implements physically grounded models and symbolic shaping layers inspired by Rayleigh-Plesset dynamics, the results produced are sensitive to parameter selection, numerical resolution, and modulation configuration. No guarantee is made regarding the empirical accuracy, predictive reliability, or suitability of the outputs for experimental replication or engineering deployment.

Users are responsible for validating the simulator’s behavior under their specific use cases. The author disclaims any liability for misinterpretation, misuse, or unintended consequences arising from the application of this software. All simulations should be interpreted within the context of their assumptions, and users are encouraged to consult domain experts when applying the simulator to critical or real-world scenarios.


"""

import numpy as np

# Simulation domain
total_time = 35e-6  # 50 microseconds
num_points = 10000
dt = total_time / num_points


# H2O in 300K  https://iopscience.iop.org/article/10.1088/0143-0807/34/3/679
Saturation_vapor_pressure = 3539   # [Pa] 
Dynamic_viscosity = 1.002e-3         # [Pa.s]
Ambient_pressure = 101325          # [Pa] https://arxiv.org/pdf/1111.5229, the static ambient pressure 
Acoustic_pressure_scale = 1.42         #  {value}* ambient_pressure, the amplitude of the driven acoustic pressure (dynamic)
Acoustic_pressure = Acoustic_pressure_scale * Ambient_pressure  # [Pa]
Argon_hard_core_radius = 2.25733634e-7        # Hard core (argon)  2.0e-06/8.86
Bubble_radius = 2.0e-06        # Initial bubble radius [m]  Req = 6.865530804666495e-06
Frequency = 26.5e+3                  # [Hz]
Angular_frequency = 2 * np.pi * Frequency  # [rad/s]
Adiabatic_index = 1.6666666    
Surface_tension = 72.8e-3          # [N/m]
Fluid_density = 1000                # [kg/m^3] https://ineeringtoolbox.com/water-vapor-saturation-pressure-d_599.html



#Additional parameters
Magnetic_permeability = 1.256637e-6 # https://arxiv.org/pdf/1111.5229
#Expected_flash_offset_ns = 5.0 # https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.80.1090
#dt_ns = dt * 1e9  # Time step in nanoseconds, flash timing and collapse profiling

fluid_properties = {
    "density": Fluid_density,
    "viscosity": Dynamic_viscosity,
    "surface_tension": Surface_tension,
    "vapor_pressure": Saturation_vapor_pressure
}


