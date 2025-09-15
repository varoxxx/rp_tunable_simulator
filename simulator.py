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
from scipy.integrate import solve_ivp
import config  # Assumes config.py is in the same directory
from scipy.interpolate import interp1d
from scipy.stats import skew
from scipy.stats import entropy
from metrics_csv import write_comparison_csv
from plots import plot_angular_comparison
from plots import plot_energy_opt_comparison, plot_diagnostics
from plots import plot_trc_comparison, plot_trc_energy
from plots import plot_qac_comparison
from plots import plot_temporal_comparison
from plots import plot_nf_ingress_comparison
from plots import plot_ingress_comparison
from plots import plot_nf_comparison
from plots import plot_nf_only_comparison
from plots import plot_rt_only_comparison
from plots import rt_nf_tm_qac_combined
from plots import tm_trc_qrc_combined
from plots import eopt_rt_qac_combined
from plots import resonance_yield_plot
from plots import resonance_yield_diagnostics
from plots import plot_coherence_surface
from plots import plot_rt_radius_growth



class Framework:
    @classmethod
    def from_config(cls):
        instance = cls.__new__(cls)
        instance.ssound = 1500.0 #speed of sound
        instance.freq = config.Frequency
        instance.fluidd = config.Fluid_density
        instance.bradi = config.Bubble_radius
        instance.amp = config.Acoustic_pressure
        instance.sft = config.Surface_tension    
        instance.pamb = config.Ambient_pressure
        instance.argon = config.Argon_hard_core_radius
        instance.dviscos = config.Dynamic_viscosity
        instance.psvp = config.Saturation_vapor_pressure
        instance.adiab = config.Adiabatic_index
        instance.P_gas_prev = -1
        instance.dt = config.dt
        #Ingress
        instance.beta_ingress = 1000.0
        instance.omega_ingress = 500.0
        instance.phi_ingress = 0.0
        instance.kappa_ingress = 0.1
        instance.apply_ingress = False
        #Temporal
        instance.apply_temporal = False
        instance.sigma_emission = 2e-6  # Default temporal spread
        # QAC
        instance.apply_qac = False
        instance.phi_acoustic = np.pi / 2
        instance.eta_ref = 1e-3
        # TRC
        instance.apply_trc = False
        instance.A1 = 2.0
        instance.A2 = 1.5
        instance.A3 = 1.0
        instance.omega1 = 1e5
        instance.omega2 = 2e5
        instance.omega3 = 3e5
        instance.phi1 = 0.0
        instance.phi2 = 0.0
        instance.phi3 = 0.0
        # Energy Optimization
        instance.apply_energy_opt = False  
        instance.theta_ingress = np.pi / 4  # Directional gating angle
        instance.mu_eff = 1.0  # Field-mediated damping
        instance.energy_modulation_strength = 0.0  # Scalar to control impact on collapse dynamics
        return instance


    def configure_tunables(self, tunables):

        for attr in ["apply_trc", "A1", "A2", "A3", "omega1", "omega2", "omega3", "phi1", "phi2", "phi3"]:
            if not hasattr(self, attr):
                setattr(self, attr, None)

        for key, value in tunables.items():
           # print(f"[CONFIGURE] Setting {key} = {value}")
            setattr(self, key, value)
        if not hasattr(self, "estimated_t_peak"):
            self.estimated_t_peak = None
        if not hasattr(self, "apply_nf"):
            self.apply_nf = False
        if not hasattr(self, "alpha_field"):
            self.alpha_field = 0.0
        if not hasattr(self, "f_ambient"):
            self.f_ambient = 0.0
        if not hasattr(self, "phi_nu"):
            self.phi_nu = 0.0
        if not hasattr(self, "kappa_nu"):
            self.kappa_nu = 0.0
        if not hasattr(self, "apply_ingress"):
            self.apply_ingress = False
        if not hasattr(self, "beta_ingress"):
            self.beta_ingress = 0.0
        if not hasattr(self, "omega_ingress"):
            self.omega_ingress = 0.0
        if not hasattr(self, "phi_ingress"):
            self.phi_ingress = 0.0
        if not hasattr(self, "kappa_ingress"):
            self.kappa_ingress = 0.0
        if not hasattr(self, "apply_temporal"):
            self.apply_temporal = False
        if not hasattr(self, "sigma_emission"):
            self.sigma_emission = 2e-6
        if not hasattr(self, "apply_qac"):
            self.apply_qac = False
        if not hasattr(self, "phi_acoustic"):
            self.phi_acoustic = np.pi / 2
        if not hasattr(self, "eta_ref"):
            self.eta_ref = 1e-3
        if not hasattr(self, "apply_energy_opt"):
            self.apply_energy_opt = False
        if not hasattr(self, "theta_ingress"):
            self.theta_ingress = np.pi / 4
        if not hasattr(self, "mu_eff"):
            self.mu_eff = 1.0
        if not hasattr(self, "energy_modulation_strength"):
            self.energy_modulation_strength = 0.0
        if not hasattr(self, "apply_angular_pressure"):
            self.apply_angular_pressure = False
        if not hasattr(self, "alpha_theta"):
            self.alpha_theta = 0.0
        if not hasattr(self, "omega_theta"):
            self.omega_theta = 0.0
        if not hasattr(self, "phi_theta"):
            self.phi_theta = 0.0
        if not hasattr(self, "kappa_theta"):
            self.kappa_theta = 0.0
        if not hasattr(self, "theta_sample"):
            self.theta_sample = np.pi / 4





    @staticmethod
    def thermal_diffusivity():
        """
        Computes thermal diffusivity (α) using the standard physical relation:
        
            α = k / (ρ · c_p)
        
        Where:
        - k: thermal conductivity [W/(m·K)]
        - ρ: gas density [kg/m³]
        - c_p: specific heat at constant pressure [J/(kg·K)]
        
        This formulation is consistent with classical heat transfer models 
        and is commonly used in bubble dynamics and gas-phase simulations.
        
        Returns:
        - Thermal diffusivity [m²/s]
        """
        wmk = 0.025       # W/(m·K) 
        jkg = 1996       # J/(kg·K)
        kgm = 0.6       # kg/m³
        return wmk / (jkg * kgm)


 
    @staticmethod
    def dynamic_polytropic_exponent(R, dR, gamma):
        """
        Computes the effective polytropic exponent (κ) to account for thermal damping
        in bubble dynamics, based on the Péclet number formulation:

            κ = 1 + (γ - 1) · exp(-A / Pe^B)

        Where:
        - γ: adiabatic index of the gas
        - Pe: Péclet number = (R · |dR|) / α
        - α: thermal diffusivity [m²/s]
        - A, B: empirical constants tuned to match experimental data

        This model reflects the transition from adiabatic to isothermal behavior
        due to thermal diffusion, as described in studies such as:
        - Yasui et al., Ultrasonics Sonochemistry (2014) [ScienceDirect]
        - Brenner et al., Nature (1995)

        Returns:
        - κ: effective polytropic exponent
        """
        thermaldiff = Framework.thermal_diffusivity()
        Pe = np.maximum(R * np.abs(dR) / thermaldiff, 1e-7) # Avoid division by zero
        A, B = 0.0036781, 1.0 # Empirical fit parameters
        return 1 + (gamma - 1) * np.exp(-A / Pe**B)


    def compute_resonance_yield(self, t_array, R_array):
        theta_vals = np.linspace(0, np.pi, 100)
        omega3 = getattr(self, "omega3", 3e5)
        lambda_decay = getattr(self, "lambda_decay", 1e4)
        sigma_emission = getattr(self, "sigma_emission", 2e-6)
        coherence_strength = getattr(self, "coherence_weighting", 0.0)
        t_peak = self.estimated_t_peak if self.estimated_t_peak else t_array[np.argmin(R_array)]
    
        def coherence_index(t, theta):
            if coherence_strength == 0.0:
                return 1.0  # No modulation
    
            # Simple coherence shaping: angular + temporal alignment
            base = np.abs(np.sin(theta)) * np.exp(-((t - t_peak) / sigma_emission)**2)
            val = coherence_strength * base
            
    
            return 1.0 if np.isnan(val) or np.isinf(val) else val
    
        def R_RP_theta(t, theta):
            idx = np.searchsorted(t_array, t)
            val = R_array[idx] if 0 <= idx < len(R_array) else 0.0
            return 0.0 if np.isnan(val) or np.isinf(val) else val
    
        R_Yield = []
        for t in t_array:
            tau = (t - t_peak) / sigma_emission
            exponent = -lambda_decay * tau
            exponent = np.clip(exponent, -700, 700)
            damping = np.exp(exponent)
            damping = 0.0 if np.isnan(damping) or np.isinf(damping) else damping
    
            integrand = []
            for theta in theta_vals:
                try:
                    value = (
                        R_RP_theta(t, theta) *
                        damping *
                        np.sin(omega3 * theta) *
                        coherence_index(t, theta)
                    )
                    if np.isnan(value) or np.isinf(value):
                        value = 0.0
                except:
                    value = 0.0
                integrand.append(value)
    
            integrand = np.array(integrand)
            integrand = np.nan_to_num(integrand, nan=0.0, posinf=0.0, neginf=0.0)
            R_val = np.trapz(integrand, theta_vals)
            R_Yield.append(R_val)
    
        R_Yield = np.array(R_Yield)
    
        # 🔧 Normalize the result to match the scale of R_array
        max_R = np.max(R_array)
        max_Yield = np.max(R_Yield)
        if max_Yield > 0:
            R_Yield *= (max_R / max_Yield)
    
        return R_Yield




    def angular_pressure_modulation(self, t, R):
        """
        Computes angular-phase pressure modulation based on directional forcing.
        """
        if not hasattr(self, "theta_sample"):
            theta = np.pi / 4  # Default angle
        else:
            theta = self.theta_sample
    
        P_drive = self.acoustic_pressure_waveform(t)
        print(f"[ANGULAR] P_drive={P_drive:.3e}")

        angular_term = 1 + self.alpha_theta * np.sin(2 * np.pi * self.omega_theta * theta + self.phi_theta) * np.exp(-self.kappa_theta * t)
        P_mod = P_drive * angular_term / R
        return P_mod



    def energy_optimization(self, R, dR, t=None):
        """
        Computes directional kinetic energy yield gated by ingress geometry and modulated by field interaction.
        Includes optional time gating to control when energy optimization is emphasized.
        """
        if R <= 0 or self.mu_eff <= 0:
            print("[E_OPT] Invalid R or mu_eff — suppressing energy output.")
            return 0.0
    
        volume = (4 / 3) * np.pi * R**3
        kinetic_energy = self.fluidd * volume * dR**2
        gating = np.sin(self.theta_ingress)
        damping = max(R * self.mu_eff, 1e-12)  # ✅ Prevent divide-by-zer
    
        if gating <= 0 and not getattr(self, "allow_negative_gating", True):
          #  print("Gating is zero or negative — suppressing energy output.")
            return 0.0

    
        energy = kinetic_energy * gating / damping
    
    ###
        if t is not None:
            if hasattr(self, "energy_gate_center") and hasattr(self, "energy_gate_width"):
                time_gate = np.exp(-((t - self.energy_gate_center) / self.energy_gate_width)**2)
                energy *= time_gate

        print(f"[E_OPT] t = {t}, gate_center = {self.energy_gate_center}, gate_width = {self.energy_gate_width}")

        return energy



    def trc_modulation(self, t, R):
        term1 = self.A1 * R * np.sin(self.omega1 * t + self.phi1)
        term2 = self.A2 * R * np.sin(self.omega2 * t + self.phi2)
        term3 = self.A3 * R * np.sin(self.omega3 * t + self.phi3)
        return (term1 + term2 + term3) ** 2



    def qac_modulation(self, t, dR):
        numerator = np.sin(self.phi_acoustic) * np.exp(dR * t)
        denominator = 1 + (self.dviscos / self.eta_ref) ** 2
        return numerator / denominator



    def temporal_modulation(self, t):
        if self.estimated_t_peak is None:
            return 1.0
        return np.exp(-((t - self.estimated_t_peak) / self.sigma_emission) ** 2)



    def ingress_modulation(self, t, R):
        return 1 + self.beta_ingress * R * self.omega_ingress * np.sin(2 * np.pi * self.omega_ingress * t + self.phi_ingress) * np.exp(-self.kappa_ingress * t)


    def ambient_field_modulation(self, t):
        return 1 + self.alpha_field * np.sin(2 * np.pi * self.f_ambient * t + self.phi_nu) * np.exp(-self.kappa_nu * t)


    def acoustic_pressure_waveform(self, t):
        """Generates time-dependent acoustic forcing signal."""
        omega = 2 * np.pi * self.freq
        return -self.amp * np.sin(omega * t)
    
    def symbolic_radiation_loss(self, R, dR, p_gas):
        """Modular wrapper for acoustic radiation damping."""
        if not hasattr(self, "_radiation_initialized"):
            self._radiation_initialized = True
            self._skip_first_call = True
        else:
            self._skip_first_call = False
    
        if self._skip_first_call:
            return 0.0
    
        try:
            dPgdt = -3 * self.adiab * p_gas * R**2 * dR / (R**3 - self.argon**3)
            return dPgdt * R / self.ssound
        except ZeroDivisionError:
            return 0.0



    def collapse_dynamics(self, t, y):

  
        R, dR = y

        try:
            """Computes the gas pressure inside the bubble based on the Rayleigh-Plesset equation."""
            volume_initial = self.bradi**3 - self.argon**3
            volume_current = R**3 - self.argon**3
            if volume_current <= 0:
                volume_current = 1e-12  # Prevent divide-by-zero or negative volume
            effective_pressure = self.pamb + (2 * self.sft / self.bradi)
            p_gas = effective_pressure * (volume_initial / volume_current) ** self.adiab
        except Exception as e:
            print(f"Pressure computation failed at t={t}, R={R}: {e}")
            p_gas = self.pamb  # Fallback to ambient pressure

    
        if hasattr(self, "apply_angular_pressure") and self.apply_angular_pressure:
            theta = getattr(self, "theta_sample", np.pi / 4)
            base = np.sin(2 * np.pi * self.omega_theta * theta + self.phi_theta)
            envelope = np.exp(-self.kappa_theta * t)
            mod_factor = 1 + self.alpha_theta * base * envelope
            p_ext = self.acoustic_pressure_waveform(t) * (0.9 + 0.1 * mod_factor)
        
            # Track diagnostic
            if not hasattr(self, "angular_mod_list"):
                self.angular_mod_list = []
            self.angular_mod_list.append(p_ext)
        else:
            p_ext = self.acoustic_pressure_waveform(t)
    
        radiation_loss = self.symbolic_radiation_loss(R, dR, p_gas)

        # Compute inertial term
        inertial_term = -3 * dR**2 / (2 * R)
        
        # Embed ingress modulation if enabled
        if hasattr(self, "apply_ingress") and self.apply_ingress:
            ingress_factor = 1 + self.beta_ingress * R * self.omega_ingress * np.sin(2 * np.pi * self.omega_ingress * t + self.phi_ingress) * np.exp(-self.kappa_ingress * t)
            inertial_term *= ingress_factor
        
        # Final acceleration term with shaping layers
        ddR = inertial_term + (1 / (self.fluidd * R)) * (
            p_gas - self.pamb - p_ext - 4 * self.dviscos * dR / R - 2 * self.sft / R + radiation_loss
        )

        """RT shaping is applied here as a modulation of the collapse dynamics.
        Note: Although the RT envelope formula defines shaping on R(t), this implementation modulates the acceleration term (ddR) directly. This is physically valid and causally embedded, not a post-hoc projection. Embed RT shaping if enabled"""
        if hasattr(self, "apply_rt") and self.apply_rt and self.estimated_t_peak is not None:
            rt_gate = np.exp(-self.lambda_decay * t) * np.exp(-((t - self.estimated_t_peak) / self.sigma_flash)**2)
            ddR *= rt_gate
            
        # Embed NF shaping if enabled
        if hasattr(self, "apply_nf") and self.apply_nf:
            nf_gate = self.ambient_field_modulation(t)
            ddR *= nf_gate
            
        # Embed Temporal Modulation if enabled
        if hasattr(self, "apply_temporal") and self.apply_temporal and self.estimated_t_peak is not None:
            temporal_gate = self.temporal_modulation(t)
            ddR *= temporal_gate

        # Embed QAC modulation if enabled
        if hasattr(self, "apply_qac") and self.apply_qac:
            qac_factor = self.qac_modulation(t, dR)
            ddR *= 1 + qac_factor

        # Embed TRC modulation if enabled
        if hasattr(self, "apply_trc") and self.apply_trc:
            trc_energy = self.trc_modulation(t, R)
            ddR *= 1 + 0.2 * trc_energy  # Apply scaled TRC shaping


        # Embed Energy Optimization if enabled
        if hasattr(self, "apply_energy_opt") and self.apply_energy_opt:
            energy_opt = self.energy_optimization(R, dR, t)
            # Apply modulation only if strength > 0
            if self.energy_modulation_strength > 0:
                print(f"[MODULATION] t={t:.6e}, ddR(before)={ddR:.6e}, energy_opt={energy_opt:.6e}")
                ddR += self.energy_modulation_strength * energy_opt
                print(f"[MODULATION] ddR(after)={ddR:.6e}")
            # Store diagnostic regardless of modulation strength
            self.E_opt_list.append(energy_opt)




        # Compute raw energy (ungated)
        raw_energy = self.fluidd * (4 / 3) * np.pi * R**3 * dR**2 * np.sin(self.theta_ingress) / max(R * self.mu_eff, 1e-12)
        
        # Compute gated energy
        gated_energy = raw_energy
        if t is not None and hasattr(self, "energy_gate_center") and hasattr(self, "energy_gate_width"):
            time_gate = np.exp(-((t - self.energy_gate_center) / self.energy_gate_width)**2)
            gated_energy *= time_gate
        
        # Store both diagnostics
        self.E_opt_raw_list.append(raw_energy)
        self.E_opt_list.append(gated_energy)
        
        # Embed Coherence Modulation if enabled
        if hasattr(self, "coherence_weighting") and self.coherence_weighting > 0 and hasattr(self, "estimated_t_peak"):
            sigma = getattr(self, "sigma_emission", 2e-6)
            t_peak = self.estimated_t_peak
            temporal_gate = np.exp(-((t - t_peak) / sigma)**2)
        
            # Optional angular shaping
            theta = getattr(self, "theta_sample", np.pi / 2)
            angular_term = np.abs(np.sin(theta))
        
            coherence_gate = self.coherence_weighting * angular_term * temporal_gate
            ddR *= 1 + coherence_gate
        
            # Track diagnostic
            if not hasattr(self, "coherence_mod_list"):
                self.coherence_mod_list = []
            self.coherence_mod_list.append(coherence_gate)


        return [dR, ddR]


    def rp_thermal_variant(self, t, y):
        R, dR = y
        kappa = self.dynamic_polytropic_exponent(R, dR, self.adiab)
        h = self.argon
    
        p_gas = (self.pamb + 2 * self.sft / self.bradi - self.psvp) * ((self.bradi**3 - h**3) / (R**3 - h**3)) ** (3 * kappa)
        p_surf = 2 * self.sft / R
        p_liq = p_gas + self.psvp - p_surf
        p_ext = self.pamb + self.acoustic_pressure_waveform(t)
    
        ddR = -3 * dR**2 / (2 * R) + (1 / (self.fluidd * R)) * (
            p_liq - 4 * self.dviscos * dR / R - p_ext
        )
        return [dR, ddR]

 

    def simulate(self, use_thermal_model=False):
        t_span = (0, config.total_time)
        t_eval = np.linspace(*t_span, config.num_points)
        y0 = [self.bradi, 0.0]
    
        # Choose RHS function
        rhs = self.rp_thermal_variant if use_thermal_model else self.collapse_dynamics
    
        # Reset energy list if energy optimization is active
        if hasattr(self, "apply_energy_opt") and self.apply_energy_opt:
            self.E_opt_list = []
        
        self.E_opt_raw_list = []  # Initialize raw energy diagnostic list
        self.E_opt_list = []      # (Re)initialize gated energy diagnostic list if not already done


        # Run simulation
        sol = solve_ivp(
            rhs,
            t_span,
            y0,
            t_eval=t_eval,
            method='Radau',
            rtol=1e-6,
            atol=1e-9
        )
    
        R = sol.y[0]
        dRdt = sol.y[1]
    
        t_array = sol.t
        collapse_index = np.argmin(R)
        t_peak = t_array[collapse_index]
        velocity_array = np.gradient(R, t_array)
        
        # Define Gaussian gate centered on collapse
        sigma_emission = getattr(self, "sigma_emission", 2e-6)  # default width if not set
        gaussian_gate = np.exp(-((t_array - t_peak) / sigma_emission)**2)
        
        # Construct emission profile
        profile = gaussian_gate * np.abs(velocity_array)
        
        # Normalize and window for spectral diagnostics
        profile -= np.mean(profile)
        profile /= np.max(profile) if np.max(profile) != 0 else 1
        
        from scipy.signal.windows import hann
        window = hann(len(profile))
        self.emission_profile = profile * window

    
        # Return energy optimization diagnostic if enabled
        if hasattr(self, "apply_energy_opt") and self.apply_energy_opt and hasattr(self, "E_opt_list") and len(self.E_opt_list) > 0:
            
    
            internal_t = np.linspace(t_span[0], t_span[1], len(self.E_opt_list))
            interp_func = interp1d(internal_t, self.E_opt_list, kind='linear', fill_value="extrapolate")
            E_opt_resampled = interp_func(sol.t)
    
            if getattr(self, "normalize_energy_opt", False):
                E_opt_output = E_opt_resampled / np.max(E_opt_resampled) if np.max(E_opt_resampled) != 0 else E_opt_resampled
            else:
                E_opt_output = E_opt_resampled
    
            return sol.t, R, dRdt, E_opt_output
    
        # Default return if energy optimization is off
        return sol.t, R, dRdt



        
"""
This portion runs different executable code blocks focused on specific use cases.  To run a use case define the run mode associated with the moduluar block, then set the values for the tunables within that block. For example:
    set run_mode to qac_only
    
    Then adjust these parameters as desired
        qac_tunables = {
            "apply_qac": True,
            "phi_acoustic": np.pi / 3,#default np.pi / 2, optimal np.pi / 3
            "eta_ref": 3.002e-3,  # [Pa·s] 1.002e-3 this is from dynamic_viscosity, default 1e-3, optimal 3.002e-3
            "apply_rt": False,
            "apply_nf": False,
            "apply_ingress": False,
            "apply_temporal": False
        }
    
    If you want to combine qac with ingress you would enable ingress but you would also need to supply the ingress tunables within this section.  rt and nf are required to be in all code blocks in order for the script to run, I set them to false to run in x_only mode. As mentioned in  the readme pdf the code is functional but it has not been refactored or optimized. The requirement that rt and nf (ambient field) be present in each tunable, is an artifact of me not updating the code and early stage work..


"""
if __name__ == '__main__':


    run_mode = "angular_only" # Options: "baseline_only", "rt_only", "nf_only", etc. Plug in the code block that you want to execute.

    #Baseline Tunables applies to all runs, they all use the baseline to compare against.
    baseline_tunables = {
        "apply_rt": False,
        "apply_nf": False
    }

    def extract_metrics(t, R, dRdt, simulator):
        collapse_index = np.argmin(R)
        collapse_time = t[collapse_index]
        peak_radius = np.max(R)
        min_radius = np.min(R)
        peak_growth = np.max(np.abs(dRdt))
        flash_threshold = 0.1 * peak_radius
        flash_indices = np.where(R < flash_threshold)[0]
        flash_duration = (t[flash_indices[-1]] - t[flash_indices[0]]) if len(flash_indices) > 1 else 0.0
        energy_proxy = np.trapz(dRdt**2, t)
        envelope_amplitude = (peak_radius - min_radius) * 1e6

        from scipy.signal import welch
        freqs, power = welch(R, fs=1/simulator.dt)
        geometric_mean = np.exp(np.mean(np.log(power + 1e-12)))
        arithmetic_mean = np.mean(power)
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-12)


        if simulator.apply_rt and hasattr(simulator, "estimated_t_peak") and simulator.estimated_t_peak is not None:
            rt_peak_time = simulator.estimated_t_peak * 1e6
            rt_compression_ratio = peak_radius / np.max(R)
        else:
            rt_peak_time = "N/A"
            rt_compression_ratio = "N/A"


        
         # Collapse Asymmetry Index
        collapse_index = np.argmin(R)
        pre_collapse = np.max(np.abs(dRdt[:collapse_index]))
        post_collapse = np.max(np.abs(dRdt[collapse_index:]))
        
        collapse_asymmetry = pre_collapse / post_collapse if post_collapse != 0 else np.inf     
        # Emission Skewness
        emission_skew = skew(simulator.emission_profile) if hasattr(simulator, "emission_profile") else "N/A"
        
        energy_density = 0.5 * simulator.fluidd * dRdt[collapse_index]**2 / (R[collapse_index]**3)

        
        # Spectral Entropy
        spectrum = np.abs(np.fft.rfft(simulator.emission_profile))
        norm_spectrum = spectrum / np.sum(spectrum) if np.sum(spectrum) != 0 else spectrum
        spectral_entropy = entropy(norm_spectrum + 1e-12, base=2)


        
        # Flash Rise Time
        peak = np.max(simulator.emission_profile)
        above_10 = np.where(simulator.emission_profile >= 0.1 * peak)[0]
        above_90 = np.where(simulator.emission_profile >= 0.9 * peak)[0]
        if len(above_10) > 0 and len(above_90) > 0:
            rise_time_ns = (above_90[0] - above_10[0]) * (t[1] - t[0]) * 1e9
        else:
            rise_time_ns = 0.0



        return {
            "Peak Radius (µm)": peak_radius * 1e6,
            "Collapse Radius (µm)": min_radius * 1e6,
            "Collapse Time (µs)": collapse_time * 1e6,
            "Flash Duration (ns)": flash_duration * 1e9,
            "Peak Growth Rate (m/s)": peak_growth,
            "Energy Proxy": energy_proxy,
            "Envelope Amplitude (µm)": envelope_amplitude,
            "Spectral Flatness": spectral_flatness,
            "RT Peak Time (µs)": rt_peak_time,
            "RT Compression Ratio": rt_compression_ratio,
            "Collapse Asymmetry Index": collapse_asymmetry,
            "Emission Skewness": emission_skew,
            "Energy Density at Collapse": energy_density,
            "Spectral Entropy": spectral_entropy,
            "Flash Rise Time (ns)": rise_time_ns
        }

    # Time setup
    t_span = (0, 0.4e-6 * 100)
    t_eval = np.linspace(t_span[0], t_span[1], 10000)


    if run_mode == "baseline_only":
        sim = Framework.from_config()
        sim.configure_tunables(baseline_tunables)
        t, R, dRdt = sim.simulate()
        plot_rt_radius_growth(t, R, dRdt, apply_rt=True, estimated_t_peak=sim.estimated_t_peak)
        metrics = extract_metrics(t, R, dRdt, sim)
        print("Baseline-only run complete.")

    elif run_mode == "rt_only":
        baseline_sim = Framework.from_config()
        baseline_sim.configure_tunables(baseline_tunables)
        t_base, R_base, dRdt_base = baseline_sim.simulate()
        t_peak_index = np.argmin(R_base)
        t_peak = t_base[t_peak_index]
        # RT only
        rt_tunables = {
            "apply_rt": True,
            "lambda_decay": 100000.0,
            "sigma_flash": 50e-6,
            "apply_nf": False
        }
        rt_sim = Framework.from_config()
        rt_sim.configure_tunables(rt_tunables)
        rt_sim.estimated_t_peak = 20.207e-6  # Collapse time in seconds  # Inject into RP equation #Baseline run t_peak found to be 20.207e-6 used for RT sim 
        t_rt, R_rt, dRdt_rt = rt_sim.simulate()

        metrics_base = extract_metrics(t_base, R_base, dRdt_base, baseline_sim)
        metrics_rt = extract_metrics(t_rt, R_rt, dRdt_rt, rt_sim)

        plot_rt_only_comparison(t_base, R_base, t_rt, R_rt, rt_sim.estimated_t_peak)



        
        write_comparison_csv(
            filename="simulation_metrics_rt.csv",
            metrics_base=metrics_base,
            metrics_modulated=metrics_rt,
            mod_label="RT Enabled"
        )


        print("RT-only comparison complete.")

    elif run_mode == "nf_only": #ambient field
        baseline_sim = Framework.from_config()
        baseline_sim.configure_tunables(baseline_tunables)
        t_base, R_base, dRdt_base = baseline_sim.simulate()
    
        # NF only
        nf_tunables = {
            "apply_rt": False,
            "apply_nf": True,
            "alpha_field": 8, #default 0.2, opt 1.4, under 0.05, over 8
            "f_ambient": 500, #Default 500, opt 1000, under 5000, over 500
            "phi_nu": np.pi, #defalt/opt np.pi / 2, under 0.0, over np.pi
            "kappa_nu": 0.01 #default/opt 0.05, under 0.02, over 0.01
        }

    
        nf_sim = Framework.from_config()
        nf_sim.configure_tunables(nf_tunables)
        t_nf, R_nf, dRdt_nf = nf_sim.simulate()
    
        metrics_base = extract_metrics(t_base, R_base, dRdt_base, baseline_sim)
        metrics_nf = extract_metrics(t_nf, R_nf, dRdt_nf, nf_sim)
    
        plot_nf_only_comparison(t_base, R_base, t_nf, R_nf)

    

        
        write_comparison_csv(
            filename="simulation_metrics_nf.csv",
            metrics_base=metrics_base,
            metrics_modulated=metrics_nf,
            mod_label="NF Enabled"
        )


        print("NF-only comparison complete.")


    elif run_mode == "rt_nf_combined":
        baseline_sim = Framework.from_config()
        baseline_sim.configure_tunables(baseline_tunables)
        t_base, R_base, dRdt_base = baseline_sim.simulate()
        t_peak_index = np.argmin(R_base)
        t_peak = t_base[t_peak_index]

        rt_nf_tunables = {
            "apply_rt": True,
            "lambda_decay": 16000.0,
            "sigma_flash": 30e-6,
            "apply_nf": True,
            "alpha_field": 1,
            "f_ambient": 1000000.0,
            "phi_nu": 20.207e-6, #  "phi_nu": np.pi / 2,
            "kappa_nu": 0.02
        }
        
        rt_nf_sim = Framework.from_config()
        rt_nf_sim.configure_tunables(rt_nf_tunables)
        rt_nf_sim.estimated_t_peak = 20.207e-6  # Collapse time in seconds  # Inject into RP equation #Baseline run t_peak found to be 20.207e-6 used for RT sim 
        t_combined, R_combined, dRdt_combined = rt_nf_sim.simulate()



        metrics_base = extract_metrics(t_base, R_base, dRdt_base, baseline_sim)
        metrics_combined = extract_metrics(t_combined, R_combined, dRdt_combined, rt_nf_sim)

        plot_nf_comparison(t_base, R_base, t_combined, R_combined)



        
        write_comparison_csv(
            filename="simulation_metrics_rt_nf.csv",
            metrics_base=metrics_base,
            metrics_modulated=metrics_combined,
            mod_label="RT + NF Enabled"
        )


    elif run_mode == "ingress_only":
        baseline_sim = Framework.from_config()
        baseline_sim.configure_tunables(baseline_tunables)
        t_base, R_base, dRdt_base = baseline_sim.simulate()


        ingress_tunables = {
            "apply_rt": False,
            "apply_nf": False,
            "apply_ingress": True,
            "beta_ingress": 1000, #default 1000.0
            "omega_ingress": 500, #default 500.0
            "phi_ingress": 0.0, #default 0.0
            "kappa_ingress": 0.1 #default 0.1 
        }

        ingress_sim = Framework.from_config()
        ingress_sim.configure_tunables(ingress_tunables)
        t_ingress, R_ingress, dRdt_ingress = ingress_sim.simulate()

        metrics_base = extract_metrics(t_base, R_base, dRdt_base, baseline_sim)
        metrics_ingress = extract_metrics(t_ingress, R_ingress, dRdt_ingress, ingress_sim)

        plot_ingress_comparison(t_base, R_base, t_ingress, R_ingress)


        
        write_comparison_csv(
            filename="simulation_metrics_ingress.csv",
            metrics_base=metrics_base,
            metrics_modulated=metrics_ingress,
            mod_label="Ingress Enabled"
        )


    elif run_mode == "nf_ingress_combined":
        baseline_sim = Framework.from_config()
        baseline_sim.configure_tunables(baseline_tunables)
        t_base, R_base, dRdt_base = baseline_sim.simulate()
        
        nf_ingress_tunables = {
            "apply_rt": False,
            "apply_nf": True,
            "alpha_field": 1,#optimal 1
            "f_ambient": 100000.0, #optimal 100000.0
            "phi_nu": np.pi / 1.5, #optimal np.pi / 1.5
            "kappa_nu": 0.02, #optimal 0.02
            "apply_ingress": True,
            "beta_ingress": 500, #optimal 500
            "omega_ingress": 100, #optimal 100
            "phi_ingress": np.pi / 10, #optimal np.pi / 10,
            "kappa_ingress": 0.1 #optimal 0.1
        }

        nf_ingress_sim = Framework.from_config()
        nf_ingress_sim.configure_tunables(nf_ingress_tunables)
        t_mod, R_mod, dRdt_mod = nf_ingress_sim.simulate()
    
        metrics_base = extract_metrics(t_base, R_base, dRdt_base, baseline_sim)
        metrics_mod = extract_metrics(t_mod, R_mod, dRdt_mod, nf_ingress_sim)
    
        plot_nf_ingress_comparison(t_base, R_base, t_mod, R_mod)

        
        write_comparison_csv(
            filename="simulation_metrics_nf_ingress.csv",
            metrics_base=metrics_base,
            metrics_modulated=metrics_mod,
            mod_label="NF + Ingress Enabled"
        )

    
        print("NF + Ingress combined run complete.")
  
    elif run_mode == "temporal_only":
        # Baseline simulation to extract t_peak
        baseline_sim = Framework.from_config()
        baseline_sim.configure_tunables(baseline_tunables)
        t_base, R_base, dRdt_base = baseline_sim.simulate()
        temporal_tunables = {
            "apply_rt": False,
            "apply_nf": False,
            "apply_ingress": False,
            "apply_temporal": True,
            "sigma_emission": 70e-6 # 1e-6 collapse the signal, 9e-6 about half of baseline, 70e-6 match baseline 
        }
        # Estimate t_peak from baseline collapse
        t_peak_index = np.argmin(R_base)
        t_peak = t_base[t_peak_index]
    
        # Optionally override with static value
        #t_peak = 20.207e-6  # Uncomment to use a fixed collapse time
    
        # Temporal-modulated simulation
        temporal_sim = Framework.from_config()
        temporal_sim.configure_tunables(temporal_tunables)
        temporal_sim.estimated_t_peak = t_peak
        t_temp, R_temp, dRdt_temp = temporal_sim.simulate()
    
        # Metrics and visualization
        metrics_base = extract_metrics(t_base, R_base, dRdt_base, baseline_sim)
        metrics_temp = extract_metrics(t_temp, R_temp, dRdt_temp, temporal_sim)
    
        plot_temporal_comparison(t_base, R_base, t_temp, R_temp)

    
        
        write_comparison_csv(
            filename="simulation_metrics_temporal.csv",
            metrics_base=metrics_base,
            metrics_modulated=metrics_temp,
            mod_label="Temporal Enabled"
        )

    
        print("Temporal-only comparison complete.")
        print("Simulation finished and plot displayed.")


    elif run_mode == "qac_only":
        baseline_sim = Framework.from_config()
        baseline_sim.configure_tunables(baseline_tunables)
        t_base, R_base, dRdt_base = baseline_sim.simulate()
    
        qac_tunables = {
            "apply_qac": True,
            "phi_acoustic": np.pi / 3,#default np.pi / 2, optimal np.pi / 3
            "eta_ref": 3.002e-3,  # [Pa·s] 1.002e-3 this is from dynamic_viscosity, default 1e-3, optimal 3.002e-3
            "apply_rt": False,
            "apply_nf": False,
            "apply_ingress": False,
            "apply_temporal": False
        }
    
    
    
        qac_sim = Framework.from_config()
        qac_sim.configure_tunables(qac_tunables)
        t_qac, R_qac, dRdt_qac = qac_sim.simulate()
    
        metrics_base = extract_metrics(t_base, R_base, dRdt_base, baseline_sim)
        metrics_qac = extract_metrics(t_qac, R_qac, dRdt_qac, qac_sim)
    
        plot_qac_comparison(t_base, R_base, t_qac, R_qac)


    
    

    

        write_comparison_csv(
            filename="simulation_metrics_qac.csv",
            metrics_base=metrics_base,
            metrics_modulated=metrics_qac,
            mod_label="QAC Enabled"
        )
        print("QAC-only comparison complete.")


    elif run_mode == "trc_only":
       # Baseline simulation
       baseline_sim = Framework.from_config()
       baseline_sim.configure_tunables(baseline_tunables)
       t_base, R_base, dRdt_base = baseline_sim.simulate()
   
    
       # TRC-modulated simulation, A's are amplitude Energy contribution of each harmonic, omega is frequency -Oscillation rate of each harmonic and phi is phase -Temporal alignment of each harmonic

       trc_tunables = {
           "apply_trc": True,
           "A1": 578000,     
           "A2": 478000,     
           "A3": 478000,     
           "omega1": 0.9e5, #1e5 works
           "omega2": 1.9e5, #2e5 works
           "omega3": 2.9e5, #3e5 works
           "phi1": np.pi / 3.3,
           "phi2": np.pi / 3.3,
           "phi3": np.pi / 3.3,
           "apply_rt": False,
           "apply_nf": False,
           "apply_ingress": False,
           "apply_temporal": False
        }




   
       trc_sim = Framework.from_config()
       trc_sim.configure_tunables(trc_tunables)
       t_trc, R_trc, dRdt_trc = trc_sim.simulate()
   
       # Extract metrics
       metrics_base = extract_metrics(t_base, R_base, dRdt_base, baseline_sim)
       metrics_trc = extract_metrics(t_trc, R_trc, dRdt_trc, trc_sim)
   
       # Plot radius and growth rate comparison
       plot_trc_comparison(t_base, R_base, t_trc, R_trc)
       plot_trc_energy(t_trc, R_trc, trc_sim.trc_modulation)



   
       # Export metrics to CSV               
       write_comparison_csv(
            filename="simulation_metrics_trc.csv",
            metrics_base=metrics_base,
            metrics_modulated=metrics_trc,
            mod_label="TRC Enabled"
        )

   
       print("TRC-only comparison complete.")

    elif run_mode == "energy_opt_only":
        # Baseline simulation (no modulation)
        baseline_sim = Framework.from_config()
        baseline_sim.configure_tunables(baseline_tunables)
        t_base, R_base, dRdt_base = baseline_sim.simulate(use_thermal_model=False)  # Explicitly use collapse_dynamics
        # Compute collapse time from baseline
        collapse_index = np.argmin(R_base)
        collapse_time = t_base[collapse_index]
        # Energy Optimization–modulated simulation
        energy_tunables = {
            "apply_energy_opt": True,
            "theta_ingress": np.pi / 21, #Controls how much of the collapse energy is considered "directionally extractable" 0.05, 100
            "mu_eff": 0.1, #Appears in the denominator of the energy optimization formula, reducing yield as permeability increases 0.0, 1.0
            "energy_modulation_strength": 0, #1000000000 shows an effect when using collapse_time and 5e-6 window; 
            

            "energy_gate_center": collapse_time,  # 18 µs 2.88e-5 and then 2.48e-5 collapse_time, 24.357e-6
            "energy_gate_width": 2e-8,      # 2 µs most runs set to 2e-8 and this is required with 24.357e-6
            "apply_rt": False,
            "apply_nf": False,
            "apply_ingress": False,
            "apply_temporal": False,
            "apply_trc": False,
            "apply_qac": False
        
        }

        energy_sim = Framework.from_config()
        energy_sim.configure_tunables(energy_tunables)
       
        # Inject collapse time into simulator instance (optional, if used elsewhere)
        energy_sim.estimated_t_peak = collapse_time
       
        t_energy, R_energy, dRdt_energy, E_opt = energy_sim.simulate(use_thermal_model=False)  # Explicitly use collapse_dynamics
        max_index = np.argmax(E_opt)
        print(f"[DIAGNOSTIC] Max E_opt at t = {t_energy[max_index] * 1e6:.2f} µs")

        # Extract metrics
        metrics_base = extract_metrics(t_base, R_base, dRdt_base, baseline_sim)
        metrics_energy = extract_metrics(t_energy, R_energy, dRdt_energy, energy_sim)
    

        # Plot comparison
        plot_diagnostics(t_energy, R_energy, dRdt_energy, E_opt, label="Energy Optimization")
        plot_energy_opt_comparison(t_base, R_base, t_energy, R_energy, E_opt)



    
        # Save metrics
        write_comparison_csv(
            filename="simulation_metrics_energy_opt.csv",
            metrics_base=metrics_base,
            metrics_modulated=metrics_energy,
            mod_label="Energy Optimization Enabled"
        )


        print("Energy_optimization-only comparison complete.")



    elif run_mode == "angular_only":
        baseline_sim = Framework.from_config()
        baseline_sim.configure_tunables(baseline_tunables)
        t_base, R_base, dRdt_base = baseline_sim.simulate()
    
        angular_tunables = {
            "apply_angular_pressure": True,
            "alpha_theta": 1.0,
            "omega_theta": 3.0,
            "phi_theta": np.pi / 3.5, #-np.pi / 2
            "kappa_theta": 0.001,
            "theta_sample": np.pi / 3, #np.pi / 7
            "apply_rt": False,
            "apply_nf": False,
            "apply_ingress": False,
            "apply_temporal": False
        }
    
        angular_sim = Framework.from_config()
        angular_sim.configure_tunables(angular_tunables)
        t_ang, R_ang, dRdt_ang = angular_sim.simulate()
    
        metrics_base = extract_metrics(t_base, R_base, dRdt_base, baseline_sim)
        metrics_ang = extract_metrics(t_ang, R_ang, dRdt_ang, angular_sim)
    
        plot_angular_comparison(
            t_base, R_base,
            t_ang, R_ang,
            angular_mod_list=getattr(angular_sim, "angular_mod_list", None)
        )

    
        write_comparison_csv(
            filename="simulation_metrics_angular.csv",
            metrics_base=metrics_base,
            metrics_modulated=metrics_ang,
            mod_label="Angular Pressure Enabled"
        )


    
        print("Angular-only comparison complete.")

    elif run_mode == "rt_nf_tm_qac_combined":
        baseline_sim = Framework.from_config()
        baseline_sim.configure_tunables(baseline_tunables)
        t_base, R_base, dRdt_base = baseline_sim.simulate()
    
        # Compute collapse time from baseline
        collapse_index = np.argmin(R_base)
        collapse_time = t_base[collapse_index]
    
        # Estimate t_peak from baseline collapse
        t_peak_index = np.argmin(R_base)
        t_peak = t_base[t_peak_index]
    
        combined_tunables = {
            "apply_rt": True,
            "lambda_decay": 16000.0,
            "sigma_flash": 30e-6,
    
            "apply_nf": True,
            "alpha_field": 1.4, #1
            "f_ambient": 1000.0, #1000000.0,
            "phi_nu": np.pi / 3, #20.207e-6, np.pi / 2
            "kappa_nu": 0.05, #0.02
    
            "apply_temporal": True,
            "sigma_emission": 90e-5,
    
            "apply_qac": True,
            "phi_acoustic": np.pi / 3.9, #np.pi / 3
            "eta_ref": 1.3002e-3
        }
    
    

        combined_sim = Framework.from_config()
        combined_sim.configure_tunables(combined_tunables)
        combined_sim.estimated_t_peak = t_peak
    
        t_combined, R_combined, dRdt_combined = combined_sim.simulate()
    
        metrics_base = extract_metrics(t_base, R_base, dRdt_base, baseline_sim)
        metrics_combined = extract_metrics(t_combined, R_combined, dRdt_combined, combined_sim)
    
        rt_nf_tm_qac_combined(t_base, R_base, t_combined, R_combined)

    
        write_comparison_csv(
            filename="simulation_metrics_rt_nf_tm_qac.csv",
            metrics_base=metrics_base,
            metrics_modulated=metrics_combined,
            mod_label="RT + NF + TM + QAC Enabled"
        )
    
        print("RT + NF + TM + QAC combined run complete.")


    elif run_mode == "tm_trc_qrc_combined":
        baseline_sim = Framework.from_config()
        baseline_sim.configure_tunables(baseline_tunables)
        t_base, R_base, dRdt_base = baseline_sim.simulate()
    
        # Estimate t_peak from baseline collapse
        t_peak_index = np.argmin(R_base)
        t_peak = t_base[t_peak_index]
    
        combined_tunables = {
            "apply_rt": False,
            "apply_nf": False,
            "apply_temporal": True,
            "sigma_emission": 90e-5,
    
            "apply_trc": True,
            "A1": 578000,
            "A2": 478000,
            "A3": 478000,
            "omega1": 0.9e5,
            "omega2": 1.9e5,
            "omega3": 2.9e5,
            "phi1": np.pi / 3.3,
            "phi2": np.pi / 3.3,
            "phi3": np.pi / 3.3,
    
            "apply_qac": True,
            "phi_acoustic": np.pi / 3.9,
            "eta_ref": 1.3002e-3
        }
    
        combined_sim = Framework.from_config()
        combined_sim.configure_tunables(combined_tunables)
        combined_sim.estimated_t_peak = t_peak
    
        t_mod, R_mod, dRdt_mod = combined_sim.simulate()
    
        metrics_base = extract_metrics(t_base, R_base, dRdt_base, baseline_sim)
        metrics_mod = extract_metrics(t_mod, R_mod, dRdt_mod, combined_sim)
    
        tm_trc_qrc_combined(t_base, R_base, t_mod, R_mod)


        write_comparison_csv(
            filename="simulation_metrics_tm_trc_qrc.csv",
            metrics_base=metrics_base,
            metrics_modulated=metrics_mod,
            mod_label="TM + TRC + QRC Enabled"
        )
    
        print("TM + TRC + QRC combined run complete.")

    #Test combining energy optimized, rt and qac with standard tpeak
    elif run_mode == "eopt_rt_qac":
        baseline_sim = Framework.from_config()
        baseline_sim.configure_tunables(baseline_tunables)
        t_base, R_base, dRdt_base = baseline_sim.simulate()
    
        # Estimate t_peak from baseline collapse
        t_peak_index = np.argmin(R_base)
        t_peak = t_base[t_peak_index]

        combined_tunables = {
            "apply_rt": True,
            "lambda_decay": 16000.0,
            "sigma_flash": 30e-6,
                
            "apply_qac": True,
            "phi_acoustic": np.pi / 3.9, #np.pi / 3.8
            "eta_ref": 1.3002e-3, #1.0e-3
    
            "apply_energy_opt": True,
            "energy_modulation_strength": 1.37e11, #1.37e11
            "energy_gate_center": t_peak, #t_peak
            "energy_gate_width": 2e-8,
            "theta_ingress": np.pi / 4,
            "mu_eff": 0.001
        }
        
    
        combined_sim = Framework.from_config()
        combined_sim.configure_tunables(combined_tunables)
        combined_sim.estimated_t_peak = t_peak
        
        # Conditional unpacking based on Eopt toggle
        if combined_tunables.get("apply_energy_opt", False):
            t_mod, R_mod, dRdt_mod, energy_opt_mod = combined_sim.simulate()
        else:
            t_mod, R_mod, dRdt_mod = combined_sim.simulate()
            energy_opt_mod = None  # Optional placeholder
       
        metrics_base = extract_metrics(t_base, R_base, dRdt_base, baseline_sim)
        metrics_mod = extract_metrics(t_mod, R_mod, dRdt_mod, combined_sim)
            
        t_gate_center = t_peak
        eopt_rt_qac_combined(t_base, R_base, t_mod, R_mod, t_peak, t_gate_center )

    
        write_comparison_csv(
            filename="simulation_metrics_eopt_rt_qac.csv",
            metrics_base=metrics_base,
            metrics_modulated=metrics_mod,
            mod_label="Eopt + RT + QAC Enabled"
        )
        print(f"t_peak = {t_peak:.6f} µs")
        print("Eopt + RT + QAC combined run complete.")
        
    #Test combining energy optimized, rt and qac but include an offset for tpeak    
    elif run_mode == "eopt_rt_qac_tpeak_offset":
        baseline_sim = Framework.from_config()
        baseline_sim.configure_tunables(baseline_tunables)
        t_base, R_base, dRdt_base = baseline_sim.simulate()
    
        # Estimate t_peak from baseline collapse
        t_peak_index = np.argmin(R_base)
        t_peak = t_base[t_peak_index]
        #To test offset use t_peak1
        offset = 5e-7  # adds 0.5 µs = 5e-7, while 5e-6 is 5 µs
        t_peak1 = t_base[t_peak_index] + offset


        combined_tunables = {
            "apply_rt": True,
            "lambda_decay": 16000.0,
            "sigma_flash": 30e-6,
                
            "apply_qac": True,
            "phi_acoustic": np.pi / 3.9, #np.pi / 3.8
            "eta_ref": 1.3002e-3, #1.0e-3
    
            "apply_energy_opt": True,
            "energy_modulation_strength": 1.37e11, #1.37e11
            "energy_gate_center": t_peak1, #t_peak
            "energy_gate_width": 2e-8,
            "theta_ingress": np.pi / 4,
            "mu_eff": 0.001
        }
        
    
        combined_sim = Framework.from_config()
        combined_sim.configure_tunables(combined_tunables)
        combined_sim.estimated_t_peak = t_peak1
        
        # Conditional unpacking based on Eopt toggle
        if combined_tunables.get("apply_energy_opt", False):
            t_mod, R_mod, dRdt_mod, energy_opt_mod = combined_sim.simulate()
        else:
            t_mod, R_mod, dRdt_mod = combined_sim.simulate()
            energy_opt_mod = None  # Optional placeholder
       
        metrics_base = extract_metrics(t_base, R_base, dRdt_base, baseline_sim)
        metrics_mod = extract_metrics(t_mod, R_mod, dRdt_mod, combined_sim)
        
       # Normalize radius and growth rate to their respective max values
        R_base_norm = R_base / np.max(R_base)
        R_mod_norm = R_mod / np.max(R_mod)
        
        dRdt_base_norm = dRdt_base / np.max(np.abs(dRdt_base))
        dRdt_mod_norm = dRdt_mod / np.max(np.abs(dRdt_mod))

        t_gate_center = t_peak1
        eopt_rt_qac_combined(t_base, R_base_norm, t_mod, R_mod_norm, t_peak, t_gate_center )
    
        write_comparison_csv(
            filename="simulation_metrics_eopt_rt_qac.csv",
            metrics_base=metrics_base,
            metrics_modulated=metrics_mod,
            mod_label="Eopt + RT + QAC Enabled"
        )
        print(f"t_peak = {t_peak:.6f} µs")
        print(f"t_peak1 = {t_peak1:.6f} µs")
        print("Eopt + RT + QAC combined run complete.")
        
        
    elif run_mode == "resonance_yield_test":
        # Baseline RP simulation
        baseline_sim = Framework.from_config()
        baseline_sim.configure_tunables(baseline_tunables)
        t_base, R_base, dRdt_base = baseline_sim.simulate()
    
        # Estimate t_peak from baseline collapse
        t_peak_index = np.argmin(R_base)
        t_peak = t_base[t_peak_index]
    
        # Configure simulation for resonance yield test (no other tunables)
        resonance_sim = Framework.from_config()
        resonance_sim.configure_tunables({
            "apply_rt": False,
            "apply_nf": False,
            "apply_temporal": False,
            "apply_qac": False,
            "coherence_weighting": 2.0,  # 0.0 disables, 1.0 full influence, >1 amplifies, <1 softens, 2.0 max
            "omega3": 14e5,              # TRC-like harmonic frequency
            "lambda_decay": 6e4,        # Damping coefficient
            "sigma_emission": 1     # Collapse phase spread
        })
        resonance_sim.estimated_t_peak = t_peak
    
        # Run RP simulation (unmodulated)
        t_resonance, R_resonance, dRdt_resonance = resonance_sim.simulate()
    
        # Apply Resonance Yield Function
        R_Yield = resonance_sim.compute_resonance_yield(t_resonance, R_resonance)

    
        # Extract metrics
        metrics_base = extract_metrics(t_base, R_base, dRdt_base, baseline_sim)
        metrics_resonance = extract_metrics(t_resonance, R_Yield, np.gradient(R_Yield, t_resonance), resonance_sim)
    
        # Plot comparison

        resonance_yield_plot(t_base, R_base, t_resonance, R_Yield)
        resonance_yield_diagnostics(t_base, R_base, t_resonance, R_Yield)

    # Define coherence_index for surface plot
        def coherence_index(t, theta):
            coherence_strength = getattr(resonance_sim, "coherence_weighting", 0.0)
            sigma_emission = getattr(resonance_sim, "sigma_emission", 2e-6)
            t_peak = resonance_sim.estimated_t_peak
    
            if coherence_strength == 0.0:
                return 1.0
    
            angular_term = np.abs(np.sin(theta))
            temporal_term = np.exp(-((t - t_peak) / sigma_emission)**2)
            val = coherence_strength * angular_term * temporal_term
    
            return 1.0 if np.isnan(val) or np.isinf(val) else val
    
        # Plot coherence surface (fix shape mismatch by transposing C inside plots.py)
        plot_coherence_surface(t_resonance, np.linspace(0, np.pi, 100), coherence_index)



    
        # Write metrics to CSV
        write_comparison_csv(
            filename="simulation_metrics_resonance_yield.csv",
            metrics_base=metrics_base,
            metrics_modulated=metrics_resonance,
            mod_label="Resonance Yield Function Enabled"
        )
    
        print("Resonance Yield Function test run complete.")
        
    elif run_mode == "ingress_calibration":
       # Baseline simulation
       baseline_sim = Framework.from_config()
       baseline_sim.configure_tunables(baseline_tunables)
       t_base, R_base, dRdt_base = baseline_sim.simulate()
       metrics_base = extract_metrics(t_base, R_base, dRdt_base, baseline_sim)
   
    # Compute directional speeds for baseline - additional sanity check step
       max_pos_base = np.max(dRdt_base)  # Expansion speed
       max_neg_base = -np.min(dRdt_base)  # Collapse speed magnitude
       print(f"[Baseline] Expansion Speed: {max_pos_base:.4f} m/s")
       print(f"[Baseline] Collapse Speed: {max_neg_base:.4f} m/s")
   
    
       # Calibration sweep — symbolic ingress configurations mapped to physical analogs
       calibration_sweep = [
           {
               "label": "No Ingress (Baseline)",
               "params": {
                   "apply_rt": False,
                   "apply_nf": False,
                   "apply_ingress": False,
                   "beta_ingress": 0.0,
                   "omega_ingress": 0.0,
                   "phi_ingress": 0.0,
                   "kappa_ingress": 0.0
               },
               "physical_model_match": "None"
           },
           {
               "label": "Solid-Body Rotation Emulation",
               "params": {
                   "apply_ingress": True,
                   "apply_rt": False,
                   "apply_nf": False,
                   "beta_ingress": 1000,
                   "omega_ingress": 500,
                   "phi_ingress": 0.0,
                   "kappa_ingress": 0.05
               },
               "physical_model_match": "Centrifugal pressure (Omega^2 * R^2)"
           },
           {
               "label": "Potential Vortex Emulation",
               "params": {
                   "apply_ingress": True,
                   "apply_rt": False,
                   "apply_nf": False,
                   "beta_ingress": 1500,
                   "omega_ingress": 100,
                   "phi_ingress": np.pi / 10,
                   "kappa_ingress": 0.01
               },
               "physical_model_match": "Vortex pressure (Gamma^2 / R^2)"
           },

       ]
   
       # Run calibration sweep
       for i, sweep_config in enumerate(calibration_sweep):
           sim = Framework.from_config()
           sim.configure_tunables(sweep_config["params"])
           t_mod, R_mod, dRdt_mod = sim.simulate()
   
           metrics_mod = extract_metrics(t_mod, R_mod, dRdt_mod, sim)
           
           # Compute directional speeds for modulated run
           max_pos = np.max(dRdt_mod)
           max_neg = -np.min(dRdt_mod)
           print(f"[{sweep_config['label']}] Expansion Speed: {max_pos:.4f} m/s")
           print(f"[{sweep_config['label']}] Collapse Speed: {max_neg:.4f} m/s")
   
           # Save results with ASCII-safe label
           safe_label = f"{sweep_config['label']} | Match: {sweep_config['physical_model_match']}"
           write_comparison_csv(
               filename=f"calibration_ingress_{i}.csv",
               metrics_base=metrics_base,
               metrics_modulated=metrics_mod,
               mod_label=safe_label
           )
   
           # Optional: plot comparison
           plot_ingress_comparison(t_base, R_base, t_mod, R_mod)
           print(f"Completed: {safe_label}")
   
   
   
    elif run_mode == "vortex_validation":
      # Load real data from experiment (e.g. collapse radius, time, velocity)
      real_metrics = {
          "collapse_radius": 0.29,  # µm
          "collapse_time": 18.0,    # µs
          "peak_velocity": 340.0,   # m/s
          "asymmetry_index": 2.2,
          "flash_rise_time": 45.0   # ns
      }
  
      # Baseline simulation
      baseline_sim = Framework.from_config()
      baseline_sim.configure_tunables(baseline_tunables)
      t_base, R_base, dRdt_base = baseline_sim.simulate()
      metrics_base = extract_metrics(t_base, R_base, dRdt_base, baseline_sim)
  
      # Sweep over symbolic vortex parameters
      gamma_sweep = [500, 1000, 1500, 2000]
      for gamma in gamma_sweep:
          sweep_config = {
              "label": f"Vortex Emulation Gamma={gamma}",
              "params": {
                  "apply_ingress": True,
                  "apply_rt": False,
                  "apply_nf": False,
                  "beta_ingress": gamma,
                  "omega_ingress": 100,
                  "phi_ingress": np.pi / 10,
                  "kappa_ingress": 0.01
              },
              "physical_model_match": f"Vortex pressure (Gamma={gamma}^2 / R^2)"
          }
  
          sim = Framework.from_config()
          sim.configure_tunables(sweep_config["params"])
          t_mod, R_mod, dRdt_mod = sim.simulate()
          metrics_mod = extract_metrics(t_mod, R_mod, dRdt_mod, sim)
  
          # Compute directional speeds
          max_pos = np.max(dRdt_mod)
          max_neg = -np.min(dRdt_mod)
  
          # Compare to real data
          collapse_radius_error = abs(metrics_mod["Collapse Radius (µm)"] - real_metrics["collapse_radius"])
          collapse_time_error = abs(metrics_mod["Collapse Time (µs)"] - real_metrics["collapse_time"])
          velocity_error = abs(max_neg - real_metrics["peak_velocity"])
  
          print(f"[Gamma={gamma}] Collapse Radius Error: {collapse_radius_error:.4f} µm")
          print(f"[Gamma={gamma}] Collapse Time Error: {collapse_time_error:.4f} µs")
          print(f"[Gamma={gamma}] Collapse Speed Error: {velocity_error:.4f} m/s")
  
          # Save results
          safe_label = f"{sweep_config['label']} | Match: {sweep_config['physical_model_match']}"
          write_comparison_csv(
              filename=f"vortex_validation_gamma_{gamma}.csv",
              metrics_base=metrics_base,
              metrics_modulated=metrics_mod,
              mod_label=safe_label
          )
  
          # Plot comparison
          plot_ingress_comparison(t_base, R_base, t_mod, R_mod)
          print(f"Completed: {safe_label}")
  
    elif run_mode == "vortex_convergence_test":
         # Target Gamma for convergence test
         gamma_target = 1500
        
         # Simulation configuration
         sweep_config = {
             "label": f"Vortex Emulation Gamma={gamma_target}",
             "params": {
                 "apply_ingress": True,
                 "apply_rt": False,
                 "apply_nf": False,
                 "beta_ingress": gamma_target,
                 "omega_ingress": 100,
                 "phi_ingress": np.pi / 10,
                 "kappa_ingress": 0.01
             },
             "physical_model_match": f"Vortex pressure (Gamma={gamma_target}^2 / R^2)"
         }
        
         # Original timestep simulation
         sim_original = Framework.from_config()
         sim_original.configure_tunables(sweep_config["params"])
         t_orig, R_orig, dRdt_orig = sim_original.simulate()
         metrics_orig = extract_metrics(t_orig, R_orig, dRdt_orig, sim_original)
         collapse_speed_orig = -np.min(dRdt_orig)
        
         # Halved timestep simulation
         sim_refined = Framework.from_config()
         sim_refined.configure_tunables(sweep_config["params"])
         sim_refined.dt = sim_original.dt / 2  # Halve the timestep
         
         print(f"Original dt: {sim_original.dt:.6e}, Refined dt: {sim_refined.dt:.6e}")

         
         t_ref, R_ref, dRdt_ref = sim_refined.simulate()
         metrics_ref = extract_metrics(t_ref, R_ref, dRdt_ref, sim_refined)
         collapse_speed_ref = -np.min(dRdt_ref)
        
         # Compare key metrics
         rmin_delta = abs(metrics_orig["Collapse Radius (µm)"] - metrics_ref["Collapse Radius (µm)"])
         time_delta = abs(metrics_orig["Collapse Time (µs)"] - metrics_ref["Collapse Time (µs)"])
         speed_delta = abs(collapse_speed_orig - collapse_speed_ref)
        
         print(f"[Convergence Test Gamma={gamma_target}] ΔRmin: {rmin_delta:.4f} µm")
         print(f"[Convergence Test Gamma={gamma_target}] ΔTime: {time_delta:.4f} µs")
         print(f"[Convergence Test Gamma={gamma_target}] ΔSpeed: {speed_delta:.4f} m/s")
        
         # Optional: Save comparison
         write_comparison_csv(
             filename=f"vortex_convergence_gamma_{gamma_target}.csv",
             metrics_base=metrics_orig,
             metrics_modulated=metrics_ref,
             mod_label=f"Convergence Test | Gamma={gamma_target}"
         )
        
         # Optional: Plot overlay
         plot_ingress_comparison(t_orig, R_orig, t_ref, R_ref)
         print(f"Completed: Convergence Test for Gamma={gamma_target}")
        
   