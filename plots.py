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

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def plot_rt_radius_growth(t, R, dRdt, apply_rt=False, estimated_t_peak=None):


    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Plot radius
    ax[0].plot(t * 1e6, R * 1e6, label='RP Radius', color='blue')

    # Annotate t_peak if RT is enabled and estimated_t_peak is provided
    if apply_rt and estimated_t_peak is not None:
        ax[0].axvline(estimated_t_peak * 1e6, color='green', linestyle=':', label='t_peak')

    ax[0].set_title('Bubble Radius vs Time')
    ax[0].set_xlabel('Time (μs)')
    ax[0].set_ylabel('Radius (μm)')
    ax[0].legend()
    ax[0].grid(True)

    # Plot growth rate
    ax[1].plot(t * 1e6, dRdt, label='Growth Rate', color='blue')
    ax[1].set_title('Bubble Growth Rate vs Time')
    ax[1].set_xlabel('Time (μs)')
    ax[1].set_ylabel('Growth Rate (m/s)')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()




        
def plot_rt_only_comparison(t_base, R_base, t_rt, R_rt, t_peak):


    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Radius comparison
    ax[0].plot(t_base * 1e6, R_base * 1e6, label='Baseline RP Radius', color='blue')
    ax[0].plot(t_rt * 1e6, R_rt * 1e6, label='RT-Modulated Radius', color='orange', linestyle='--')
    ax[0].axvline(t_peak * 1e6, color='green', linestyle=':', label='t_peak')
    ax[0].set_title('Bubble Radius vs Time (RT Only)')
    ax[0].set_xlabel('Time (μs)')
    ax[0].set_ylabel('Radius (μm)')
    ax[0].legend()
    ax[0].grid(True)

    # Growth rate comparison
    ax[1].plot(t_base * 1e6, np.gradient(R_base, t_base), label='Baseline Growth Rate', color='blue')
    ax[1].plot(t_rt * 1e6, np.gradient(R_rt, t_rt), label='RT Growth Rate', color='orange')
    ax[1].set_title('Bubble Growth Rate vs Time (RT Only)')
    ax[1].set_xlabel('Time (μs)')
    ax[1].set_ylabel('Growth Rate (m/s)')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()



def plot_nf_only_comparison(t_base, R_base, t_nf, R_nf):


    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Radius comparison
    ax[0].plot(t_base * 1e6, R_base * 1e6, label='Baseline RP Radius', color='blue')
    ax[0].plot(t_nf * 1e6, R_nf * 1e6, label='NF-Modulated Radius', color='orange', linestyle='--')
    ax[0].set_title('Bubble Radius vs Time (NF Only)')
    ax[0].set_xlabel('Time (μs)')
    ax[0].set_ylabel('Radius (μm)')
    ax[0].legend()
    ax[0].grid(True)

    # Growth rate comparison
    ax[1].plot(t_base * 1e6, np.gradient(R_base, t_base), label='Baseline Growth Rate', color='blue')
    ax[1].plot(t_nf * 1e6, np.gradient(R_nf, t_nf), label='NF Growth Rate', color='orange')
    ax[1].set_title('Bubble Growth Rate vs Time (NF Only)')
    ax[1].set_xlabel('Time (μs)')
    ax[1].set_ylabel('Growth Rate (m/s)')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()



def plot_nf_comparison(t_base, R_base, t_nf, R_nf):


    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Radius comparison
    ax[0].plot(t_base * 1e6, R_base * 1e6, label='Baseline RP Radius', color='blue')
    ax[0].plot(t_nf * 1e6, R_nf * 1e6, label='RT + NF-Modulated Radius', color='orange', linestyle='--')
    ax[0].set_title('Bubble Radius vs Time')
    ax[0].set_xlabel('Time (μs)')
    ax[0].set_ylabel('Radius (μm)')
    ax[0].legend()
    ax[0].grid(True)

    # Growth rate comparison
    ax[1].plot(t_base * 1e6, np.gradient(R_base, t_base), label='Baseline Growth Rate', color='blue')
    ax[1].plot(t_nf * 1e6, np.gradient(R_nf, t_nf), label='RT + NF Growth Rate', color='orange')
    ax[1].set_title('Bubble Growth Rate vs Time')
    ax[1].set_xlabel('Time (μs)')
    ax[1].set_ylabel('Growth Rate (m/s)')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()



def plot_ingress_comparison(t_base, R_base, t_ingress, R_ingress):


    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Radius comparison
    ax[0].plot(t_base * 1e6, R_base * 1e6, label='Baseline RP Radius', color='blue')
    ax[0].plot(t_ingress * 1e6, R_ingress * 1e6, label='Ingress-Modulated Radius', color='orange', linestyle='--')
    ax[0].set_title('Bubble Radius vs Time')
    ax[0].set_xlabel('Time (μs)')
    ax[0].set_ylabel('Radius (μm)')
    ax[0].legend()
    ax[0].grid(True)

    # Growth rate comparison
    ax[1].plot(t_base * 1e6, np.gradient(R_base, t_base), label='Baseline Growth Rate', color='blue')
    ax[1].plot(t_ingress * 1e6, np.gradient(R_ingress, t_ingress), label='Ingress Growth Rate', color='orange')
    ax[1].set_title('Bubble Growth Rate vs Time')
    ax[1].set_xlabel('Time (μs)')
    ax[1].set_ylabel('Growth Rate (m/s)')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()



def plot_nf_ingress_comparison(t_base, R_base, t_mod, R_mod):


    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Radius comparison
    ax[0].plot(t_base * 1e6, R_base * 1e6, label='Baseline RP Radius', color='blue')
    ax[0].plot(t_mod * 1e6, R_mod * 1e6, label='NF + Ingress Radius', color='orange', linestyle='--')
    ax[0].set_title('Bubble Radius vs Time')
    ax[0].set_xlabel('Time (μs)')
    ax[0].set_ylabel('Radius (μm)')
    ax[0].legend()
    ax[0].grid(True)

    # Growth rate comparison
    ax[1].plot(t_base * 1e6, np.gradient(R_base, t_base), label='Baseline Growth Rate', color='blue')
    ax[1].plot(t_mod * 1e6, np.gradient(R_mod, t_mod), label='NF + Ingress Growth Rate', color='orange')
    ax[1].set_title('Bubble Growth Rate vs Time')
    ax[1].set_xlabel('Time (μs)')
    ax[1].set_ylabel('Growth Rate (m/s)')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()



def plot_temporal_comparison(t_base, R_base, t_temp, R_temp):


    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Radius comparison
    ax[0].plot(t_base * 1e6, R_base * 1e6, label='Baseline RP Radius', color='blue')
    ax[0].plot(t_temp * 1e6, R_temp * 1e6, label='Temporal-Modulated Radius', color='orange', linestyle='--')
    ax[0].set_title('Bubble Radius vs Time')
    ax[0].set_xlabel('Time (μs)')
    ax[0].set_ylabel('Radius (μm)')
    ax[0].legend()
    ax[0].grid(True)

    # Growth rate comparison
    ax[1].plot(t_base * 1e6, np.gradient(R_base, t_base), label='Baseline Growth Rate', color='blue')
    ax[1].plot(t_temp * 1e6, np.gradient(R_temp, t_temp), label='Temporal Growth Rate', color='orange')
    ax[1].set_title('Bubble Growth Rate vs Time')
    ax[1].set_xlabel('Time (μs)')
    ax[1].set_ylabel('Growth Rate (m/s)')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()



def plot_qac_comparison(t_base, R_base, t_mod, R_mod):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    ax[0].plot(t_base * 1e6, R_base * 1e6, label='Baseline RP Radius', color='blue')
    ax[0].plot(t_mod * 1e6, R_mod * 1e6, label='QAC-Modulated Radius', color='orange', linestyle='--')
    ax[0].set_title('Bubble Radius vs Time')
    ax[0].set_xlabel('Time (μs)')
    ax[0].set_ylabel('Radius (μm)')
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(t_base * 1e6, np.gradient(R_base, t_base), label='Baseline Growth Rate', color='blue')
    ax[1].plot(t_mod * 1e6, np.gradient(R_mod, t_mod), label='QAC Growth Rate', color='orange')
    ax[1].set_title('Bubble Growth Rate vs Time')
    ax[1].set_xlabel('Time (μs)')
    ax[1].set_ylabel('Growth Rate (m/s)')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()




def plot_trc_comparison(t_base, R_base, t_trc, R_trc):
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Radius comparison
    ax[0].plot(t_base * 1e6, R_base * 1e6, label='Baseline RP Radius', color='blue')
    ax[0].plot(t_trc * 1e6, R_trc * 1e6, label='TRC-Modulated Radius', color='orange', linestyle='--')
    ax[0].set_title('Bubble Radius vs Time')
    ax[0].set_xlabel('Time (μs)')
    ax[0].set_ylabel('Radius (μm)')
    ax[0].legend()
    ax[0].grid(True)

    # Growth rate comparison
    ax[1].plot(t_base * 1e6, np.gradient(R_base, t_base), label='Baseline Growth Rate', color='blue')
    ax[1].plot(t_trc * 1e6, np.gradient(R_trc, t_trc), label='TRC Growth Rate', color='orange')
    ax[1].set_title('Bubble Growth Rate vs Time')
    ax[1].set_xlabel('Time (μs)')
    ax[1].set_ylabel('Growth Rate (m/s)')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()

def plot_trc_energy(t, R, trc_modulation_func):
    """
    Plots TRC energy over time using a passed-in modulation function.
    
    Parameters:
        t (array): Time array
        R (array): Radius array
        trc_modulation_func (callable): Function that computes TRC energy from t and R
    """
    energy = trc_modulation_func(t, R)
    
    import matplotlib.pyplot as plt
    plt.plot(t * 1e6, energy, color='blue')
    plt.xlabel('Time (μs)')
    plt.ylabel('TRC Energy')
    plt.title('Triadic Resonance Coupling Energy Over Time')
    plt.grid(True)
    plt.tight_layout()
    plt.show()




def plot_energy_opt_comparison(t_base, R_base, t_energy, R_energy, E_opt=None):
   # fig, ax = plt.subplots(3, 1, figsize=(10, 12))
    fig, ax = plt.subplots(3, 1, figsize=(9, 9))  # ✅ 10% smaller THIS IS 9 inches!

    # Radius comparison
    ax[0].plot(t_base * 1e6, R_base * 1e6, label='Baseline RP Radius', color='blue')
    ax[0].plot(t_energy * 1e6, R_energy * 1e6, label='Energy-Optimized Radius', color='orange', linestyle='--')
    ax[0].set_title('Bubble Radius vs Time')
    ax[0].set_xlabel('Time (μs)')
    ax[0].set_ylabel('Radius (μm)')
    ax[0].legend()
    ax[0].grid(True)

    # Growth rate comparison
    ax[1].plot(t_base * 1e6, np.gradient(R_base, t_base), label='Baseline Growth Rate', color='blue')
    ax[1].plot(t_energy * 1e6, np.gradient(R_energy, t_energy), label='Energy-Optimized Growth Rate', color='orange', linestyle='--')
    ax[1].set_title('Bubble Growth Rate vs Time')
    ax[1].set_xlabel('Time (μs)')
    ax[1].set_ylabel('Growth Rate (m/s)')
    ax[1].legend()
    ax[1].grid(True)

    # Energy optimization diagnostic
    if E_opt is not None:
        ax[2].plot(t_energy * 1e6, E_opt, label='Energy Optimization Diagnostic', color='purple')
        ax[2].set_title('E_opt vs Time')
        ax[2].set_xlabel('Time (μs)')
        ax[2].set_ylabel('E_opt')
        ax[2].legend()
        ax[2].grid(True)


    plt.tight_layout()
    plt.show()




def plot_diagnostics(t, R, dRdt, diagnostic, label='Diagnostic'):
    collapse_index = np.argmin(R)
    collapse_time = t[collapse_index]
    print(f"[BASELINE] Collapse time: {collapse_time:.6e} s")

    #fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    fig, ax = plt.subplots(3, 1, figsize=(9, 9))  # ✅ 10% smaller THIS IS 9 inches!


    # Radius
    ax[0].plot(t * 1e6, R * 1e6, label='Bubble Radius')
    ax[0].axvline(collapse_time * 1e6, color='red', linestyle='--', label='Collapse Peak')
    ax[0].set_ylabel('Radius (μm)')
    ax[0].set_title('Bubble Radius vs Time')
    ax[0].legend()
    ax[0].grid(True)

    # Growth Rate
    ax[1].plot(t * 1e6, dRdt * 1e6, label='Growth Rate')
    ax[1].axvline(collapse_time * 1e6, color='red', linestyle='--')
    ax[1].set_ylabel('Growth Rate (μm/s)')
    ax[1].set_title('Bubble Growth Rate vs Time')
    ax[1].legend()
    ax[1].grid(True)

    # Diagnostic Layer


    ax[2].plot(t * 1e6, diagnostic, color='purple', label=label)
    ax[2].axvline(collapse_time * 1e6, color='red', linestyle='--')
    ax[2].set_ylabel(f'{label}')
    ax[2].set_xlabel('Time (μs)')
    ax[2].set_title(f'{label} Diagnostic Over Time')
    ax[2].legend()
    ax[2].grid(True)

    plt.tight_layout()
    plt.tight_layout()
    plt.show()





def plot_angular_comparison(t_base, R_base, t_ang, R_ang, angular_mod_list=None):

    fig, ax = plt.subplots(3, 1, figsize=(9, 9))

    # Radius comparison
    ax[0].plot(t_base * 1e6, R_base * 1e6, label='Baseline RP Radius', color='blue')
    ax[0].plot(t_ang * 1e6, R_ang * 1e6, label='Angular-Modulated Radius', color='orange', linestyle='--')
    ax[0].set_title('Bubble Radius vs Time')
    ax[0].set_xlabel('Time (μs)')
    ax[0].set_ylabel('Radius (μm)')
    ax[0].legend()
    ax[0].grid(True)

    # Growth rate comparison
    ax[1].plot(t_base * 1e6, np.gradient(R_base, t_base), label='Baseline Growth Rate', color='blue')
    ax[1].plot(t_ang * 1e6, np.gradient(R_ang, t_ang), label='Angular Growth Rate', color='orange')
    ax[1].set_title('Bubble Growth Rate vs Time')
    ax[1].set_xlabel('Time (μs)')
    ax[1].set_ylabel('Growth Rate (m/s)')
    ax[1].legend()
    ax[1].grid(True)

    # Modulated external pressure diagnostic (p_ext)
    if angular_mod_list is not None and len(angular_mod_list) > 0:
        if len(angular_mod_list) != len(t_ang):
            internal_t = np.linspace(t_ang[0], t_ang[-1], len(angular_mod_list))
            interp_func = interp1d(internal_t, angular_mod_list, kind='linear', fill_value="extrapolate")
            angular_mod_resampled = interp_func(t_ang)
        else:
            angular_mod_resampled = angular_mod_list


        ax[2].plot(t_ang * 1e6, angular_mod_resampled, label='Modulated External Pressure (p_ext)', color='purple')
        ax[2].set_title('Modulated External Pressure vs Time')
        ax[2].set_xlabel('Time (μs)')
        ax[2].set_ylabel('p_ext (Pa)')
        ax[2].legend()
        ax[2].grid(True)


    plt.tight_layout()
    plt.show()
    
    
def rt_nf_tm_qac_combined(t_base, R_base, t_mod, R_mod):


    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Radius comparison
    ax[0].plot(t_base * 1e6, R_base * 1e6, label='Baseline RP Radius', color='blue')
    ax[0].plot(t_mod * 1e6, R_mod * 1e6, label='RT + NF + TM + QAC Radius', color='darkorange', linestyle='--')
    ax[0].set_title('Bubble Radius vs Time')
    ax[0].set_xlabel('Time (μs)')
    ax[0].set_ylabel('Radius (μm)')
    ax[0].legend()
    ax[0].grid(True)

    # Growth rate comparison
    ax[1].plot(t_base * 1e6, np.gradient(R_base, t_base), label='Baseline Growth Rate', color='blue')
    ax[1].plot(t_mod * 1e6, np.gradient(R_mod, t_mod), label='RT + NF + TM + QAC Growth Rate', color='darkorange')
    ax[1].set_title('Bubble Growth Rate vs Time')
    ax[1].set_xlabel('Time (μs)')
    ax[1].set_ylabel('Growth Rate (m/s)')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()

def tm_trc_qrc_combined(t_base, R_base, t_mod, R_mod):


    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Radius comparison
    ax[0].plot(t_base * 1e6, R_base * 1e6, label='Baseline RP Radius', color='blue')
    ax[0].plot(t_mod * 1e6, R_mod * 1e6, label='TM + TRC + QRC Radius', color='orange', linestyle='--')
    ax[0].set_title('Bubble Radius vs Time')
    ax[0].set_xlabel('Time (μs)')
    ax[0].set_ylabel('Radius (μm)')
    ax[0].legend()
    ax[0].grid(True)

    # Growth rate comparison
    ax[1].plot(t_base * 1e6, np.gradient(R_base, t_base), label='Baseline Growth Rate', color='blue')
    ax[1].plot(t_mod * 1e6, np.gradient(R_mod, t_mod), label='TM + TRC + QRC Growth Rate', color='orange')
    ax[1].set_title('Bubble Growth Rate vs Time')
    ax[1].set_xlabel('Time (μs)')
    ax[1].set_ylabel('Growth Rate (m/s)')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()

    
    #This is the same but with tpeak shown, use this when you need to show the offset, in that case just enable this chart, for normal if you want it then enable the one above.
def eopt_rt_qac_combined(t_base, R_base, t_mod, R_mod, t_peak, t_gate_center):

    
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    
    # Radius comparison
    ax[0].plot(t_base * 1e6, R_base * 1e6, label='Baseline RP Radius', color='blue')
    ax[0].plot(t_mod * 1e6, R_mod * 1e6, label='Eopt + RT + QAC Radius', color='orange', linestyle='--')
    

    # Add vertical lines for t_peak and t_peak1
    ax[0].axvline(t_peak * 1e6, color='blue', linestyle=':', label='Baseline t_peak')
    #ax[0].axvline(t_peak1 * 1e6, color='orange', linestyle=':', label='Eopt Gate Center')
    ax[0].axvline(t_gate_center * 1e6, color='orange', linestyle='--', label='Eopt Gate Center')
    
    ax[0].set_title('Bubble Radius vs Time')
    ax[0].set_xlabel('Time (μs)')
    ax[0].set_ylabel('Radius (μm)')
    ax[0].legend()
    ax[0].grid(True)
    
    # Growth rate comparison
    ax[1].plot(t_base * 1e6, np.gradient(R_base, t_base), label='Baseline Growth Rate', color='blue')
    ax[1].plot(t_mod * 1e6, np.gradient(R_mod, t_mod), label='Eopt + RT + QAC Growth Rate', color='orange')
    
    # Add vertical lines for t_peak and t_peak1
    ax[1].axvline(t_peak * 1e6, color='blue', linestyle=':', label='Baseline t_peak')
    #ax[1].axvline(t_peak1 * 1e6, color='orange', linestyle=':', label='Eopt Gate Center')
    ax[1].axvline(t_gate_center * 1e6, color='orange', linestyle=':', label='Eopt Gate Center')
    
    ax[1].set_title('Bubble Growth Rate vs Time')
    ax[1].set_xlabel('Time (μs)')
    ax[1].set_ylabel('Growth Rate (m/s)')
    ax[1].legend()
    ax[1].grid(True)
    
    plt.tight_layout()
    plt.show()

    
    
def resonance_yield_plot(t_base, R_base, t_mod, R_mod):


    fig, ax = plt.subplots(2, 1, figsize=(10, 8))

    # Radius comparison
    ax[0].plot(t_base * 1e6, R_base * 1e6, label='Baseline RP Radius', color='blue')
    ax[0].plot(t_mod * 1e6, R_mod * 1e6, label='Resonance Yield Radius', color='orange', linestyle='--')
    ax[0].set_title('Bubble Radius vs Time')
    ax[0].set_xlabel('Time (μs)')
    ax[0].set_ylabel('Radius (μm)')
    ax[0].legend()
    ax[0].grid(True)

    # Growth rate comparison
    ax[1].plot(t_base * 1e6, np.gradient(R_base, t_base), label='Baseline Growth Rate', color='blue')
    ax[1].plot(t_mod * 1e6, np.gradient(R_mod, t_mod), label='Resonance Yield Growth Rate', color='orange')
    ax[1].set_title('Bubble Growth Rate vs Time')
    ax[1].set_xlabel('Time (μs)')
    ax[1].set_ylabel('Growth Rate (m/s)')
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()

def resonance_yield_diagnostics(t_raw, R_raw, t_mod, R_mod):


    dRdt_raw = np.gradient(R_raw, t_raw)
    dRdt_mod = np.gradient(R_mod, t_mod)

    energy_raw = dRdt_raw**2
    energy_mod = dRdt_mod**2

    collapse_index_raw = np.argmin(R_raw)
    collapse_index_mod = np.argmin(R_mod)

    fig, ax = plt.subplots(3, 1, figsize=(10, 8))

    # Radius comparison
    ax[0].plot(t_raw * 1e6, R_raw * 1e6, label='Raw RP Radius', color='blue')
    ax[0].plot(t_mod * 1e6, R_mod * 1e6, label='Resonance Yield Radius', color='orange', linestyle='--')
    ax[0].axvline(t_raw[collapse_index_raw] * 1e6, color='blue', linestyle=':', label='Raw Collapse')
    ax[0].axvline(t_mod[collapse_index_mod] * 1e6, color='purple', linestyle=':', label='Modulated Collapse')
    ax[0].set_title('Bubble Radius vs Time')
    ax[0].set_ylabel('Radius (µm)')
    ax[0].legend()
    ax[0].grid(True)

    # Growth rate comparison
    ax[1].plot(t_raw * 1e6, dRdt_raw, label='Raw Growth Rate', color='blue')
    ax[1].plot(t_mod * 1e6, dRdt_mod, label='Modulated Growth Rate', color='orange')
    ax[1].set_title('Growth Rate vs Time')
    ax[1].set_ylabel('Growth Rate (m/s)')
    ax[1].legend()
    ax[1].grid(True)


    # Energy proxy comparison
    raw_level = np.mean(energy_raw)  # or np.max(energy_raw), depending on what you want to show
    
    ax[2].plot(
        [t_raw[0] * 1e6, t_raw[-1] * 1e6],  # span full time range
        [raw_level, raw_level],            # constant value
        label='Raw Energy Proxy (dR/dt²)', 
        color='blue', 
        linestyle='dotted'
    )
    
    ax[2].plot(
        t_mod * 1e6, 
        energy_mod, 
        label='Modulated Energy Proxy (dR/dt²)', 
        color='orange'
    )
    
    ax[2].set_title('Energy Proxy vs Time')
    ax[2].set_xlabel('Time (µs)')
    ax[2].set_ylabel('Energy Proxy')
    ax[2].legend()
    ax[2].grid(True)
    
    plt.tight_layout()
    plt.show()


def plot_coherence_surface(t_array, theta_vals, coherence_func):
    import matplotlib.pyplot as plt
    import numpy as np

    T, Θ = np.meshgrid(t_array * 1e6, theta_vals)
    C = np.array([[coherence_func(t, theta) for theta in theta_vals] for t in t_array]).T


    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(T, Θ, C, cmap='viridis', edgecolor='none')
    ax.set_title('Coherence Index Surface')
    ax.set_xlabel('Time (μs)')
    ax.set_ylabel('Angle θ (rad)')
    ax.set_zlabel('Coherence Index C(t, θ)')
    plt.tight_layout()
    plt.show()
