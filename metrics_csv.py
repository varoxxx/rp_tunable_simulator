
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
import csv
def write_comparison_csv(filename, metrics_base, metrics_modulated, mod_label):
    with open(filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Baseline", mod_label])

        all_keys = sorted(set(metrics_base.keys()) | set(metrics_modulated.keys()))
        for key in all_keys:
            def safe_format(value):
                return f"{value:.4f}" if isinstance(value, (int, float, np.float64)) else str(value)
            writer.writerow([
                key,
                safe_format(metrics_base.get(key, "")),
                safe_format(metrics_modulated.get(key, ""))
            ])
