import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar

# Enable LaTeX-style fonts
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 14,
    "font.size": 12,
    "legend.fontsize": 10
})

# Parameters
kB = 1.0
Tc = 15.03
omega_D = 20 * Tc

# --------------- Gap equation from Problem 1 ---------------

def gap_integral(Delta, T):
    def integrand(xi):
        E = np.sqrt(xi**2 + Delta**2)
        term1 = np.tanh(E / (2 * kB * T)) / E
        term2 = np.tanh(xi / (2 * kB * Tc)) / xi if xi != 0 else 1/(2*kB*Tc)
        return term1 - term2
    val, _ = quad(integrand, 0, omega_D, limit=200)
    return val

def solve_gap(T):
    if T >= Tc:
        return 0.0
    sol = root_scalar(gap_integral, args=(T,), bracket=[1e-6, 5*kB*Tc], method='bisect')
    return sol.root if sol.converged else np.nan

# Precompute gap on a fine temperature grid
T_vals = np.linspace(0.001, Tc, 500)
Delta_vals = np.array([solve_gap(T) for T in T_vals])
Delta0 = Delta_vals[0]

# Interpolate for quick lookup
from scipy.interpolate import interp1d
gap_interp = interp1d(T_vals, Delta_vals, kind='cubic', fill_value=0.0, bounds_error=False)

# --------------- Tunnelling conductance ---------------

def conductance(eV, T):
    D = gap_interp(T)
    if D < 1e-8:
        return 1.0

    def integrand(E):
        absE = np.abs(E)
        if absE <= D:
            return 0.0
        Ns = absE / np.sqrt(absE**2 - D**2)
        arg = (E + eV) / (2 * kB * T)
        if np.abs(arg) > 500:
            return 0.0
        sech2 = 1.0 / np.cosh(arg)**2
        return Ns * sech2 / (4 * kB * T)

    Emax = max(10 * D, 5 * abs(eV) + 40 * kB * T)
    eps = 1e-6 * D

    # Integrate in pieces to avoid the singularity at E = ±D
    val1, _ = quad(integrand, -Emax, -D - eps, limit=300, epsabs=1e-10, epsrel=1e-8)
    val2, _ = quad(integrand, D + eps, Emax, limit=300, epsabs=1e-10, epsrel=1e-8)
    return val1 + val2

# --------------- Compute conductance curves ---------------

eV_values = np.linspace(0, 3.0 * Delta0, 400)

t_ratios = [0.03, 0.3, 0.5, 0.7, 0.9, 0.95]
colors = ["#0000FF", "#007070", "#008000", "#FF8C00", "#CC0000", "#800080"]
labels = [r"$T/T_c = 0.03$", r"$T/T_c = 0.3$", r"$T/T_c = 0.5$",
          r"$T/T_c = 0.7$", r"$T/T_c = 0.9$", r"$T/T_c = 0.95$"]

plt.figure(figsize=(6, 4))

for tr, col, lab in zip(t_ratios, colors, labels):
    T = tr * Tc
    print(f"Computing T/Tc = {tr} ...")
    G_vals = np.array([conductance(eV, T) for eV in eV_values])
    # Clamp for display
    G_vals = np.clip(G_vals, 0, 6)
    plt.plot(eV_values / Delta0, G_vals, color=col, linewidth=1.2, label=lab)

# T = Tc line
plt.axhline(1.0, color='black', linewidth=1.0, linestyle='--', label=r"$T = T_c$")

# Fancy grid
plt.minorticks_on()
plt.grid(which='major', linestyle='--', linewidth=0.8, alpha=0.6)
plt.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.4)

plt.xlabel(r"$eV/\Delta(0)$")
plt.ylabel(r"$G(V)/G_N$")
plt.xlim(0, 3)
plt.ylim(0, 5)
plt.legend(loc='upper right')
plt.tight_layout()
plt.show()

print("Done. Saved as graphics/3.png")
