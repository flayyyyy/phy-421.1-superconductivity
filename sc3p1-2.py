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
    "legend.fontsize": 12
})

# Physical constants (can set kB = 1 for convenience)
kB = 1.0

# Parameters
Tc = 15.03                  # Critical temperature (K)
omega_D = 20 * Tc          # Debye cutoff (large compared to Tc)

# Integral equation (difference form to eliminate coupling constant)
def gap_integral(Delta, T):
    def integrand(xi):
        E = np.sqrt(xi**2 + Delta**2)
        term1 = np.tanh(E / (2 * kB * T)) / E
        term2 = np.tanh(xi / (2 * kB * Tc)) / xi if xi != 0 else 1/(2*kB*Tc)
        return term1 - term2
    val, _ = quad(integrand, 0, omega_D, limit=200)
    return val

# Solve Δ(T) for a given T
def solve_gap(T):
    if T >= Tc:
        return 0.0
    sol = root_scalar(gap_integral, args=(T,), bracket=[1e-6, 5*kB*Tc], method='bisect')
    return sol.root if sol.converged else np.nan

# Temperature range
T_vals = np.linspace(0.001, Tc, 1000)

# Compute gap values
Delta_vals = np.array([solve_gap(T) for T in T_vals])

# Normalize if needed
Delta0 = Delta_vals[0]
Delta_norm = Delta_vals / Delta0
T_norm = T_vals / Tc

# Example: print values
for T, D in zip(T_vals, Delta_vals):
    print(f"T = {T:.3f}, Delta = {D:.5f}")


# Dimensionless ratio Δ(T) / (kB Tc)
Delta_scaled = Delta_vals / (kB * Tc)
T_scaled = T_vals / Tc

plt.figure(figsize=(6,4))

# Curve (change color + thickness)
plt.plot(T_scaled, Delta_scaled, color="#000000", linewidth=1)

# Fancy grid (major + minor)
plt.minorticks_on()
plt.grid(which='major', linestyle='--', linewidth=0.8, alpha=0.6)
plt.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.4)

# # Optional: horizontal reference line at Δ/Δ(0)=1
# plt.axhline(1, linestyle='--', linewidth=1.0, alpha=0.6, color='gray')

# Compute values
Delta0 = Delta_vals[0]
ratio = Delta0 / (kB * Tc)

# Add text box inside plot
textstr = (
    r"$T_c = %.2f\,\mathrm{K}$" "\n"
    r"$\Delta(0)/(k_B T_c) = %.3f$"
) % (Tc, ratio)

plt.text(
    0.68, 0.93, textstr,             # position in axes coordinates
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top',
    # bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
)

plt.xlabel(r"$T/T_c$")
plt.ylabel(r"$\Delta(T)/(k_B T_c)$")

plt.xlim(0, 1)
plt.ylim(0, None)

plt.tight_layout()
plt.show()






def superfluid_density(T, Delta):
    if T == 0:
        return 1.0
    
    def integrand(E):
        return (E / np.sqrt(E**2 - Delta**2)) * \
               (1 / (4 * kB * T)) * \
               (1 / np.cosh(E / (2 * kB * T))**2)
    
    val, _ = quad(integrand, Delta, 20*Tc, limit=200)
    return 1 - 2 * val

# Compute 1/lambda^2(T)
rho_s = np.array([superfluid_density(T, D) for T, D in zip(T_vals, Delta_vals)])

# # Plot
# plt.figure()
# plt.plot(T_vals/Tc, rho_s)

# plt.xlabel(r"$T/T_c$")
# plt.ylabel(r"$\lambda^2(0)/\lambda^2(T)$")
# plt.title(r"Superfluid Density vs Temperature")

# plt.grid()
# plt.show()

plt.figure(figsize=(6,4))

# Curve (change color + thickness)
plt.plot(T_vals/Tc, rho_s, color="#000000", linewidth=1)

# Fancy grid (major + minor)
plt.minorticks_on()
plt.grid(which='major', linestyle='--', linewidth=0.8, alpha=0.6)
plt.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.4)

# # Optional: horizontal reference line at Δ/Δ(0)=1
# plt.axhline(1, linestyle='--', linewidth=1.0, alpha=0.6, color='gray')

# # Compute values
# Delta0 = Delta_vals[0]
# ratio = Delta0 / (kB * Tc)

# # Add text box inside plot
# textstr = (
#     r"$T_c = %.2f\,\mathrm{K}$" "\n"
#     r"$\Delta(0)/(k_B T_c) = %.3f$"
# ) % (Tc, ratio)

# plt.text(
#     0.68, 0.93, textstr,             # position in axes coordinates
#     transform=plt.gca().transAxes,
#     fontsize=12,
#     verticalalignment='top',
#     # bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
# )

plt.xlabel(r"$T/T_c$")
plt.ylabel(r"$\lambda^2(0)/\lambda^2(T)$")

plt.xlim(0, 1)
plt.ylim(0, None)

plt.tight_layout()
plt.show()