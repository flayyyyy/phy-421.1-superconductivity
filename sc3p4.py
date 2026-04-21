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

# Compute gap on a fine temperature grid
print("Computing gap...")
T_vals = np.linspace(0.001, Tc, 1000)
Delta_vals = np.array([solve_gap(T) for T in T_vals])
Delta0 = Delta_vals[0]

# --------------- Specific heat ---------------

def ces_over_gammaTc(T, Delta, dDelta2_dT):
    """
    C_es / (gamma * Tc) = (6 / pi^2 t^2) * integral of
    f(E)[1-f(E)] * (E^2/(kB Tc)^2 - (t/2) d(Delta/kBTc)^2/dt ) dxi/(kB Tc)
    where t = T/Tc, and everything is in units of kB=1.
    """
    t = T / Tc
    if t < 1e-6:
        return 0.0

    D = Delta
    dD2dt = dDelta2_dT  # d(Delta^2)/dT, but we need d(Delta^2)/dt = Tc * d(Delta^2)/dT

    # Convert dDelta^2/dT to dDelta^2/dt (where t = T/Tc)
    dD2_dtred = Tc * dD2dt  # d(Delta^2)/d(t)

    def integrand(xi):
        E = np.sqrt(xi**2 + D**2)
        u = E / T
        if u > 500:
            return 0.0
        eu = np.exp(u)
        f = 1.0 / (eu + 1.0)
        f1mf = f * (1.0 - f)
        Etilde2 = (E / Tc)**2
        term = Etilde2 - 0.5 * t * dD2_dtred / Tc**2
        return f1mf * term

    ximax = max(20.0 * T, 10.0 * D, 10.0 * Tc)
    val, _ = quad(integrand, 0, ximax, limit=500, epsabs=1e-12, epsrel=1e-10)
    return (6.0 / (np.pi**2 * t**2)) * val / Tc


# Compute d(Delta^2)/dT numerically
print("Computing specific heat...")
dDelta2_dT = np.zeros_like(T_vals)
for i in range(len(T_vals)):
    if i == 0:
        dDelta2_dT[i] = (Delta_vals[i+1]**2 - Delta_vals[i]**2) / (T_vals[i+1] - T_vals[i])
    elif i == len(T_vals) - 1:
        dDelta2_dT[i] = (Delta_vals[i]**2 - Delta_vals[i-1]**2) / (T_vals[i] - T_vals[i-1])
    else:
        dDelta2_dT[i] = (Delta_vals[i+1]**2 - Delta_vals[i-1]**2) / (T_vals[i+1] - T_vals[i-1])

# Compute C_es / (gamma Tc)
ces_vals = np.array([ces_over_gammaTc(T_vals[i], Delta_vals[i], dDelta2_dT[i])
                     for i in range(len(T_vals))])

# Extend above Tc with normal-state value C_en = gamma T => C_en/(gamma Tc) = T/Tc
T_above = np.linspace(Tc * 1.001, 1.3 * Tc, 50)
ces_above = T_above / Tc

T_all = np.concatenate([T_vals, T_above])
ces_all = np.concatenate([ces_vals, ces_above])

# --------------- Plot ---------------

plt.figure(figsize=(6, 4))

# BCS curve
plt.plot(T_all / Tc, ces_all, color="#0000FF", linewidth=1.2, label=r"$C_{es}/\gamma T_c$ (BCS)")

# Normal state line
t_plot = np.linspace(0, 1.3, 200)
plt.plot(t_plot, t_plot, color="#CC0000", linewidth=1.0, linestyle='--',
         label=r"$C_{en}/\gamma T_c = T/T_c$ (Normal)")

# Fancy grid
plt.minorticks_on()
plt.grid(which='major', linestyle='--', linewidth=0.8, alpha=0.6)
plt.grid(which='minor', linestyle=':', linewidth=0.5, alpha=0.4)

plt.xlabel(r"$T/T_c$")
plt.ylabel(r"$C_{es}/\gamma T_c$")
plt.xlim(0, 1.3)
plt.ylim(0, 3)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
