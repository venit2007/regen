from matplotlib._api import define_aliases
import matplotlib.pyplot as plt
import numpy as np
from fluids.friction import friction_factor
from fluids.core import Reynolds
from CoolProp.CoolProp import PropsSI

edh = 0.001     # Relative roughness (epsilon / d)
d_h = 0.003
p_in = 32e5
p_c = 20e5
T_in = 300

def p_loss(f, rho, v, l, d_h):
    dp = f * rho * v**2 * l / (2 * d_h)
    return dp


def qrad_h2o(p_h2o, r_s, t_aw):
    return 5.74 * (p_h2o * r_s)**0.3 * (t_aw / 100)**3.5


def qrad_co2(p_co2, r_s, t_aw):
    return 4 * (p_co2 * r_s)**0.3 * (t_aw / 100)**3.5


def calculate_q(t_aw, t_c, q_rad, h_g, t, k_w, h_c) -> float:
    numerator = (t_aw - t_c) + (q_rad / h_g)
    denominator = (1 / h_g) + (t / k_w) + (1 / h_c)
    return numerator / denominator

def properties(temp,pressure):
    fluid = "Ethanol"
    T = temp
    p= pressure

    rho = PropsSI("D", "T", T, "P", p, fluid)  # kg/m^3
    mu  = PropsSI("V", "T", T, "P", p, fluid)  # Pa s
    cp  = PropsSI("C", "T", T, "P", p, fluid)  # J/(kg K)
    h   = PropsSI("H", "T", T, "P", p, fluid)  # J/kg
    k   = PropsSI("L", "T", T, "P", p, fluid)  # W/(m K)
    return [rho,mu,cp,h,k]

def main():
    dp = p_in - p_c
    length = 0.3 
    l = np.linspace(0, length, 200)
    N = len(l)
    v = np.zeros(N)
    p = np.zeros(N)
    T = np.zeros(N)

    p[0] = p_in
    T[0] = T_in

    # Thermodynamic property arrays
    rho_local = np.zeros(N)
    mu_local = np.zeros(N)
    cp_local = np.zeros(N)
    h_local = np.zeros(N)
    k_local = np.zeros(N)

    # Initialise properties at the inlet
    rho_local[0], mu_local[0], cp_local[0], h_local[0], k_local[0] = (properties(T[0], p[0]))
    dl = np.diff(l)
    p = np.zeros(N)
    area= np.repeat(1,N)
    p[0] = p_in
    rho_local = np.full(N,rho)
    re = Reynolds(V=v, D=d_h, rho=rho_local[0], mu=1e-3)
    f = np.full(N, friction_factor(Re=re, eD=edh))

    while True:
        for i in range(N - 1):
            dp = p_loss(f[i], rho_local[i], v, dl[i], d_h)
            p[i + 1] = p[i] - dp
            v[i]= m_dot/area[i]


            re = Reynolds(
                V=v, D=d_h,
                rho=rho_local[i + 1], mu=1e-3
            )
            f[i + 1] = friction_factor(Re=re, eD=edh)

        p_res = p[N - 1] - p_c
        res = p_res / (p[0] - p_c)
        print(res)

        diff= max(np.diff(p))
        
        if abs(res) < 0.001  and abs(diff) <=0.05*abs(p_in-p_c):
            break

        v = v / np.sqrt(1 - res)


    fig,ax= plt.subplots()
    ax.plot(l,p,label="p versus l")
#    ax.plot(l,v,label="v versus l")
    print(v)
    plt.show()
    return 0

if __name__ == "__main__":
    main()

