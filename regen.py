from matplotlib._api import define_aliases
import matplotlib.pyplot as plt
import numpy as np
from fluids.friction import friction_factor
from fluids.core import Reynolds
from CoolProp.CoolProp import PQ_INPUTS, PropsSI

edh = 0.001     # Relative roughness (epsilon / d)
d_h = 0.003
p_in = 32e5 #pressure in also defined inside calculation loop. Change there also
p_c = 20e5
T_in = 300
m_dot = 0.003
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
    p_in = 30e5
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
    
    #fluid properties
    re_local= np.zeros(N)

    # Initialise properties at the inlet
    dl = np.diff(l)
    area= np.repeat(1,N)
    p[0] = p_in

    for j in range(10):
        for i in range(N - 1):
            rho_local, mu_local, cp_local, h_local, k_local = (properties(T, p))
            v= m_dot/(area*rho_local)
            re_local = Reynolds(V=v, D=d_h, rho=rho_local,mu=mu_local)
            f= friction_factor(Re=re_local, eD=edh)
            dp = p_loss(f, rho_local, v, dl, d_h)
            for i in range(N-1):
                p[i+1]=p[i]- dp[i]
            p[0]=p_in

        diff= max(np.diff(p))
        res = (p[N-1]-p_c)/(p_in - p_c)
        print(res)
        if abs(res) < 0.001  and abs(diff) <=0.05*abs(p_in-p_c):
            break

        p_in = p_in / np.sqrt(1 - res)


    fig,ax= plt.subplots()
    ax.plot(l,p,label="p versus l")
#    ax.plot(l,v,label="v versus l")
    print(v)
    plt.show()
    return 0

if __name__ == "__main__":
    main()

