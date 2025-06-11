# -*- coding: utf-8 -*-
"""
Created on Thu May 15 13:19:55 2025

@author: 10150018
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

R_cell, L_cell = 9e-3, 0.065
nr, nz = 60,80
Dr, Dz = R_cell/nr, L_cell/nz
r_nodes = (np.arange(nr)+0.5)*Dr
z_nodes = (np.arange(nz)+0.5)*Dz

k_rz = np.full((nr, nz), 0.20)            # separator
idx_pos = r_nodes < 0.25*R_cell
idx_neg = r_nodes > 0.35*R_cell
k_rz[idx_pos,:] = 0.50                    # positive
k_rz[idx_neg,:] = 0.40                    # negative
SOC_grid  = np.array([0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95])
R_curve   = np.array([0.08,0.06,0.048,0.043,0.040,0.040,0.043,0.048,0.058,0.064])
DuT_curve = np.array([+9e-5,+6e-5,+4e-5,+2e-5,0.0,-2e-5,-6e-5,-9e-5,-1.1e-4,-1.3e-4])
R_int_SOC = lambda soc: np.interp(soc, SOC_grid, R_curve)
dUdT_SOC  = lambda soc: np.interp(soc, SOC_grid, DuT_curve)

def solve_T(src, kappa, T_inf=300, h_side=50, h_end=10,
            omega=1.8, tol=1e-5, max_iter=60000):
    T = np.full_like(kappa, T_inf)
    for _ in range(max_iter):
        err = 0.0
        for i in range(nr):
            r_i = (i+0.5)*Dr
            for j in range(1, nz-1):
                if i==0:
                    T_im1, T_ip1 = T[1,j], T[i+1,j]
                elif i==nr-1:
                    k_loc = kappa[i,j]
                    T_im1 = T[i-1,j]
                    T_ip1 = (k_loc/Dr*T_im1 + h_side*T_inf)/(k_loc/Dr + h_side)
                else:
                    T_im1, T_ip1 = T[i-1,j], T[i+1,j]
                T_jm1, T_jp1 = T[i,j-1], T[i,j+1]

                num = (T_ip1+T_im1)*Dz**2 + (T_jp1+T_jm1)*Dr**2 \
                      + src[i,j]*Dr**2*Dz**2 \
                      + (T_ip1-T_im1)*Dz**2*Dr/(2*r_i)
                new = num / (2*(Dr**2+Dz**2))
                diff = new - T[i,j]
                T[i,j] += omega*diff
                err = max(err, abs(diff))
        # axial convection BC
        for i in range(nr):
            k_bot, k_top = kappa[i,0], kappa[i,-1]
            T[i,0]  = (T[i,1] + h_end*Dz/k_bot*T_inf)/(1+h_end*Dz/k_bot)
            T[i,-1] = (T[i,-2]+ h_end*Dz/k_top*T_inf)/(1+h_end*Dz/k_top)
        if err < tol: break
    return T

# ---------- 4. Sweep definition ----------
I_list   = [0.5,1.0,3.0,5.0]
SOC_list = [0.10,0.30,0.50,0.70,0.90]
h_list   = [10,50,200]                 # three cooling coefficients
T_env    = 300.0
V_cell   = np.pi*R_cell**2*L_cell

records=[]
loop=tqdm(total=len(I_list)*len(SOC_list)*len(h_list),desc="I×SOC×h sweep")
for I in I_list:
    for soc in SOC_list:
        R_int = R_int_SOC(soc)
        dUdT  = dUdT_SOC(soc)
        q_irrev = I**2 * R_int / V_cell
        q_rev   = I * T_env * dUdT / V_cell
        q_tot   = q_irrev + q_rev
        src = q_tot / k_rz
        for h in h_list:
            dTmax = np.max(solve_T(src, k_rz, h_side=h)) - T_env
            records.append([I, soc, h, R_int, dUdT, q_irrev, q_rev, q_tot, dTmax])
            loop.update(1)
loop.close()

df = pd.DataFrame(records, columns=[
    "I","SOC","h","R_int","dUdT","q_irrev","q_rev","q_tot","dT_max"
])

df.to_csv("thermal_simulation_results.csv", index=False)
print("CSV saved -> thermal_simulation_results.csv")

plt.figure(figsize=(8,4))
for I in I_list:
    dsub = df.query("I==@I & h==50")
    plt.plot(dsub.SOC, dsub.q_irrev/1e4, '-',  label=f'q_irrev {I}A')
    plt.plot(dsub.SOC, dsub.q_rev/1e4,  '--', label=f'q_rev {I}A')

plt.xlabel("SOC")
plt.ylabel("Heat density (×10⁴ W/m³)")# plt.title("Fig.2 Heat‑source components vs SOC")
plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
plt.grid(True)
plt.tight_layout()
plt.savefig("TSOCh50.png", dpi=300)
plt.show()
for h_val in h_list:
    sub = df[df.h==h_val]
    # Fig.3
    plt.figure(figsize=(6,4))
    for I in I_list:
        plt.plot(sub[sub.I==I].SOC, sub[sub.I==I].dT_max,
                 'o-', label=f'I={I}A')
    plt.xlabel("SOC"); plt.ylabel("ΔT_max (K)")
    plt.ylim(-1,50)
    plt.legend(); plt.grid(); plt.tight_layout();plt.savefig(f"ISOC{h_val}.png", dpi=300);plt.show();plt.savefig("ISOC.png", dpi=300)
    pivot = sub.pivot(index="SOC", columns="I", values="dT_max")
    I_mesh, SOC_mesh = np.meshgrid(pivot.columns, pivot.index)
    plt.figure(figsize=(6,4))
    cp = plt.contourf(I_mesh, SOC_mesh, pivot.values, 20, cmap='hot')
    plt.colorbar(cp, label="ΔT_max (K)")
    plt.xlabel("Current I (A)"); plt.ylabel("SOC")
    plt.tight_layout();plt.savefig(f"cooling(h={h_val}Wm-²K-1)", dpi=300);plt.show()

sens = df.query("I==5 & SOC==0.5").sort_values("h")
sens1 = df.query("I==3 & SOC==0.5").sort_values("h")
sens2 = df.query("I==1 & SOC==0.5").sort_values("h")
sens3 = df.query("I==0.5 & SOC==0.5").sort_values("h")

plt.figure(figsize=(10,6))
plt.plot(sens.h, sens.dT_max, 's-', color='red', label='I = 5 A')
plt.plot(sens1.h, sens1.dT_max, 's-', color='blue', label='I = 3 A')
plt.plot(sens2.h, sens2.dT_max, 's-', color='green', label='I = 1 A')
plt.plot(sens3.h, sens3.dT_max, 's-', color='gray', label='I = 0.5 A')

plt.xscale('log')
plt.xlabel("h_side (W/m²K)")
plt.ylabel("ΔT_max (K)")
#plt.title("Fig.5 Cooling sensitivity (I = 0.5–5 A, SOC = 0.5)")
plt.grid(True, which='both')
plt.legend()  # 加上圖例
plt.tight_layout()
plt.savefig("cooling.png", dpi=300)
plt.show()

sens = df.query("I==5 & SOC==0.9").sort_values("h")
sens1 = df.query("I==5 & SOC==0.7").sort_values("h")
sens2 = df.query("I==5 & SOC==0.5").sort_values("h")
sens3 = df.query("I==5 & SOC==0.3").sort_values("h")
sens4 = df.query("I==5 & SOC==0.1").sort_values("h")
plt.figure(figsize=(10,6))
plt.plot(sens.h, sens.dT_max, 's-', color='red', label='SOC=0.9')
plt.plot(sens1.h, sens1.dT_max, 's-', color='blue', label='SOC=0.7')
plt.plot(sens2.h, sens2.dT_max, 's-', color='green', label='SOC=0.5')
plt.plot(sens3.h, sens3.dT_max, 's-', color='gray', label='SOC=0.3')
plt.plot(sens3.h, sens3.dT_max, 's-', color='black', label='SOC=0.1')
plt.xscale('log')
plt.xlabel("h_side (W/m²K)")
plt.ylabel("ΔT_max (K)")
plt.grid(True, which='both')
plt.legend()  # 加上圖例
plt.tight_layout()
plt.savefig("cooling.png", dpi=300)
plt.show()
print("繪製溫度場 (I=3A, SOC=0.5, h=50)")
I_plot, SOC_plot, h_plot = 3.0, 0.5, 50
q_irrev_plot = I_plot**2 * R_int_SOC(SOC_plot) / V_cell
q_rev_plot   = I_plot * T_env * dUdT_SOC(SOC_plot) / V_cell
q_tot_plot   = q_irrev_plot + q_rev_plot
src_plot     = q_tot_plot / k_rz
T_plot = solve_T(src_plot, k_rz, h_side=h_plot)
plt.figure(figsize=(7, 4.5))
extent = [0, L_cell*1e3, 0, R_cell*1e3] 
im = plt.imshow(T_plot, origin='lower', aspect='auto',
                extent=extent, cmap='hot')
plt.colorbar(im, label="Temperature (K)")
plt.xlabel("Axial position z (mm)")
plt.ylabel("Radial position r (mm)")
plt.title("Temperature field (I=3 A, SOC=0.5, h=50 W/m²K)")
plt.tight_layout()
plt.savefig("Tfield_I3_SOC0.5_h50.png", dpi=300)
plt.show()
