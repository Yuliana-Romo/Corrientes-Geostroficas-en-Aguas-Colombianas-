"""
=====================================================
  PARTE III — INTERPRETACIÓN FÍSICA Y ESTACIONAL
  Pregunta 3.1: La Guajira — marzo vs agosto
  Pregunta 3.2: Pacífico — ENSO, surgencia, batimetría
  Pregunta 3.3: Limitaciones geostrófica cerca ecuador
=====================================================
Uso:
    python pregunta3.py
Requisitos:
    pip install netCDF4 numpy matplotlib gsw
=====================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import netCDF4 as nc
import gsw

# ─── compatibilidad numpy ─────────────────────────
try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

# ─── RUTAS ───────────────────────────────────────
BASE        = r"C:\Users\luisf\Downloads\Trabajo Yuliana"
CARPETA_T   = os.path.join(BASE, "temperatura")
CARPETA_S   = os.path.join(BASE, "sanity")
CARPETA_OUT = os.path.join(BASE, "figuras")
os.makedirs(CARPETA_OUT, exist_ok=True)

# ─── CONSTANTES ──────────────────────────────────
g     = 9.81
OMEGA = 7.2921e-5
rho0  = 1025.0

def coriolis(lat):
    return 2 * OMEGA * np.sin(np.radians(lat))

# ─── ARCHIVOS POR MES ────────────────────────────
ARCH = {
    "t01": "woa23_decav_t01_04.nc",
    "t02": "woa23_decav_t02_04.nc",
    "t03": "woa23_decav_t03_04.nc",
    "t07": "woa23_decav_t07_04.nc",
    "t08": "woa23_decav_t08_04.nc",
    "t10": "woa23_decav_t10_04.nc",
    "s01": "woa23_decav91C0_s01_04.nc",
    "s02": "woa23_decav91C0_s02_04.nc",
    "s03": "woa23_decav91C0_s03_04.nc",
    "s07": "woa23_decav91C0_s07_04.nc",
    "s08": "woa23_decav91C0_s08_04.nc",
    "s10": "woa23_decav91C0_s10_04.nc",
}

# ─── LEER PERFIL ─────────────────────────────────
def leer_perfil(arch_t, arch_s, lat_est, lon_est, prof_max=500):
    ds_t = nc.Dataset(os.path.join(CARPETA_T, arch_t))
    ds_s = nc.Dataset(os.path.join(CARPETA_S, arch_s))

    lat = np.array(ds_t.variables["lat"][:])
    lon = np.array(ds_t.variables["lon"][:])
    dep = np.array(ds_t.variables["depth"][:])

    ilat = int(np.argmin(np.abs(lat - lat_est)))
    ilon = int(np.argmin(np.abs(lon - lon_est)))
    idep = np.where(dep <= prof_max)[0]
    dep  = dep[idep]

    temp = np.array(np.ma.filled(ds_t.variables["t_an"][0, idep, ilat, ilon], np.nan),
                    dtype=np.float64)
    sal  = np.array(np.ma.filled(ds_s.variables["s_an"][0, idep, ilat, ilon], np.nan),
                    dtype=np.float64)
    ds_t.close(); ds_s.close()

    presion = gsw.p_from_z(-dep, lat_est)
    SA = CT = sigma0 = specv = np.full_like(temp, np.nan)
    ok = (~np.isnan(temp)) & (~np.isnan(sal))
    if ok.any():
        SA     = np.full_like(temp, np.nan)
        CT     = np.full_like(temp, np.nan)
        sigma0 = np.full_like(temp, np.nan)
        specv  = np.full_like(temp, np.nan)
        SA[ok]     = gsw.SA_from_SP(sal[ok],  presion[ok], lon_est, lat_est)
        CT[ok]     = gsw.CT_from_t(SA[ok], temp[ok], presion[ok])
        sigma0[ok] = gsw.sigma0(SA[ok], CT[ok])
        specv[ok]  = gsw.specvol(SA[ok], CT[ok], presion[ok])

    return dep, temp, sal, sigma0, presion, specv


def calcular_vg_par(specv_c, specv_o, P_c, P_o, lat_c, lon_c, lat_o, lon_o):
    """
    Velocidad geostrófica superficial entre dos estaciones
    usando anomalía de altura dinámica con nivel de referencia común.
    """
    alpha_0 = 1.0 / rho0
    n = min(len(specv_c), len(specv_o))
    P = P_c[:n]
    d_c = specv_c[:n] - alpha_0
    d_o = specv_o[:n] - alpha_0
    ok  = (~np.isnan(d_c)) & (~np.isnan(d_o))
    if ok.sum() < 2:
        return np.nan, np.nan

    idx = np.where(ok)[0]
    D_c = np.full(n, np.nan); D_o = np.full(n, np.nan)
    D_c[idx[-1]] = D_o[idx[-1]] = 0.0
    for k in range(len(idx)-2, -1, -1):
        i, i1 = idx[k], idx[k+1]
        dp = (P[i1]-P[i]) * 1e4
        D_c[i] = D_c[i1] + 0.5*(d_c[i]+d_c[i1])*dp
        D_o[i] = D_o[i1] + 0.5*(d_o[i]+d_o[i1])*dp

    dlat   = abs(lat_o-lat_c)*111000
    dlon   = abs(lon_o-lon_c)*111000*np.cos(np.radians((lat_c+lat_o)/2))
    dist_m = np.sqrt(dlat**2+dlon**2)
    f      = coriolis((lat_c+lat_o)/2)
    i0     = idx[0]
    vg     = (D_o[i0]-D_c[i0])/(f*dist_m)
    return vg, dist_m


# ══════════════════════════════════════════════════
#  PREGUNTA 3.1 — La Guajira | Marzo vs Agosto
# ══════════════════════════════════════════════════

EST_GUAJIRA = {
    "Manaure":  [(11.80, -72.60), (12.00, -72.50)],
    "Riohacha": [(11.60, -73.00), (12.00, -73.00)],
    "Palomino": [(11.40, -73.80), (11.80, -73.80)],
}

def pregunta_3_1():
    print("\n" + "═"*62)
    print("  PREGUNTA 3.1 — La Guajira | Marzo (afloramiento) vs")
    print("                              Agosto (relajación)")
    print("═"*62)

    meses = {
        "Marzo\n(afloramiento)":  (ARCH["t03"], ARCH["s03"]),
        "Agosto\n(relajación)":   (ARCH["t08"], ARCH["s08"]),
    }

    resultados = {t: {} for t in EST_GUAJIRA}
    tabla = []

    for mes_label, (arch_t, arch_s) in meses.items():
        for transecto, (est_c, est_o) in EST_GUAJIRA.items():
            lat_c, lon_c = est_c
            lat_o, lon_o = est_o
            dep_c, T_c, S_c, sig_c, P_c, specv_c = leer_perfil(arch_t, arch_s, lat_c, lon_c)
            dep_o, T_o, S_o, sig_o, P_o, specv_o = leer_perfil(arch_t, arch_s, lat_o, lon_o)
            vg, dist_m = calcular_vg_par(specv_c, specv_o, P_c, P_o,
                                          lat_c, lon_c, lat_o, lon_o)
            resultados[transecto][mes_label] = {
                "vg": vg,
                "T_sup_c": T_c[0] if not np.isnan(T_c[0]) else np.nan,
                "S_sup_c": S_c[0] if not np.isnan(S_c[0]) else np.nan,
                "sig_sup_c": sig_c[0] if not np.isnan(sig_c[0]) else np.nan,
                "dep_c": dep_c, "T_c": T_c, "S_c": S_c, "sig_c": sig_c,
                "dep_o": dep_o, "T_o": T_o, "S_o": S_o, "sig_o": sig_o,
            }
            vg_str = f"{vg*100:.2f}" if not np.isnan(vg) else "N/A"
            tabla.append((transecto, mes_label.replace("\n"," "), vg_str))

    print(f"\n  {'Transecto':12s} {'Mes':22s} {'v_g (cm/s)':12s}")
    print(f"  {'-'*48}")
    for row in tabla:
        print(f"  {row[0]:12s} {row[1]:22s} {row[2]:12s}")

    # ── Figura: Comparación estacional T y σθ ─────
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    col_mar = "#e63946"
    col_ago = "#457b9d"
    transectos = list(EST_GUAJIRA.keys())

    for col, transecto in enumerate(transectos):
        # Panel superior: Temperatura
        ax_T = fig.add_subplot(gs[0, col])
        for mes_label, color in zip(meses.keys(), [col_mar, col_ago]):
            r  = resultados[transecto][mes_label]
            ok = ~np.isnan(r["T_c"])
            if ok.any():
                ax_T.plot(r["T_c"][ok], r["dep_c"][ok],
                          color=color, linewidth=2,
                          label=mes_label.replace("\n"," "))
        ax_T.invert_yaxis()
        ax_T.set_title(f"{transecto}", fontsize=11, fontweight="bold")
        ax_T.set_xlabel("Temperatura (°C)", fontsize=9)
        ax_T.set_ylabel("Prof (m)", fontsize=9)
        ax_T.legend(fontsize=7)
        ax_T.grid(True, linestyle="--", alpha=0.4)
        ax_T.set_ylim([300, 0])

        # Panel inferior: Densidad potencial
        ax_D = fig.add_subplot(gs[1, col])
        for mes_label, color in zip(meses.keys(), [col_mar, col_ago]):
            r  = resultados[transecto][mes_label]
            ok = ~np.isnan(r["sig_c"])
            if ok.any():
                ax_D.plot(r["sig_c"][ok], r["dep_c"][ok],
                          color=color, linewidth=2,
                          label=mes_label.replace("\n"," "))
        ax_D.invert_yaxis()
        ax_D.set_xlabel("σθ (kg/m³)", fontsize=9)
        ax_D.set_ylabel("Prof (m)", fontsize=9)
        ax_D.legend(fontsize=7)
        ax_D.grid(True, linestyle="--", alpha=0.4)
        ax_D.set_ylim([300, 0])

    fig.suptitle(
        "Variabilidad Estacional — La Guajira\n"
        "Marzo (afloramiento) vs Agosto (relajación) | WOA23",
        fontsize=13, fontweight="bold"
    )
    plt.savefig(os.path.join(CARPETA_OUT, "P3.1_Guajira_Estacional_T_sigma.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  → Guardada: P3.1_Guajira_Estacional_T_sigma.png")

    # ── Figura: velocidades geostróficas comparadas ─
    fig, ax = plt.subplots(figsize=(9, 5))
    x      = np.arange(len(transectos))
    width  = 0.35
    vg_mar = [resultados[t]["Marzo\n(afloramiento)"]["vg"]*100
               if not np.isnan(resultados[t]["Marzo\n(afloramiento)"]["vg"]) else 0
               for t in transectos]
    vg_ago = [resultados[t]["Agosto\n(relajación)"]["vg"]*100
               if not np.isnan(resultados[t]["Agosto\n(relajación)"]["vg"]) else 0
               for t in transectos]

    b1 = ax.bar(x - width/2, vg_mar, width, label="Marzo (afloramiento)",
                color=col_mar, edgecolor="black", linewidth=0.8)
    b2 = ax.bar(x + width/2, vg_ago, width, label="Agosto (relajación)",
                color=col_ago, edgecolor="black", linewidth=0.8)

    for bar in list(b1)+list(b2):
        v = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2,
                v + (0.3 if v >= 0 else -0.8),
                f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(transectos, fontsize=11)
    ax.set_ylabel("v_g superficial (cm/s)", fontsize=11)
    ax.set_title("Velocidad Geostrófica Superficial — La Guajira\n"
                 "Marzo vs Agosto", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_OUT, "P3.1_Guajira_Vg_MarzoVsAgosto.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Guardada: P3.1_Guajira_Vg_MarzoVsAgosto.png")


# ══════════════════════════════════════════════════
#  PREGUNTA 3.2 — Pacífico | ENSO, surgencia, batimetría
# ══════════════════════════════════════════════════

EST_PACIFICO = {
    "Cupica": [(6.50, -77.60), (6.50, -77.80)],
    "Nuqui":  [(5.80, -77.50), (5.50, -77.80)],
    "Tumaco": [(2.00, -79.00), (1.50, -79.50)],
}

def pregunta_3_2():
    print("\n" + "═"*62)
    print("  PREGUNTA 3.2 — Pacífico | ENSO + Surgencia + Batimetría")
    print("═"*62)

    meses_pac = {
        "Enero\n(Niña/seco)":   (ARCH["t01"], ARCH["s01"]),
        "Julio\n(neutro)":      (ARCH["t07"], ARCH["s07"]),
        "Octubre\n(Niño/lluvioso)": (ARCH["t10"], ARCH["s10"]),
    }

    colores_mes = ["#e63946", "#2a9d8f", "#e9c46a"]
    resultados_pac = {t: {} for t in EST_PACIFICO}

    print(f"\n  {'Transecto':10s} {'Mes':22s} {'v_g (cm/s)':12s} {'T_sup (°C)':12s} {'S_sup (psu)':12s}")
    print(f"  {'-'*70}")

    for mes_label, (arch_t, arch_s) in meses_pac.items():
        for transecto, (est_c, est_o) in EST_PACIFICO.items():
            lat_c, lon_c = est_c
            lat_o, lon_o = est_o
            dep_c, T_c, S_c, sig_c, P_c, specv_c = leer_perfil(arch_t, arch_s, lat_c, lon_c)
            dep_o, T_o, S_o, sig_o, P_o, specv_o = leer_perfil(arch_t, arch_s, lat_o, lon_o)
            vg, _ = calcular_vg_par(specv_c, specv_o, P_c, P_o,
                                     lat_c, lon_c, lat_o, lon_o)
            resultados_pac[transecto][mes_label] = {
                "vg": vg,
                "T_sup": T_c[0], "S_sup": S_c[0],
                "dep_c": dep_c, "T_c": T_c, "S_c": S_c, "sig_c": sig_c,
            }
            vg_str = f"{vg*100:.2f}" if not np.isnan(vg) else "N/A"
            mes_s  = mes_label.replace("\n"," ")
            print(f"  {transecto:10s} {mes_s:22s} {vg_str:12s} "
                  f"{T_c[0]:.2f}{' ':8s} {S_c[0]:.2f}")

    # ── Figura: T superficial por mes y transecto ─
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    transectos_pac = list(EST_PACIFICO.keys())

    for col, transecto in enumerate(transectos_pac):
        ax = axes[col]
        for mes_label, color in zip(meses_pac.keys(), colores_mes):
            r  = resultados_pac[transecto][mes_label]
            ok = ~np.isnan(r["T_c"])
            if ok.any():
                ax.plot(r["T_c"][ok], r["dep_c"][ok],
                        color=color, linewidth=2,
                        label=mes_label.replace("\n"," "))
        ax.invert_yaxis()
        ax.set_title(f"Transecto {transecto}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Temperatura (°C)", fontsize=10)
        ax.set_ylabel("Prof (m)", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim([300, 0])

    fig.suptitle(
        "Pacífico Colombiano — Variabilidad Estacional / ENSO\n"
        "Perfiles de Temperatura: Enero vs Julio vs Octubre",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_OUT, "P3.2_Pacifico_ENSO_PerfilesT.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  → Guardada: P3.2_Pacifico_ENSO_PerfilesT.png")

    # ── Figura: velocidades geostróficas 3 meses ─
    fig, ax = plt.subplots(figsize=(10, 5))
    x     = np.arange(len(transectos_pac))
    width = 0.25
    for j, (mes_label, color) in enumerate(zip(meses_pac.keys(), colores_mes)):
        vgs = []
        for t in transectos_pac:
            v = resultados_pac[t][mes_label]["vg"]
            vgs.append(v*100 if not np.isnan(v) else 0.0)
        bars = ax.bar(x + (j-1)*width, vgs, width,
                      label=mes_label.replace("\n"," "),
                      color=color, edgecolor="black", linewidth=0.8)
        for bar, v in zip(bars, vgs):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height() + (0.2 if bar.get_height() >= 0 else -0.6),
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(transectos_pac, fontsize=11)
    ax.set_ylabel("v_g superficial (cm/s)", fontsize=11)
    ax.set_title("Velocidad Geostrófica Superficial — Pacífico Colombiano\n"
                 "Variabilidad Estacional / ENSO", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_OUT, "P3.2_Pacifico_Vg_ENSO.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Guardada: P3.2_Pacifico_Vg_ENSO.png")


# ══════════════════════════════════════════════════
#  PREGUNTA 3.3 — Limitaciones near-ecuador
# ══════════════════════════════════════════════════

def pregunta_3_3():
    print("\n" + "═"*62)
    print("  PREGUNTA 3.3 — Limitaciones Geostrófica | Near-Ecuador")
    print("═"*62)

    # Parámetros representativos de cada transecto
    estaciones_3_3 = [
        ("Cupica",  6.5,  22e3, 0.05),   # lat, L (m), U (m/s)
        ("Nuqui",   5.65, 47e3, 0.03),
        ("Tumaco",  1.75, 78e3, 0.14),   # U estimada de v_g calculada
    ]

    print(f"\n  {'Transecto':12s} {'Lat (°N)':10s} {'f (s⁻¹)':14s} "
          f"{'Ro = U/fL':12s} {'Válido?':10s}")
    print(f"  {'-'*62}")

    resultados_ro = []
    for nombre, lat, L, U in estaciones_3_3:
        f  = coriolis(lat)
        Ro = U / (abs(f) * L)
        valido = "SÍ (Ro<0.1)" if Ro < 0.1 else ("LÍMITE" if Ro < 0.5 else "NO (Ro>0.5)")
        print(f"  {nombre:12s} {lat:10.2f} {f:14.2e} {Ro:12.4f} {valido:10s}")
        resultados_ro.append((nombre, lat, f, Ro, valido))

    # ── Figura: Número de Rossby por transecto ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    nombres = [r[0] for r in resultados_ro]
    lats    = [r[1] for r in resultados_ro]
    f_vals  = [abs(r[2]) for r in resultados_ro]
    Ro_vals = [r[3] for r in resultados_ro]

    # Panel 1: f vs latitud
    lat_cont = np.linspace(0, 10, 200)
    f_cont   = 2*OMEGA*np.sin(np.radians(lat_cont))
    axes[0].plot(lat_cont, f_cont*1e5, color="navy", linewidth=2, label="f = 2Ω sin(φ)")
    for nombre, lat, f, Ro, valido in resultados_ro:
        color = "#e63946" if "NO" in valido else ("#f4a261" if "LÍMITE" in valido else "#2a9d8f")
        axes[0].scatter(lat, abs(f)*1e5, s=120, color=color, zorder=5)
        axes[0].annotate(nombre, (lat, abs(f)*1e5),
                         textcoords="offset points", xytext=(5, 5), fontsize=9)
    axes[0].axhline(0, color="black", linewidth=0.5)
    axes[0].set_xlabel("Latitud (°N)", fontsize=11)
    axes[0].set_ylabel("f (×10⁻⁵ s⁻¹)", fontsize=11)
    axes[0].set_title("Parámetro de Coriolis\nvs Latitud", fontsize=11, fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].legend(fontsize=9)

    # Panel 2: Número de Rossby
    colores_ro = []
    for r in resultados_ro:
        if "NO" in r[4]:   colores_ro.append("#e63946")
        elif "LÍMITE" in r[4]: colores_ro.append("#f4a261")
        else:              colores_ro.append("#2a9d8f")

    bars = axes[1].bar(nombres, Ro_vals, color=colores_ro,
                       edgecolor="black", linewidth=0.8)
    axes[1].axhline(0.1, color="#e63946", linestyle="--", linewidth=1.5,
                    label="Ro = 0.1 (límite geostrófico)")
    axes[1].axhline(0.5, color="darkred",  linestyle="--", linewidth=1.5,
                    label="Ro = 0.5 (no geostrófico)")
    for bar, v in zip(bars, Ro_vals):
        axes[1].text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.005,
                     f"{v:.3f}", ha="center", va="bottom",
                     fontsize=10, fontweight="bold")
    axes[1].set_ylabel("Número de Rossby (Ro)", fontsize=11)
    axes[1].set_title("Validez de la Aproximación Geostrófica\n"
                      "Ro = U / (f · L)", fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.4)
    axes[1].set_ylim([0, max(Ro_vals)*1.4])

    # Añadir etiqueta de validez
    for i, r in enumerate(resultados_ro):
        axes[1].text(i, 0.005, r[4], ha="center", va="bottom",
                     fontsize=8, style="italic", color="black")

    fig.suptitle(
        "Limitaciones del Balance Geostrófico cerca del Ecuador\n"
        "Pacífico Colombiano — Número de Rossby por Transecto",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_OUT, "P3.3_Pacifico_NumeroRossby.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  → Guardada: P3.3_Pacifico_NumeroRossby.png")

    # ── Tabla resumen zona ecuatorial ────────────
    print(f"\n  Análisis ecuatorial adicional (Tumaco, lat~1.75°N):")
    lat_t = 1.75
    f_t   = coriolis(lat_t)
    U_t   = 0.14
    L_t   = 78e3
    Ro_t  = U_t / (abs(f_t) * L_t)
    Ld    = np.sqrt(g * 100) / abs(f_t) / 1000  # radio de Rossby (km), H=100m
    print(f"    f = {f_t:.2e} s⁻¹  →  radio de deformación de Rossby = {Ld:.0f} km")
    print(f"    Ro = {Ro_t:.3f}  →  {"la aproximación geostrófica NO es estrictamente válida" if Ro_t > 0.1 else "aproximación válida"}")
    print(f"    A lat < 3°N se recomienda usar ecuaciones primitivas completas")


# ─── MAIN ─────────────────────────────────────────
def main():
    print("=" * 62)
    print("  PARTE III — INTERPRETACIÓN FÍSICA Y ESTACIONAL")
    print("=" * 62)
    pregunta_3_1()
    pregunta_3_2()
    pregunta_3_3()
    print(f"\n{'='*62}")
    figuras = [f for f in sorted(os.listdir(CARPETA_OUT)) if f.startswith("P3")]
    print(f"  Figuras Parte III ({len(figuras)}):")
    for f in figuras:
        print(f"    • {f}")
    print("=" * 62)

if __name__ == "__main__":
    main()