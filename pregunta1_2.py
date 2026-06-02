"""
=====================================================
  PREGUNTA 1.2 — Golfo de Salamanca | Octubre
  a) Estructura vertical cuña salina (haloclina)
  b) Profundidad capa de mezcla
  c) Gradiente horizontal de densidad (primeros 100m)
=====================================================
Uso:
    python pregunta1_2.py

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

# ─── RUTAS ────────────────────────────────────────
BASE        = r"C:\Users\luisf\Downloads\Trabajo Yuliana"
CARPETA_T   = os.path.join(BASE, "temperatura")
CARPETA_S   = os.path.join(BASE, "sanity")
CARPETA_OUT = os.path.join(BASE, "figuras")
os.makedirs(CARPETA_OUT, exist_ok=True)

PROFUNDIDAD_MAX = 300  # Para el Golfo con 100m de análisis, 300m es suficiente

# ─── ESTACIONES GOLFO DE SALAMANCA ───────────────
# Tres transectos perpendiculares a la costa
# 2 estaciones por transecto (costera y oceánica)
ESTACIONES = [
    ("Cienaga-C",     11.10, -74.30),
    ("Cienaga-O",     11.30, -74.50),
    ("Barravieja-C",  11.20, -74.40),
    ("Barravieja-O",  11.40, -74.60),
    ("Km19-C",        11.30, -74.50),
    ("Km19-O",        11.50, -74.70),
]

COLORES    = ["#e63946", "#e63946", "#2a9d8f", "#2a9d8f", "#e9c46a", "#e9c46a"]
ESTILOS    = ["-",       "--",      "-",       "--",      "-",       "--"      ]
MARCADORES = ["o",       "s",       "o",       "s",       "o",       "s"       ]

# ─── LEER PERFIL ──────────────────────────────────
def leer_perfil(archivo_t, archivo_s, lat_est, lon_est, prof_max=300):
    ds_t = nc.Dataset(archivo_t)
    ds_s = nc.Dataset(archivo_s)

    lat = ds_t.variables["lat"][:]
    lon = ds_t.variables["lon"][:]
    dep = ds_t.variables["depth"][:]

    ilat = int(np.argmin(np.abs(lat - lat_est)))
    ilon = int(np.argmin(np.abs(lon - lon_est)))

    idep = np.where(dep <= prof_max)[0]
    dep  = dep[idep]

    temp = np.ma.filled(ds_t.variables["t_an"][0, idep, ilat, ilon], np.nan)
    sal  = np.ma.filled(ds_s.variables["s_an"][0, idep, ilat, ilon], np.nan)

    ds_t.close()
    ds_s.close()

    presion = gsw.p_from_z(-dep, lat_est)
    SA      = gsw.SA_from_SP(sal,  presion, lon_est, lat_est)
    CT      = gsw.CT_from_t(SA, temp, presion)
    sigma0  = gsw.sigma0(SA, CT)

    return dep, temp, sal, sigma0


# ─── b) PROFUNDIDAD CAPA DE MEZCLA ────────────────
def calcular_capa_mezcla(dep, temp, sal, criterio_delta_T=0.2):
    """
    Método de umbral de temperatura:
    MLD = profundidad donde T cae más de 0.2°C respecto a T superficial.
    Criterio estándar de de Boyer Montégut et al. (2004).
    """
    T_sup = temp[0]
    for i in range(1, len(dep)):
        if not np.isnan(temp[i]):
            if abs(temp[i] - T_sup) >= criterio_delta_T:
                # Interpolar linealmente entre niveles
                dT = abs(temp[i] - temp[i-1])
                if dT > 0:
                    fraccion = (criterio_delta_T - abs(temp[i-1] - T_sup)) / dT
                    mld = dep[i-1] + fraccion * (dep[i] - dep[i-1])
                else:
                    mld = dep[i]
                return float(mld)
    return np.nan


# ─── a) DETECTAR HALOCLINA ────────────────────────
def detectar_haloclina(dep, sal):
    """
    Encuentra la profundidad de máximo gradiente de salinidad (haloclina).
    Devuelve: profundidad de la haloclina y el gradiente máximo (psu/m)
    """
    gradientes = []
    for i in range(len(dep) - 1):
        if not np.isnan(sal[i]) and not np.isnan(sal[i+1]):
            dS = sal[i+1] - sal[i]
            dz = dep[i+1]  - dep[i]
            gradientes.append((dep[i], dS / dz))
        else:
            gradientes.append((dep[i], np.nan))

    grad_vals = [g[1] for g in gradientes if not np.isnan(g[1])]
    if not grad_vals:
        return np.nan, np.nan

    idx_max = np.nanargmax(np.abs(grad_vals))
    prof_haloclina = gradientes[idx_max][0]
    grad_max       = gradientes[idx_max][1]
    return float(prof_haloclina), float(grad_max)


# ─── c) GRADIENTE HORIZONTAL DE DENSIDAD ─────────
def gradiente_horizontal_densidad(perfiles, prof_max=100):
    """
    Calcula el gradiente horizontal de densidad promedio en los primeros 100 m
    entre estaciones costera y oceánica de cada transecto.
    Δσθ / Δx  [kg/m³ / km]
    """
    # Transectos: pares (costera, oceánica)
    pares = [
        ("Ciénaga",     perfiles[0], perfiles[1], 30.0),   # distancia aprox en km
        ("Barravieja",  perfiles[2], perfiles[3], 35.0),
        ("Km19",        perfiles[4], perfiles[5], 32.0),
    ]

    resultados = []
    for nombre, p_cost, p_ocea, dist_km in pares:
        # Desempaquetar correctamente (nombre, dep, temp, sal, sigma0)
        _, dep_c, _, _, sig_c = p_cost  # Ignoramos nombre, temp, sal
        _, dep_o, _, _, sig_o = p_ocea  # Ignoramos nombre, temp, sal
        
        # Resto del código igual...
        idx_c = np.where((dep_c <= prof_max) & ~np.isnan(sig_c))[0]
        idx_o = np.where((dep_o <= prof_max) & ~np.isnan(sig_o))[0]

        if len(idx_c) == 0 or len(idx_o) == 0:
            resultados.append((nombre, np.nan, np.nan, np.nan, dist_km))
            continue

        sig_media_c = np.nanmean(sig_c[idx_c])
        sig_media_o = np.nanmean(sig_o[idx_o])

        grad = (sig_media_o - sig_media_c) / dist_km  # kg/m³/km
        resultados.append((nombre, grad, sig_media_c, sig_media_o, dist_km))

    return resultados


# ─── FIGURA 1: Perfiles T y S con haloclina y MLD ─
def figura_perfiles(perfiles, archivo_salida):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 9), sharey=True)

    tabla_mld      = []
    tabla_haloclina = []

    for i, (nombre, dep, temp, sal, sigma0) in enumerate(perfiles):
        validos = ~np.isnan(temp)
        if validos.sum() == 0:
            print(f"  ⚠ Sin datos: {nombre}")
            continue

        # Temperatura
        ax1.plot(temp[validos], dep[validos],
                 color=COLORES[i], linestyle=ESTILOS[i],
                 marker=MARCADORES[i], markersize=4, linewidth=1.8,
                 markevery=3, label=nombre)

        # Salinidad
        ax2.plot(sal[validos], dep[validos],
                 color=COLORES[i], linestyle=ESTILOS[i],
                 marker=MARCADORES[i], markersize=4, linewidth=1.8,
                 markevery=3, label=nombre)

        # Calcular MLD y haloclina
        mld                     = calcular_capa_mezcla(dep, temp, sal)
        prof_halo, grad_halo    = detectar_haloclina(dep, sal)
        tabla_mld.append((nombre, mld))
        tabla_haloclina.append((nombre, prof_halo, grad_halo))

        # Marcar MLD en el perfil de T
        if not np.isnan(mld):
            ax1.axhline(mld, color=COLORES[i], linestyle=":", linewidth=1.0, alpha=0.7)

        # Marcar haloclina en el perfil de S
        if not np.isnan(prof_halo):
            ax2.axhline(prof_halo, color=COLORES[i], linestyle=":", linewidth=1.0, alpha=0.7)

    for ax in [ax1, ax2]:
        ax.invert_yaxis()
        ax.set_ylabel("Profundidad (m)", fontsize=12)
        ax.set_ylim([PROFUNDIDAD_MAX, 0])
        ax.legend(fontsize=8, framealpha=0.85)
        ax.grid(True, linestyle="--", alpha=0.4)
        for pref in [20, 50, 100]:
            ax.axhline(pref, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)

    ax1.set_xlabel("Temperatura (°C)", fontsize=12)
    ax2.set_xlabel("Salinidad Práctica (psu)", fontsize=12)

    fig.suptitle(
        "Golfo de Salamanca — Octubre (WOA23, TEOS-2010)\n"
        "Perfiles Temperatura y Salinidad | Líneas punteadas: MLD (T) y Haloclina (S)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(archivo_salida, dpi=150, bbox_inches="tight")
    plt.close()

    return tabla_mld, tabla_haloclina


# ─── FIGURA 2: Densidad potencial σθ ──────────────
def figura_densidad(perfiles, archivo_salida):
    fig, ax = plt.subplots(figsize=(7, 9))

    for i, (nombre, dep, temp, sal, sigma0) in enumerate(perfiles):
        validos = ~np.isnan(sigma0)
        if validos.sum() == 0:
            continue
        ax.plot(sigma0[validos], dep[validos],
                color=COLORES[i], linestyle=ESTILOS[i],
                marker=MARCADORES[i], markersize=4, linewidth=1.8,
                markevery=3, label=nombre)

    ax.invert_yaxis()
    ax.set_ylabel("Profundidad (m)", fontsize=12)
    ax.set_xlabel("Densidad Potencial σθ (kg/m³)", fontsize=12)
    ax.set_ylim([PROFUNDIDAD_MAX, 0])
    ax.set_title(
        "Densidad Potencial σθ\nGolfo de Salamanca — Octubre (WOA23, TEOS-2010)",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9, framealpha=0.85)
    ax.grid(True, linestyle="--", alpha=0.4)
    for pref in [20, 50, 100]:
        ax.axhline(pref, color="gray", linestyle=":", linewidth=0.8, alpha=0.4)

    plt.tight_layout()
    plt.savefig(archivo_salida, dpi=150, bbox_inches="tight")
    plt.close()


# ─── FIGURA 3: Gradiente horizontal de densidad ───
def figura_gradiente(grad_resultados, archivo_salida):
    nombres = [r[0] for r in grad_resultados]
    grads   = [r[1] for r in grad_resultados]
    sig_c   = [r[2] for r in grad_resultados]
    sig_o   = [r[3] for r in grad_resultados]
    dists   = [r[4] for r in grad_resultados]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(nombres))
    bars = ax.bar(x, grads, color=["#e63946", "#2a9d8f", "#e9c46a"],
                  edgecolor="black", linewidth=0.8, width=0.5)

    for bar, g, sc, so, d in zip(bars, grads, sig_c, sig_o, dists):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.0002,
                f"{g:.4f}\nkg/m³/km", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(nombres, fontsize=11)
    ax.set_ylabel("Gradiente horizontal σθ (kg/m³/km)", fontsize=11)
    ax.set_title(
        "Gradiente Horizontal de Densidad Potencial (0–100 m)\n"
        "Golfo de Salamanca — Octubre",
        fontsize=12, fontweight="bold"
    )
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.axhline(0, color="black", linewidth=0.8)

    plt.tight_layout()
    plt.savefig(archivo_salida, dpi=150, bbox_inches="tight")
    plt.close()


# ─── MAIN ─────────────────────────────────────────
def main():
    print("=" * 58)
    print("  PREGUNTA 1.2 — Golfo de Salamanca | Octubre")
    print("=" * 58)

    archivo_t = os.path.join(CARPETA_T, "woa23_decav_t10_04.nc")
    archivo_s = os.path.join(CARPETA_S, "woa23_decav91C0_s10_04.nc")

    # Leer todos los perfiles
    print("\n  Leyendo perfiles...")
    perfiles = []
    for nombre, lat, lon in ESTACIONES:
        dep, temp, sal, sigma0 = leer_perfil(archivo_t, archivo_s, lat, lon)
        n_val = np.sum(~np.isnan(temp))
        print(f"    {nombre:20s} T={temp[0]:.2f}°C  S={sal[0]:.2f} psu  niveles={n_val}")
        perfiles.append((nombre, dep, temp, sal, sigma0))

    # ── Figura 1: T y S con MLD y haloclina
    print("\n  Generando figura de perfiles T y S...")
    f1 = os.path.join(CARPETA_OUT, "P1.2_GolfoSalamanca_PerfilesTS_Octubre.png")
    tabla_mld, tabla_halo = figura_perfiles(perfiles, f1)
    print(f"  → Guardada: figuras/P1.2_GolfoSalamanca_PerfilesTS_Octubre.png")

    # ── Figura 2: Densidad potencial
    print("\n  Generando figura de densidad potencial...")
    f2 = os.path.join(CARPETA_OUT, "P1.2_GolfoSalamanca_Densidad_Octubre.png")
    figura_densidad(perfiles, f2)
    print(f"  → Guardada: figuras/P1.2_GolfoSalamanca_Densidad_Octubre.png")

    # ── c) Gradiente horizontal de densidad
    print("\n  Calculando gradientes horizontales de densidad (0-100m)...")
    grad_resultados = gradiente_horizontal_densidad(perfiles)
    f3 = os.path.join(CARPETA_OUT, "P1.2_GolfoSalamanca_GradienteDensidad_Octubre.png")
    figura_gradiente(grad_resultados, f3)
    print(f"  → Guardada: figuras/P1.2_GolfoSalamanca_GradienteDensidad_Octubre.png")

    # ── Resumen de resultados numéricos
    print("\n" + "═" * 58)
    print("  RESULTADOS NUMÉRICOS")
    print("═" * 58)

    print("\n  a) HALOCLINA (máximo gradiente de salinidad):")
    print(f"  {'Estación':22s} {'Prof. haloclina (m)':20s} {'Grad. (psu/m)':15s}")
    print(f"  {'-'*55}")
    for nombre, prof, grad in tabla_halo:
        print(f"  {nombre:22s} {prof:20.1f} {grad:15.4f}")

    print("\n  b) PROFUNDIDAD CAPA DE MEZCLA (criterio ΔT=0.2°C):")
    print(f"  {'Estación':22s} {'MLD (m)':10s}")
    print(f"  {'-'*35}")
    for nombre, mld in tabla_mld:
        print(f"  {nombre:22s} {mld:10.1f}")

    print("\n  c) GRADIENTE HORIZONTAL DE DENSIDAD (0-100m):")
    print(f"  {'Transecto':15s} {'σθ costera':12s} {'σθ oceánica':12s} {'Dist(km)':10s} {'Grad(kg/m³/km)':15s}")
    print(f"  {'-'*65}")
    for r in grad_resultados:
        print(f"  {r[0]:15s} {r[2]:12.4f} {r[3]:12.4f} {r[4]:10.1f} {r[1]:15.6f}")

    print("\n" + "=" * 58)
    print("  ✓ Pregunta 1.2 completa.")
    print("=" * 58)


if __name__ == "__main__":
    main()