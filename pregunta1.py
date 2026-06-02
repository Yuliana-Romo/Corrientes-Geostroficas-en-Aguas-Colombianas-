"""
=====================================================
  PREGUNTA 1.1 — VERSION 3
  Perfiles T, S, σθ — UNA GRÁFICA POR TRANSECTO
  La Guajira y Pacífico Colombiano | Febrero
  TEOS-2010 (gsw)
=====================================================
Genera:
  Guajira:
    - P1.1_Guajira_Manaure_Temperatura.png
    - P1.1_Guajira_Manaure_Salinidad.png
    - P1.1_Guajira_Riohacha_Temperatura.png
    - P1.1_Guajira_Riohacha_Salinidad.png
    - P1.1_Guajira_Palomino_Temperatura.png
    - P1.1_Guajira_Palomino_Salinidad.png
  Pacifico:
    - P1.1_Pacifico_Cupica_Temperatura.png
    ... etc
  + Densidad potencial σθ por transecto (misma estructura)

Total: 18 figuras
=====================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import gsw

# ─── RUTAS ────────────────────────────────────────
BASE        = r"C:\Users\luisf\Downloads\Trabajo Yuliana"
CARPETA_T   = os.path.join(BASE, "temperatura")
CARPETA_S   = os.path.join(BASE, "sanity")
CARPETA_OUT = os.path.join(BASE, "figuras")
os.makedirs(CARPETA_OUT, exist_ok=True)

PROFUNDIDAD_MAX = 1000  # metros

# ─── TRANSECTOS ───────────────────────────────────
# Cada transecto tiene 2 estaciones: costera y oceánica
# Formato: { "Region": { "Transecto": [(nombre, lat, lon), ...] } }

TRANSECTOS = {
    "Guajira": {
        "Manaure": [
            ("Manaure-Costera",   11.80, -72.60),
            ("Manaure-Oceanica",  12.00, -72.50),
        ],
        "Riohacha": [
            ("Riohacha-Costera",  11.60, -73.00),
            ("Riohacha-Oceanica", 12.00, -73.00),
        ],
        "Palomino": [
            ("Palomino-Costera",  11.40, -73.80),
            ("Palomino-Oceanica", 11.80, -73.80),
        ],
    },
    "Pacifico": {
        "Cupica": [
            ("Cupica-Costera",    6.50, -77.60),
            ("Cupica-Oceanica",   6.50, -77.80),
        ],
        "Nuqui": [
            ("Nuqui-Costera",     5.80, -77.50),
            ("Nuqui-Oceanica",    5.50, -77.80),
        ],
        "Tumaco": [
            ("Tumaco-Costera",    2.00, -79.00),
            ("Tumaco-Oceanica",   1.50, -79.50),
        ],
    },
}

# ─── COLORES por estación dentro del transecto ───
# 2 estaciones por transecto → 2 colores distintos
COLORES    = ["#e63946", "#457b9d"]
MARCADORES = ["o", "s"]

# ─── LEER PERFIL ──────────────────────────────────
def leer_perfil(archivo_t, archivo_s, lat_est, lon_est, prof_max=1000):
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

    # TEOS-2010
    presion = gsw.p_from_z(-dep, lat_est)
    SA      = gsw.SA_from_SP(sal,  presion, lon_est, lat_est)
    CT      = gsw.CT_from_t(SA, temp, presion)
    sigma0  = gsw.sigma0(SA, CT)

    return dep, temp, sal, sigma0


# ─── GRAFICAR UNA VARIABLE PARA UN TRANSECTO ──────
def graficar_transecto(region, transecto, estaciones, archivo_t, archivo_s):
    """Genera 3 figuras por transecto: T, S, σθ"""

    # Leer perfiles
    perfiles = []
    for nombre, lat, lon in estaciones:
        dep, temp, sal, sigma0 = leer_perfil(archivo_t, archivo_s, lat, lon)
        perfiles.append((nombre, dep, temp, sal, sigma0))
        print(f"    {nombre:25s} T={temp[0]:.2f}°C  S={sal[0]:.2f} psu")

    variables = [
        ("Temperatura",        "temp",   "Temperatura (°C)"),
        ("Salinidad",          "sal",    "Salinidad Práctica (psu)"),
        ("Densidad_Potencial", "sigma0", "Densidad Potencial σθ (kg/m³)"),
    ]

    for var_id, var_key, label_x in variables:
        fig, ax = plt.subplots(figsize=(6, 9))

        for i, (nombre, dep, temp, sal, sigma0) in enumerate(perfiles):
            datos = {"temp": temp, "sal": sal, "sigma0": sigma0}[var_key]
            validos = ~np.isnan(datos)
            if validos.sum() == 0:
                print(f"    ⚠ Sin datos: {nombre}")
                continue
            ax.plot(
                datos[validos], dep[validos],
                color=COLORES[i % len(COLORES)],
                marker=MARCADORES[i % len(MARCADORES)],
                markersize=5, linewidth=2.0, markevery=3,
                label=nombre
            )

        ax.invert_yaxis()
        ax.set_ylabel("Profundidad (m)", fontsize=12)
        ax.set_xlabel(label_x, fontsize=12)
        ax.set_title(
            f"{label_x}\nTransecto {transecto} — {region} | Febrero (WOA23, TEOS-2010)",
            fontsize=12, fontweight="bold"
        )
        ax.legend(fontsize=10, framealpha=0.85, loc="lower right")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim([PROFUNDIDAD_MAX, 0])

        # Líneas de referencia
        for pref in [50, 100, 200]:
            ax.axhline(pref, color="gray", linestyle=":", linewidth=0.9, alpha=0.5)

        plt.tight_layout()
        nombre_fig = f"P1.1_{region}_{transecto}_{var_id}_Febrero.png"
        ruta_fig   = os.path.join(CARPETA_OUT, nombre_fig)
        plt.savefig(ruta_fig, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    → Guardada: figuras/{nombre_fig}")


# ─── MAIN ─────────────────────────────────────────
def main():
    print("=" * 58)
    print("  PREGUNTA 1.1 v3 — Una gráfica por transecto")
    print("  Temperatura, Salinidad, σθ | Febrero")
    print("=" * 58)

    archivo_t = os.path.join(CARPETA_T, "woa23_decav_t02_04.nc")
    archivo_s = os.path.join(CARPETA_S, "woa23_decav91C0_s02_04.nc")

    for region, transectos in TRANSECTOS.items():
        print(f"\n{'─'*55}")
        print(f"  REGIÓN: {region}")
        print(f"{'─'*55}")
        for transecto, estaciones in transectos.items():
            print(f"\n  Transecto: {transecto}")
            graficar_transecto(region, transecto, estaciones, archivo_t, archivo_s)

    print(f"\n{'='*58}")
    print(f"  ✓ Listo. Figuras en: {CARPETA_OUT}")
    print(f"\n  Archivos generados:")
    for f in sorted(os.listdir(CARPETA_OUT)):
        if "P1.1" in f:
            print(f"    • {f}")
    print("=" * 58)


if __name__ == "__main__":
    main()