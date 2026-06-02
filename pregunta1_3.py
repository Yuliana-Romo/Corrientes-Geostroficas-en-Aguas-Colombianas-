"""
=====================================================
  PREGUNTA 1.3 — Diagramas T-S
  6 Regiones (Caribe + Pacífico)
  Meses: Enero, Julio, Octubre
  Identificación de masas de agua
=====================================================
Uso:
    python pregunta1_3.py

Requisitos:
    pip install netCDF4 numpy matplotlib gsw
=====================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import netCDF4 as nc
import gsw

# ─── RUTAS ────────────────────────────────────────
BASE        = r"C:\Users\luisf\Downloads\Trabajo Yuliana"
CARPETA_T   = os.path.join(BASE, "temperatura")
CARPETA_S   = os.path.join(BASE, "sanity")
CARPETA_OUT = os.path.join(BASE, "figuras")
os.makedirs(CARPETA_OUT, exist_ok=True)

# ─── ARCHIVOS POR MES ─────────────────────────────
ARCHIVOS = {
    "Enero":   ("woa23_decav_t01_04.nc", "woa23_decav91C0_s01_04.nc"),
    "Julio":   ("woa23_decav_t07_04.nc", "woa23_decav91C0_s07_04.nc"),
    "Octubre": ("woa23_decav_t10_04.nc", "woa23_decav91C0_s10_04.nc"),
}

# ─── REGIONES: una estación representativa por región
# Se usa la estación oceánica de cada transecto (más datos válidos)
REGIONES = {
    # ── CARIBE ──────────────────────────────────────
    "Manaure\n(Guajira)",    (12.00, -72.50),
    "Riohacha\n(Guajira)",   (12.00, -73.00),
    "Palomino\n(Guajira)",   (11.80, -73.80),
    # ── GOLFO DE SALAMANCA ───────────────────────────
    "Ciénaga\n(Salamanca)",  (11.30, -74.50),
    # ── PACÍFICO ─────────────────────────────────────
    "Nuquí\n(Pacífico)",     (5.50,  -77.80),
    "Tumaco\n(Pacífico)",    (1.50,  -79.50),
}

# Redefinir como diccionario correctamente
REGIONES = {
    "Manaure\n(Guajira)":   (12.00, -72.50),
    "Riohacha\n(Guajira)":  (12.00, -73.00),
    "Palomino\n(Guajira)":  (11.80, -73.80),
    "Ciénaga\n(Salamanca)": (11.30, -74.50),
    "Nuquí\n(Pacífico)":    (5.50,  -77.80),
    "Tumaco\n(Pacífico)":   (1.50,  -79.50),
}

PROF_MAX = 1000  # metros

# ─── COLORES por región ───────────────────────────
COLORES_REGION = {
    "Manaure\n(Guajira)":   "#e63946",
    "Riohacha\n(Guajira)":  "#ff6b6b",
    "Palomino\n(Guajira)":  "#c1121f",
    "Ciénaga\n(Salamanca)": "#f4a261",
    "Nuquí\n(Pacífico)":    "#2a9d8f",
    "Tumaco\n(Pacífico)":   "#457b9d",
}

# ─── MARCADORES por mes ───────────────────────────
MARCADORES_MES = {"Enero": "o", "Julio": "s", "Octubre": "^"}
ALPHA_MES      = {"Enero": 0.9, "Julio": 0.7, "Octubre": 0.5}

# ─── MASAS DE AGUA conocidas (T, S aproximados) ──
# Para dibujar etiquetas de referencia en el diagrama
MASAS_AGUA = [
    # (nombre, S_psu, T_celsius)
    ("ACT\nAgua Caribe\nTropical",      36.5, 27.0),
    ("AST\nAgua Subtropical",           37.0, 22.0),
    ("ACISN\nAgua Intermedia\nSubárt.", 34.8,  5.0),
    ("APS\nAgua Pac. Superficial",      32.0, 26.5),
    ("APSS\nAgua Pac. Subsup.",         34.5, 13.0),
]


# ─── LEER PERFIL ──────────────────────────────────
def leer_perfil_ts(archivo_t, archivo_s, lat_est, lon_est, prof_max=1000):
    ds_t = nc.Dataset(os.path.join(CARPETA_T, archivo_t))
    ds_s = nc.Dataset(os.path.join(CARPETA_S, archivo_s))

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

    # Filtrar NaN
    validos = ~np.isnan(temp) & ~np.isnan(sal)
    return sal[validos], temp[validos], dep[validos]


# ─── ISOPICNAS (líneas de densidad constante) ─────
def dibujar_isopicnas(ax, S_range, T_range):
    S_grid = np.linspace(S_range[0], S_range[1], 100)
    T_grid = np.linspace(T_range[0], T_range[1], 100)
    S_2d, T_2d = np.meshgrid(S_grid, T_grid)

    # Calcular densidad potencial en la grilla
    SA_grid = gsw.SA_from_SP(S_2d, 0, -75, 10)  # presión 0, lon/lat referencia
    CT_grid = gsw.CT_from_t(SA_grid, T_2d, 0)
    sig_grid = gsw.sigma0(SA_grid, CT_grid)

    niveles = np.arange(20, 29, 0.5)
    cs = ax.contour(S_2d, T_2d, sig_grid,
                    levels=niveles, colors="gray",
                    linewidths=0.5, alpha=0.4, linestyles="--")
    ax.clabel(cs, fmt="%.1f", fontsize=7, colors="gray")


# ─── FIGURA 1: Un diagrama T-S por mes (3 figuras) ─
def figura_TS_por_mes(mes, archivo_t, archivo_s):
    fig, ax = plt.subplots(figsize=(9, 7))

    S_todos, T_todos = [], []

    for region, (lat, lon) in REGIONES.items():
        sal, temp, dep = leer_perfil_ts(archivo_t, archivo_s, lat, lon)
        if len(sal) == 0:
            print(f"    ⚠ Sin datos: {region.replace(chr(10), ' ')}")
            continue

        S_todos.extend(sal.tolist())
        T_todos.extend(temp.tolist())

        # Colorear por profundidad dentro de cada región
        sc = ax.scatter(sal, temp,
                        c=dep, cmap="viridis_r",
                        vmin=0, vmax=PROF_MAX,
                        s=15, alpha=0.7,
                        marker=MARCADORES_MES[mes],
                        label=region.replace("\n", " "),
                        edgecolors=COLORES_REGION[region],
                        linewidths=0.8)

    if not S_todos:
        plt.close()
        return

    # Isopicnas
    S_min = max(28, min(S_todos) - 0.5)
    S_max = min(40, max(S_todos) + 0.5)
    T_min = max(0,  min(T_todos) - 1)
    T_max = min(35, max(T_todos) + 1)
    dibujar_isopicnas(ax, [S_min, S_max], [T_min, T_max])

    # Barra de color (profundidad)
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Profundidad (m)", fontsize=10)

    # Etiquetas de masas de agua
    for nombre, S_ma, T_ma in MASAS_AGUA:
        if S_min <= S_ma <= S_max and T_min <= T_ma <= T_max:
            ax.annotate(nombre,
                        xy=(S_ma, T_ma),
                        fontsize=7.5,
                        color="navy",
                        bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow",
                                  ec="navy", alpha=0.7),
                        ha="center")

    ax.set_xlabel("Salinidad Práctica (psu)", fontsize=12)
    ax.set_ylabel("Temperatura (°C)", fontsize=12)
    ax.set_title(
        f"Diagrama T-S — {mes}\nCaribe y Pacífico Colombiano (WOA23, TEOS-2010)",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85,
              title="Región", title_fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    nombre_fig = f"P1.3_DiagramaTS_{mes}.png"
    plt.savefig(os.path.join(CARPETA_OUT, nombre_fig), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    → Guardada: figuras/{nombre_fig}")


# ─── FIGURA 2: Los 3 meses juntos por región ──────
def figura_TS_comparativa():
    """
    Un solo gráfico con los 3 meses superpuestos para todas las regiones.
    Permite comparar estacionalidad.
    """
    fig, ax = plt.subplots(figsize=(11, 8))

    S_todos, T_todos = [], []

    for mes, (arch_t, arch_s) in ARCHIVOS.items():
        for region, (lat, lon) in REGIONES.items():
            sal, temp, dep = leer_perfil_ts(arch_t, arch_s, lat, lon)
            if len(sal) == 0:
                continue

            S_todos.extend(sal.tolist())
            T_todos.extend(temp.tolist())

            ax.scatter(sal, temp,
                       color=COLORES_REGION[region],
                       marker=MARCADORES_MES[mes],
                       s=12,
                       alpha=ALPHA_MES[mes],
                       linewidths=0)

    if not S_todos:
        plt.close()
        return

    # Isopicnas
    S_min = max(28, min(S_todos) - 0.5)
    S_max = min(40, max(S_todos) + 0.5)
    T_min = max(0,  min(T_todos) - 1)
    T_max = min(35, max(T_todos) + 1)
    dibujar_isopicnas(ax, [S_min, S_max], [T_min, T_max])

    # Leyenda regiones (colores)
    parches_region = [
        mpatches.Patch(color=COLORES_REGION[r], label=r.replace("\n", " "))
        for r in REGIONES
    ]
    # Leyenda meses (marcadores)
    import matplotlib.lines as mlines
    parches_mes = [
        mlines.Line2D([], [], color="black",
                      marker=MARCADORES_MES[m], linestyle="None",
                      markersize=7, label=m, alpha=ALPHA_MES[m])
        for m in ARCHIVOS
    ]

    leg1 = ax.legend(handles=parches_region, loc="upper left",
                     fontsize=8, title="Región", title_fontsize=9,
                     framealpha=0.85)
    ax.add_artist(leg1)
    ax.legend(handles=parches_mes, loc="lower right",
              fontsize=9, title="Mes", title_fontsize=9,
              framealpha=0.85)

    # Masas de agua
    for nombre, S_ma, T_ma in MASAS_AGUA:
        if S_min <= S_ma <= S_max and T_min <= T_ma <= T_max:
            ax.annotate(nombre, xy=(S_ma, T_ma),
                        fontsize=7.5, color="navy",
                        bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow",
                                  ec="navy", alpha=0.8),
                        ha="center")

    ax.set_xlabel("Salinidad Práctica (psu)", fontsize=12)
    ax.set_ylabel("Temperatura (°C)", fontsize=12)
    ax.set_title(
        "Diagrama T-S Comparativo — Enero, Julio, Octubre\n"
        "Caribe y Pacífico Colombiano (WOA23, TEOS-2010)",
        fontsize=13, fontweight="bold"
    )
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    nombre_fig = "P1.3_DiagramaTS_Comparativo_3meses.png"
    plt.savefig(os.path.join(CARPETA_OUT, nombre_fig), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    → Guardada: figuras/{nombre_fig}")


# ─── MAIN ─────────────────────────────────────────
def main():
    print("=" * 58)
    print("  PREGUNTA 1.3 — Diagramas T-S")
    print("  Regiones: Caribe + Pacífico | Enero, Julio, Octubre")
    print("=" * 58)

    # Figura por mes
    for mes, (arch_t, arch_s) in ARCHIVOS.items():
        print(f"\n  Mes: {mes}")
        figura_TS_por_mes(mes, arch_t, arch_s)

    # Figura comparativa (los 3 meses juntos)
    print(f"\n  Generando figura comparativa (3 meses juntos)...")
    figura_TS_comparativa()

    print(f"\n{'='*58}")
    print(f"  ✓ Listo. Figuras generadas:")
    for f in sorted(os.listdir(CARPETA_OUT)):
        if "P1.3" in f:
            print(f"    • {f}")
    print("=" * 58)


if __name__ == "__main__":
    main()