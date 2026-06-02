"""
=====================================================
  PARTE II — CORRIENTES GEOSTRÓFICAS (v4 final)
=====================================================
Corrección clave v4:
  - Altura dinámica calculada hasta el nivel más profundo
    COMÚN entre las dos estaciones del transecto.
  - Esto evita integrar hasta profundidades distintas
    y obtener diferencias ΔD sin sentido físico.
=====================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
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


# ─── LEER PERFIL ─────────────────────────────────
def leer_perfil(mes_t, mes_s, lat_est, lon_est, prof_max=1000):
    ds_t = nc.Dataset(os.path.join(CARPETA_T, mes_t))
    ds_s = nc.Dataset(os.path.join(CARPETA_S, mes_s))

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

    SA     = np.full_like(sal,  np.nan)
    CT     = np.full_like(temp, np.nan)
    sigma0 = np.full_like(temp, np.nan)
    specv  = np.full_like(temp, np.nan)

    ok = (~np.isnan(temp)) & (~np.isnan(sal))
    if ok.any():
        SA[ok]     = gsw.SA_from_SP(sal[ok],  presion[ok], lon_est, lat_est)
        CT[ok]     = gsw.CT_from_t(SA[ok], temp[ok], presion[ok])
        sigma0[ok] = gsw.sigma0(SA[ok], CT[ok])
        specv[ok]  = gsw.specvol(SA[ok], CT[ok], presion[ok])

    return dep, temp, sal, sigma0, presion, specv


# ─── ALTURA DINÁMICA (anomalía, ref = nivel más profundo común) ──
def altura_dinamica_par(specv_c, specv_o, presion_c, presion_o):
    """
    Calcula D(z) para dos estaciones usando el mismo nivel de referencia:
    el nivel más profundo donde AMBAS tienen datos válidos.

    D(p) = ∫_p^p_ref  δα dp'    donde δα = α - α₀, α₀ = 1/rho0

    Devuelve D_c, D_o evaluados en los índices válidos comunes,
    y los vectores de profundidad y presión comunes.
    """
    alpha_0 = 1.0 / rho0
    delta_c = specv_c - alpha_0
    delta_o = specv_o - alpha_0

    # Niveles válidos en AMBAS estaciones
    n   = min(len(specv_c), len(specv_o))
    P   = presion_c[:n]  # presión común (misma grilla de profundidades WOA)

    ok_ambas = (~np.isnan(delta_c[:n])) & (~np.isnan(delta_o[:n]))
    if ok_ambas.sum() < 2:
        return None, None, None

    idx = np.where(ok_ambas)[0]

    # Referencia = último nivel válido común
    D_c = np.full(n, np.nan)
    D_o = np.full(n, np.nan)
    D_c[idx[-1]] = 0.0
    D_o[idx[-1]] = 0.0

    for k in range(len(idx) - 2, -1, -1):
        i  = idx[k]
        i1 = idx[k + 1]
        dp = (P[i1] - P[i]) * 1e4   # dbar → Pa
        D_c[i] = D_c[i1] + 0.5 * (delta_c[i] + delta_c[i1]) * dp
        D_o[i] = D_o[i1] + 0.5 * (delta_o[i] + delta_o[i1]) * dp

    return D_c, D_o, idx


# ══════════════════════════════════════════════════
#  PREGUNTA 2.1 — La Guajira | Febrero
# ══════════════════════════════════════════════════

EST_GUAJIRA = {
    "Manaure":  [(11.80, -72.60), (12.00, -72.50)],
    "Riohacha": [(11.60, -73.00), (12.00, -73.00)],
    "Palomino": [(11.40, -73.80), (11.80, -73.80)],
}

ARCH_T_FEB = "woa23_decav_t02_04.nc"
ARCH_S_FEB = "woa23_decav91C0_s02_04.nc"


def pregunta_2_1():
    print("\n" + "═"*62)
    print("  PREGUNTA 2.1 — Velocidad Geostrófica | La Guajira")
    print("  Método: Altura Dinámica + Altimetría Satelital")
    print("═"*62)

    colores = ["#e63946", "#2a9d8f", "#e9c46a"]
    resultados = {}

    for transecto, (est_c, est_o) in EST_GUAJIRA.items():
        lat_c, lon_c = est_c
        lat_o, lon_o = est_o

        dep_c, T_c, S_c, sig_c, P_c, specv_c = leer_perfil(
            ARCH_T_FEB, ARCH_S_FEB, lat_c, lon_c)
        dep_o, T_o, S_o, sig_o, P_o, specv_o = leer_perfil(
            ARCH_T_FEB, ARCH_S_FEB, lat_o, lon_o)

        D_c, D_o, idx = altura_dinamica_par(specv_c, specv_o, P_c, P_o)

        # Distancia y Coriolis
        dlat   = abs(lat_o - lat_c) * 111000
        dlon   = abs(lon_o - lon_c) * 111000 * np.cos(np.radians((lat_c+lat_o)/2))
        dist_m = np.sqrt(dlat**2 + dlon**2)
        lat_med = (lat_c + lat_o) / 2
        f = coriolis(lat_med)

        # a) Velocidad por altura dinámica en superficie (primer idx válido)
        if D_c is not None:
            i0   = idx[0]   # nivel más superficial válido
            vg_AD = (D_o[i0] - D_c[i0]) / (f * dist_m)
            prof_ref = dep_c[idx[-1]]   # profundidad de referencia
            D_c0 = D_c[i0]
            D_o0 = D_o[i0]
        else:
            vg_AD, prof_ref, D_c0, D_o0 = np.nan, np.nan, np.nan, np.nan

        # b) Altimetría satelital (dato del enunciado)
        vg_ALT = (g / f) * (0.02 / 100e3)   # 2 cm / 100 km → m/s

        resultados[transecto] = {
            "dist_km":  dist_m / 1000,
            "f":        f,
            "prof_ref": prof_ref,
            "D_c0":     D_c0,
            "D_o0":     D_o0,
            "vg_AD":    vg_AD,
            "vg_ALT":   vg_ALT,
            "dep":      dep_o,
            "D_perfil": D_o,
        }

        vg_AD_str = f"{vg_AD*100:.2f} cm/s" if not np.isnan(vg_AD) else "N/A"
        dif_str   = f"{abs(vg_AD - vg_ALT)*100:.2f} cm/s" if not np.isnan(vg_AD) else "N/A"
        ref_str   = f"{prof_ref:.0f} m"    if not np.isnan(prof_ref) else "N/A"
        print(f"\n  Transecto: {transecto}")
        print(f"    Distancia:                   {dist_m/1000:.1f} km")
        print(f"    f (Coriolis):                {f:.2e} s⁻¹")
        print(f"    Nivel de referencia común:   {ref_str}")
        print(f"    ΔD sup (océan−cost):         {(D_o0-D_c0):.6f} m²/s²")
        print(f"    ── a) v_g Altura Dinámica:   {vg_AD_str}")
        print(f"    ── b) v_g Altimetría:         {vg_ALT*100:.2f} cm/s")
        print(f"    ── c) Diferencia:             {dif_str}")

    # ── Figura: barras comparativas ────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for i, (transecto, res) in enumerate(resultados.items()):
        ax = axes[i]
        vg_AD  = res["vg_AD"]
        vg_ALT = res["vg_ALT"]

        val_AD  = abs(vg_AD  * 100) if not np.isnan(vg_AD) else 0.0
        val_ALT = abs(vg_ALT * 100)
        lim_max = max(val_AD, val_ALT) * 1.5
        if lim_max < 0.5:
            lim_max = 10.0

        bars = ax.bar(["Altura\nDinámica", "Altimetría\nSatelital"],
                      [val_AD, val_ALT],
                      color=[colores[0], colores[1]],
                      edgecolor="black", linewidth=0.8, width=0.5)
        etiq_AD = f"{vg_AD*100:.2f} cm/s" if not np.isnan(vg_AD) else "N/A"
        ax.text(bars[0].get_x() + bars[0].get_width()/2,
                bars[0].get_height() + lim_max*0.02,
                etiq_AD, ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text(bars[1].get_x() + bars[1].get_width()/2,
                bars[1].get_height() + lim_max*0.02,
                f"{val_ALT:.2f} cm/s", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

        ref_str = f"Ref: {res['prof_ref']:.0f} m" if not np.isnan(res['prof_ref']) else ""
        ax.set_title(f"Transecto {transecto}\n({ref_str})", fontsize=11, fontweight="bold")
        ax.set_ylabel("v_g superficial |cm/s|", fontsize=10)
        ax.set_ylim([0, lim_max])
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(
        "Velocidad Geostrófica Superficial — La Guajira | Febrero\n"
        "Altura Dinámica (ref. nivel más profundo común) vs Altimetría Satelital",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_OUT, "P2.1_Guajira_VelocidadGeostrofica.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  → Guardada: P2.1_Guajira_VelocidadGeostrofica.png")

    # ── Figura: perfiles de altura dinámica ────────
    fig, ax = plt.subplots(figsize=(7, 9))
    for i, (transecto, res) in enumerate(resultados.items()):
        dep = res["dep"]
        D   = res["D_perfil"]
        ok  = ~np.isnan(D)
        if ok.sum() > 1:
            ax.plot(D[ok] * 10, dep[ok],          # ×10 → dyn-cm
                    color=colores[i], linewidth=2,
                    marker="o", markersize=4, markevery=2, label=transecto)
    ax.invert_yaxis()
    ax.set_ylabel("Profundidad (m)", fontsize=12)
    ax.set_xlabel("Anomalía Altura Dinámica (dyn-cm)", fontsize=12)
    ax.set_title("Perfil Altura Dinámica — La Guajira | Febrero\n"
                 "(referencia = nivel más profundo común del par)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_ylim([1000, 0])
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_OUT, "P2.1_Guajira_AlturaDinamica.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Guardada: P2.1_Guajira_AlturaDinamica.png")

    return resultados


# ══════════════════════════════════════════════════
#  PREGUNTA 2.2 — Pacífico Colombiano | Febrero
# ══════════════════════════════════════════════════

EST_PACIFICO = {
    "Cupica": [(6.50, -77.60), (6.50, -77.80)],
    "Nuqui":  [(5.80, -77.50), (5.50, -77.80)],
    "Tumaco": [(2.00, -79.00), (1.50, -79.50)],
}


def pregunta_2_2():
    print("\n" + "═"*62)
    print("  PREGUNTA 2.2 — Viento Térmico | Pacífico Colombiano")
    print("═"*62)

    for transecto, (est_c, est_o) in EST_PACIFICO.items():
        lat_c, lon_c = est_c
        lat_o, lon_o = est_o

        dep_c, _, _, sig_c, _, _ = leer_perfil(ARCH_T_FEB, ARCH_S_FEB, lat_c, lon_c)
        dep_o, _, _, sig_o, _, _ = leer_perfil(ARCH_T_FEB, ARCH_S_FEB, lat_o, lon_o)

        n     = min(len(dep_c), len(dep_o))
        dep   = dep_c[:n]
        sig_c = sig_c[:n]
        sig_o = sig_o[:n]

        dlat   = abs(lat_o - lat_c) * 111000
        dlon   = abs(lon_o - lon_c) * 111000 * np.cos(np.radians((lat_c+lat_o)/2))
        dist_m = np.sqrt(dlat**2 + dlon**2)
        f      = coriolis((lat_c + lat_o) / 2)

        # Gradiente horizontal de densidad y cortante (viento térmico)
        drho_dx  = np.full(n, np.nan)
        ok = (~np.isnan(sig_c)) & (~np.isnan(sig_o))
        drho_dx[ok] = (sig_o[ok] - sig_c[ok]) / dist_m
        cortante = np.where(~np.isnan(drho_dx),
                            (g / (rho0 * f)) * drho_dx, np.nan)

        # a) 0-200 m
        idx_200      = np.where(dep <= 200)[0]
        dep_200       = dep[idx_200]
        cortante_200  = cortante[idx_200]

        # b) Integrar desde 500 m (vg=0 en 500 m)
        idx_500      = np.where(dep <= 500)[0]
        dep_500       = dep[idx_500]
        cortante_500  = cortante[idx_500]

        vg_abs = np.zeros(len(dep_500))
        for i in range(len(dep_500) - 2, -1, -1):
            dz    = dep_500[i+1] - dep_500[i]
            c_mid = 0.5*(cortante_500[i] + cortante_500[i+1])
            vg_abs[i] = vg_abs[i+1] + (c_mid if not np.isnan(c_mid) else 0.0)*dz

        # c) Transporte 0-300 m
        idx_300 = np.where(dep_500 <= 300)[0]
        transporte = float(_trapz(vg_abs[idx_300], dep_500[idx_300])) \
                     if len(idx_300) > 1 else 0.0

        c_sup = cortante_200[0]  if not np.isnan(cortante_200[0])  else 0.0
        c_200 = cortante_200[-1] if not np.isnan(cortante_200[-1]) else 0.0

        print(f"\n  Transecto: {transecto}")
        print(f"    Distancia:          {dist_m/1000:.1f} km")
        print(f"    f (Coriolis):       {f:.2e} s⁻¹")
        print(f"    Cortante sup (0m):  {c_sup*1000:.4f} ×10⁻³ s⁻¹")
        print(f"    Cortante 200m:      {c_200*1000:.4f} ×10⁻³ s⁻¹")
        print(f"    v_g sup (ref 500m): {vg_abs[0]*100:.2f} cm/s")
        print(f"    Transporte 0-300m:  {transporte:.2f} m²/s")

        # ── Figura ───────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(15, 7))

        v_ok = ~np.isnan(cortante_200)
        if v_ok.any():
            axes[0].plot(cortante_200[v_ok]*1000, dep_200[v_ok],
                         color="#e63946", linewidth=2, marker="o", markersize=4)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("∂v_g/∂z (×10⁻³ s⁻¹)", fontsize=11)
        axes[0].set_ylabel("Profundidad (m)", fontsize=11)
        axes[0].set_title("a) Cortante vertical (0-200 m)", fontsize=11)
        axes[0].axvline(0, color="black", linewidth=0.8)
        axes[0].grid(True, linestyle="--", alpha=0.4)

        axes[1].plot(vg_abs*100, dep_500,
                     color="#2a9d8f", linewidth=2, marker="s", markersize=4)
        axes[1].invert_yaxis()
        axes[1].set_xlabel("v_g (cm/s)", fontsize=11)
        axes[1].set_title("b) v_g absoluta\n(ref: 500 m, vg=0)", fontsize=11)
        axes[1].axvline(0, color="black", linewidth=0.8)
        axes[1].grid(True, linestyle="--", alpha=0.4)
        # Sombrear transporte
        axes[1].fill_betweenx(dep_500, 0, vg_abs*100, alpha=0.15, color="#2a9d8f")

        s_ok = (~np.isnan(sig_c)) & (~np.isnan(sig_o))
        axes[2].plot(sig_c[s_ok], dep[s_ok], color="#e63946",
                     label="Costera", linewidth=2)
        axes[2].plot(sig_o[s_ok], dep[s_ok], color="#457b9d",
                     label="Oceánica", linewidth=2, linestyle="--")
        axes[2].invert_yaxis()
        axes[2].set_xlabel("σθ (kg/m³)", fontsize=11)
        axes[2].set_title("Densidad potencial σθ", fontsize=11)
        axes[2].legend(fontsize=9)
        axes[2].grid(True, linestyle="--", alpha=0.4)
        axes[2].set_ylim([500, 0])

        fig.suptitle(
            f"Viento Térmico — {transecto} | Pacífico Colombiano | Febrero\n"
            f"Transporte 0-300m: {transporte:.2f} m²/s",
            fontsize=12, fontweight="bold"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(CARPETA_OUT,
                    f"P2.2_Pacifico_{transecto}_VientoTermico.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  → Guardada: P2.2_Pacifico_{transecto}_VientoTermico.png")


# ══════════════════════════════════════════════════
#  PREGUNTA 2.3 — Golfo de Salamanca
# ══════════════════════════════════════════════════

def pregunta_2_3():
    print("\n" + "═"*62)
    print("  PREGUNTA 2.3 — Efecto Río Magdalena | Golfo Salamanca")
    print("═"*62)

    delta_S  = 5.0
    delta_x  = 50e3
    lat_sal  = 11.2
    f        = coriolis(lat_sal)
    beta     = 0.8
    drho_dx  = beta * (delta_S / delta_x)
    cortante = (g / (rho0 * f)) * drho_dx

    print(f"\n  Parámetros del enunciado:")
    print(f"    ΔS = {delta_S} psu / {delta_x/1000:.0f} km")
    print(f"    f  = {f:.2e} s⁻¹")
    print(f"    β  = {beta} kg/m³/psu")
    print(f"    ∂ρ/∂x = {drho_dx*1e6:.1f} ×10⁻⁶ kg/m⁴")
    print(f"    ∂vg/∂z = {cortante*1000:.4f} ×10⁻³ s⁻¹")

    z_ref = 30.0
    profundidades = np.array([0, 10, 20, 30, 50, 75, 100], dtype=float)
    vg = np.array([cortante*(z_ref - z) if z <= z_ref else 0.0
                   for z in profundidades])
    vg_sup = cortante * z_ref

    print(f"    vg superficial = {vg_sup*100:.2f} cm/s")
    print(f"\n  {'Prof(m)':10s} {'v_g (cm/s)':12s}")
    print(f"  {'-'*24}")
    for z, v in zip(profundidades, vg):
        print(f"  {z:10.0f} {v*100:12.4f}")

    transporte = float(_trapz(vg[:4], profundidades[:4]))
    print(f"\n  Transporte geostrófico (0-{z_ref:.0f}m): {transporte:.4f} m²/s")

    # ── Figura ───────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].plot(vg*100, profundidades, color="#e63946",
                 linewidth=2.5, marker="o", markersize=6)
    axes[0].invert_yaxis()
    axes[0].fill_betweenx(profundidades, 0, vg*100, alpha=0.2, color="#e63946")
    axes[0].axhline(z_ref, color="navy", linestyle="--", linewidth=1.5,
                    label=f"Ref (base MLD): {z_ref:.0f} m")
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("v_g (cm/s)", fontsize=12)
    axes[0].set_ylabel("Profundidad (m)", fontsize=12)
    axes[0].set_title("Velocidad Geostrófica\npor efecto del Río Magdalena",
                      fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.4)
    axes[0].text(vg_sup*100*0.5, 5,
                 f"vg(0) = {vg_sup*100:.1f} cm/s",
                 fontsize=10, color="#e63946", fontweight="bold")

    etiq = ["ΔS\n(psu)", "Δx\n(km)", "∂ρ/∂x\n(×10⁻⁶\nkg/m⁴)",
            "∂vg/∂z\n(×10⁻³ s⁻¹)", "vg sup\n(cm/s)"]
    vals = [delta_S, delta_x/1000, drho_dx*1e6, cortante*1000, vg_sup*100]
    cols = ["#457b9d", "#2a9d8f", "#e9c46a", "#f4a261", "#e63946"]
    bars = axes[1].bar(etiq, vals, color=cols, edgecolor="black", linewidth=0.8)
    for bar, v in zip(bars, vals):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + max(vals)*0.02,
                     f"{v:.2f}", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")
    axes[1].set_ylabel("Valor", fontsize=11)
    axes[1].set_title("Parámetros del balance geostrófico", fontsize=11, fontweight="bold")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.4)

    fig.suptitle(
        "Efecto del Río Magdalena en el Balance Geostrófico\n"
        "Golfo de Salamanca — ΔS = 5 psu / 50 km",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA_OUT, "P2.3_GolfoSalamanca_EfectoMagdalena.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → Guardada: P2.3_GolfoSalamanca_EfectoMagdalena.png")


# ─── MAIN ─────────────────────────────────────────
def main():
    print("=" * 62)
    print("  PARTE II — CORRIENTES GEOSTRÓFICAS (v4 final)")
    print("=" * 62)
    pregunta_2_1()
    pregunta_2_2()
    pregunta_2_3()
    print(f"\n{'='*62}")
    figuras = [f for f in sorted(os.listdir(CARPETA_OUT)) if f.startswith("P2")]
    print(f"  Figuras generadas ({len(figuras)}):")
    for f in figuras:
        print(f"    • {f}")
    print("=" * 62)

if __name__ == "__main__":
    main()