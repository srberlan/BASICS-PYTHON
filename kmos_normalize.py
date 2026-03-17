import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from astropy.stats import sigma_clip


# =========================
# CONFIGURACION
# =========================

band = "K"   # elegir: "YJ", "H", "K"

# parametros por banda
params = {

    "YJ": dict(s=0.002, sigma_low=1.6, sigma_high=3, niter=6),

    "H": dict(s=0.005, sigma_low=1.7, sigma_high=3, niter=8),

    "K": dict(s=0.01, sigma_low=1.8, sigma_high=3, niter=8)

}

p = params[band]

# crear carpetas salida
os.makedirs("normalized", exist_ok=True)
os.makedirs("plots", exist_ok=True)


# =========================
# PROCESADO
# =========================

files = glob.glob("*.txt")

print("Procesando", len(files), "espectros")

for f in files:

    data = np.loadtxt(f)

    lam = data[:,0]
    flux = data[:,1]

    mask = np.isfinite(flux)

    # mascara Brγ para banda K
    if band == "K":
        brgamma = (lam > 2.155) & (lam < 2.175)
        mask = mask & (~brgamma)

    lam = lam[mask]
    flux = flux[mask]

    # ajuste inicial
    spline = UnivariateSpline(lam, flux, s=p["s"]*len(lam))
    cont = spline(lam)

    # iteraciones sigma clipping
    for i in range(p["niter"]):

        ratio = flux/cont

        clipped = sigma_clip(
            ratio,
            sigma_lower=p["sigma_low"],
            sigma_upper=p["sigma_high"]
        )

        good = ~clipped.mask

        spline = UnivariateSpline(
            lam[good],
            flux[good],
            s=p["s"]*len(lam)
        )

        cont = spline(lam)

    flux_norm = flux/cont

    # =================
    # GUARDAR ESPECTRO
    # =================

    out = np.column_stack([lam, flux_norm])

    outfile = os.path.join("normalized", "norm_"+f)
    np.savetxt(outfile, out)


    # =================
    # PLOT CONTROL
    # =================

    plt.figure(figsize=(8,4))

    plt.plot(lam, flux, lw=1, label="Original")
    plt.plot(lam, cont, lw=2, label="Continuo")

    plt.xlabel("Wavelength")
    plt.ylabel("Flux")

    plt.title(f)
    plt.legend()

    plotfile = os.path.join("plots", f.replace(".txt",".png"))

    plt.savefig(plotfile, dpi=150)
    plt.close()


print("Normalización terminada.")
