"""Render the publication figures for the Casimir-electrostatic NEMS pull-in
article (Physical Review Applied).

This script is *purely a renderer*: it loads only already-saved arrays from
``results/`` and ``logs/`` and writes vector PDF + 600-dpi PNG for each of the
four figures to ``figures/``.  No physics is recomputed and no model is
retrained here.

Run:  python src/make_figures.py
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import figstyle as fs

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, os.pardir))
RESULTS = os.path.join(ROOT, "results")
LOGS = os.path.join(ROOT, "logs")


def _load(name):
    return np.load(os.path.join(RESULTS, name), allow_pickle=True)


# ----------------------------------------------------------------------------
# Fig. 1 -- time trajectories xi(tau) for the three regimes
# ----------------------------------------------------------------------------
def figure1():
    regimes = [
        ("stable", "Stable", "traj_stable.npz"),
        ("growing", "Growing", "traj_growing.npz"),
        ("pullin", "Pull-in", "traj_pullin.npz"),
    ]
    # tau_PI for the pull-in case comes from the DOP853 reference verification.
    with open(os.path.join(RESULTS, "verification_summary.json")) as fh:
        tau_pi = json.load(fh)["reference"]["tau_star"]

    ref_c = fs.PALETTE["black"]
    pinn_c = fs.PALETTE["vermillion"]

    fig, axes = plt.subplots(1, 3, figsize=(fs.COL_DOUBLE, 2.55))
    labels = ["(a)", "(b)", "(c)"]

    for ax, (key, title, fname), lab in zip(axes, regimes, labels):
        d = _load(fname)
        t, xr, xp = d["t"], d["xi_ref"], d["xi_pinn"]
        a, b, z = float(d["alpha"]), float(d["beta"]), float(d["zeta"])

        ax.plot(t, xr, "-", color=ref_c, lw=1.3, label="Reference (DOP853)", zorder=3)
        ax.plot(t, xp, "--", color=pinn_c, lw=1.3, label="PINN", zorder=4)
        # sparse open markers to show PINN is a distinct solution
        step = max(1, len(t) // 12)
        ax.plot(t[::step], xp[::step], "o", mfc="none", mec=pinn_c,
                mew=0.9, ms=3.5, zorder=5)

        ax.grid(True)
        ax.set_xlabel(r"scaled time  $\tau$")
        ax.margins(x=0.02)
        ax.text(0.04, 0.93, lab, transform=ax.transAxes, fontweight="bold",
                va="top", ha="left")
        # regime name + parameters
        ptxt = (rf"{title}" "\n"
                rf"$\alpha={a:.3g}$" "\n"
                rf"$\beta={b:.3g}$" "\n"
                rf"$\zeta={z:.2g}$")

        if key == "pullin":
            ax.axvline(tau_pi, ls=":", color=fs.PALETTE["blue"], lw=1.1, zorder=2)
            ax.axhline(1.0, ls="-", color=fs.PALETTE["grey"], lw=0.8, zorder=1)
            ax.set_ylim(0, 1.08)
            ax.set_xlim(0, tau_pi * 1.04)
            ax.annotate(r"$\xi\to 1$", xy=(tau_pi, 1.0),
                        xytext=(tau_pi * 0.60, 0.90), fontsize=7.5,
                        color=fs.PALETTE["blue"],
                        arrowprops=dict(arrowstyle="->", color=fs.PALETTE["blue"],
                                        lw=0.8))
            ax.text(tau_pi, 0.05, r"$\tau_{\mathrm{PI}}$", color=fs.PALETTE["blue"],
                    fontsize=7.5, ha="right", va="bottom", rotation=90)
            ax.text(0.055, 0.83, ptxt, transform=ax.transAxes, va="top",
                    ha="left", fontsize=7.5)
        else:
            ax.text(0.06, 0.60, ptxt, transform=ax.transAxes, va="center",
                    ha="left", fontsize=7.5)

        # error inset (pointwise |xi_pinn - xi_ref|)
        axin = inset_axes(ax, width="42%", height="30%", loc="upper right",
                          borderpad=0.6)
        err = np.abs(xp - xr)
        axin.semilogy(t, np.maximum(err, 1e-12), "-", color=fs.PALETTE["green"],
                      lw=0.8)
        axin.set_ylim(1e-6, 1e-2)
        axin.tick_params(labelsize=5.5, length=2, pad=1.5)
        axin.yaxis.set_major_locator(mticker.LogLocator(numticks=3))
        axin.grid(True, which="major", alpha=0.25)
        axin.set_title(r"$|\xi_{\mathrm{PINN}}-\xi_{\mathrm{ref}}|$", fontsize=5.8,
                       pad=1.5)
        if key == "pullin":
            axin.set_xlim(0, tau_pi * 1.04)

    axes[0].set_ylabel(r"normalized displacement  $\xi(\tau)=x/d$")

    # single legend (top, spanning)
    handles = [
        Line2D([], [], color=ref_c, lw=1.3, ls="-", label="Reference (DOP853)"),
        Line2D([], [], color=pinn_c, lw=1.3, ls="--", marker="o", mfc="none",
               label="PINN"),
        Line2D([], [], color=fs.PALETTE["blue"], lw=1.1, ls=":",
               label=r"$\tau_{\mathrm{PI}}$ (pull-in time)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.03), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fs.savefig(fig, "fig1_trajectories")


# ----------------------------------------------------------------------------
# Fig. 2 -- (alpha,beta) pull-in phase diagram  [flagship]
# ----------------------------------------------------------------------------
def figure2():
    d = _load("phase_diagram.npz")
    ag, bg = d["alpha_grid"], d["beta_grid"]
    tau = d["tau_pullin"].copy()
    mask = d["pullin_mask"]
    mask_cmp = d["pullin_mask_compare"]
    zeta = float(d["zeta"])
    zeta_cmp = float(d["zeta_compare"])
    fa, fb = d["fold_alpha_c"], d["fold_beta_c"]

    pb = _load("pinn_boundary.npz")
    b_pinn = pb["fold_beta_samples"]
    a_pinn = pb["fold_alpha_pinn"]
    alpha_c0 = 4.0 / 27.0            # beta = 0 endpoint
    beta_star = float(pb["beta_c_alpha0_true"])  # alpha = 0 endpoint (0.08192)

    fig, ax = plt.subplots(figsize=(fs.COL_SINGLE, 2.95))

    # stable region -> flat light grey background
    ax.set_facecolor("#E9E9E9")

    tau_plot = np.ma.array(tau, mask=~mask)
    vmax = float(np.percentile(tau[mask], 97))
    vmin = float(tau[mask].min())
    pcm = ax.pcolormesh(ag, bg, tau_plot, cmap=fs.SEQ_CMAP, shading="auto",
                        vmin=vmin, vmax=vmax, rasterized=True)

    cbar = fig.colorbar(pcm, ax=ax, pad=0.02, extend="max", fraction=0.055)
    cbar.set_label(r"pull-in time  $\tau_{\mathrm{PI}}$", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # analytic static fold boundary (solid)
    ax.plot(fa, fb, "-", color=fs.PALETTE["black"], lw=1.4,
            label="static fold (analytic)", zorder=6)

    # dynamic pull-in boundaries extracted as the 0.5-level contour of the
    # boolean masks (clean and robust).  zeta=0.1 sits inside the static fold
    # (kinetic overshoot band); zeta=0.7 nearly coincides with the static fold.
    ax.contour(ag, bg, mask.astype(float), levels=[0.5],
               colors=[fs.PALETTE["vermillion"]], linewidths=1.3,
               linestyles="--", zorder=7)
    ax.contour(ag, bg, mask_cmp.astype(float), levels=[0.5],
               colors=[fs.PALETTE["blue"]], linewidths=1.2,
               linestyles=":", zorder=7)

    # PINN-predicted boundary (sparse markers only, distinct colour)
    good = np.isfinite(a_pinn)
    ax.plot(a_pinn[good][::10], b_pinn[good][::10], "s",
            color=fs.PALETTE["purple"], ms=3.0, mec="k", mew=0.3, zorder=8,
            label="PINN boundary")

    # endpoint markers
    ax.plot([alpha_c0], [0.0], "*", color=fs.PALETTE["orange"], ms=9,
            mec="k", mew=0.4, zorder=9)
    ax.plot([0.0], [beta_star], "*", color=fs.PALETTE["orange"], ms=9,
            mec="k", mew=0.4, zorder=9)
    ax.annotate(r"$\alpha_c=\dfrac{4}{27}$", xy=(alpha_c0, 0.0),
                xytext=(alpha_c0 - 0.006, 0.026), ha="right", fontsize=7.5,
                arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.annotate(r"$\beta^{*}\!\approx 0.082$", xy=(0.0, beta_star),
                xytext=(0.030, beta_star - 0.002), ha="left", fontsize=7.5,
                arrowprops=dict(arrowstyle="->", lw=0.7))

    # region labels
    ax.text(0.145, 0.070, "PULL-IN", color="w", fontsize=8, fontweight="bold",
            ha="center", va="center")
    ax.text(0.058, 0.036, "STABLE", color="#555555", fontsize=8,
            fontweight="bold", ha="center", va="center")

    ax.set_xlim(ag.min(), ag.max())
    ax.set_ylim(bg.min(), bg.max())
    ax.set_xlabel(r"electrostatic number  $\alpha \propto V^{2}$")
    ax.set_ylabel(r"Casimir number  $\beta$")

    leg_handles = [
        Line2D([], [], color=fs.PALETTE["black"], lw=1.4, ls="-",
               label="static fold (analytic)"),
        Line2D([], [], color=fs.PALETTE["vermillion"], lw=1.3, ls="--",
               label=rf"dynamic boundary  $\zeta={zeta:g}$"),
        Line2D([], [], color=fs.PALETTE["blue"], lw=1.2, ls=":",
               label=rf"dynamic boundary  $\zeta={zeta_cmp:g}$"),
        Line2D([], [], color=fs.PALETTE["purple"], lw=0, marker="s", ms=3.0,
               mec="k", mew=0.3, label="PINN boundary"),
    ]
    ax.legend(handles=leg_handles, loc="lower left", fontsize=6.3,
              framealpha=0.9, frameon=True, facecolor="white",
              edgecolor="none", labelspacing=0.28, handlelength=1.9,
              borderpad=0.5)
    ax.text(0.965, 0.50, r"$\tau_{\mathrm{PI}}$ map:  $\zeta=0.1$",
            transform=ax.transAxes, ha="right", va="center", fontsize=7,
            color="w")

    fig.tight_layout()
    return fs.savefig(fig, "fig2_phase_diagram")


# ----------------------------------------------------------------------------
# Fig. 3 -- Lifshitz thermal shift of the pull-in boundary
# ----------------------------------------------------------------------------
def figure3():
    L = _load("lifshitz_boundaries.npz")
    series = [
        ("T0", r"$T=0$ K", fs.PALETTE["blue"], "-"),
        ("T100", r"$T=100$ K", fs.PALETTE["vermillion"], "--"),
        ("T300", r"$T=300$ K", fs.PALETTE["green"], "-."),
    ]

    fig, ax = plt.subplots(figsize=(fs.COL_SINGLE, 2.95))
    ax.grid(True)

    for tag, lab, c, ls in series:
        a = L["alpha_c_" + tag]
        b = L["beta_c_" + tag]
        ax.plot(a, b, ls, color=c, lw=1.4, label=lab)
        bstar = float(L["beta_star_" + tag])
        ax.plot([0.0], [bstar], "o", color=c, ms=4, mec="k", mew=0.3, zorder=5)

    # device operating points A-D: each device sits at its beta on the T=0
    # fold, marking its beta-coordinate (alpha = fold alpha_c(beta), from
    # device_table).  Labels are offset to avoid overlap.
    with open(os.path.join(RESULTS, "device_table.json")) as fh:
        devs = json.load(fh)["devices"]
    # per-device label offsets (data units) tuned to keep A/D readable
    off = {"A": (0.006, -0.0015), "B": (0.006, 0.0009),
           "C": (0.004, 0.0035), "D": (-0.009, 0.0021)}
    da = np.array([dv["alpha_c_T0"] for dv in devs])
    db = np.array([dv["beta"] for dv in devs])
    ax.plot(da, db, "D", color=fs.PALETTE["black"], ms=4.2, mec="w", mew=0.6,
            zorder=7, label=r"device ($\beta$-coord.)")
    for dv in devs:
        nm = dv["name"]
        dx, dy = off[nm]
        ax.annotate(nm, xy=(dv["alpha_c_T0"], dv["beta"]),
                    xytext=(dv["alpha_c_T0"] + dx, dv["beta"] + dy),
                    fontsize=7.5, fontweight="bold", color=fs.PALETTE["black"],
                    ha="center", va="center", zorder=8)

    # T-independent beta = 0 endpoint (alpha_c = 4/27)
    ax.plot([4.0 / 27.0], [0.0], "*", color=fs.PALETTE["orange"], ms=9,
            mec="k", mew=0.4, zorder=6)
    ax.annotate(r"$\alpha_c=4/27$" "\n(T-indep.)", xy=(4.0 / 27.0, 0.0),
                xytext=(0.158, 0.020), ha="left", va="bottom", fontsize=7,
                arrowprops=dict(arrowstyle="->", lw=0.7))

    # beta* intercept motion 0.082 -> 0.063
    b0 = float(L["beta_star_T0"])
    b3 = float(L["beta_star_T300"])
    ax.annotate("", xy=(0.0, b3), xytext=(0.0, b0),
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.0))
    ax.text(0.008, 0.5 * (b0 + b3),
            rf"$\beta^{{*}}:\,{b0:.3f}\!\to\!{b3:.3f}$",
            fontsize=6.8, rotation=90, va="center", ha="left")

    ax.set_xlim(-0.003, 0.20)
    ax.set_ylim(-0.003, 0.092)
    ax.set_xlabel(r"electrostatic number  $\alpha \propto V^{2}$")
    ax.set_ylabel(r"Casimir number  $\beta$")
    ax.legend(loc="lower left", fontsize=7.2, title="pull-in boundary",
              title_fontsize=7.2)

    # inset: Delta V_PI / V_PI at beta = 0.03 (alpha ~ V^2 -> dV/V=sqrt-1)
    a_b003 = {t: float(L[f"alpha_at_beta003_{t}"]) for t in ("T0", "T100", "T300")}
    dv100 = np.sqrt(a_b003["T100"] / a_b003["T0"]) - 1.0
    dv300 = np.sqrt(a_b003["T300"] / a_b003["T0"]) - 1.0
    axin = inset_axes(ax, width="38%", height="32%", loc="upper right",
                      borderpad=1.1)
    xs = [100, 300]
    ys = [100 * dv100, 100 * dv300]
    axin.bar([0, 1], ys, color=[fs.PALETTE["vermillion"], fs.PALETTE["green"]],
             width=0.6)
    for x, y in zip([0, 1], ys):
        axin.text(x, y - 0.5, f"{y:.1f}%", ha="center", va="top", fontsize=6)
    axin.set_xticks([0, 1])
    axin.set_xticklabels(["100 K", "300 K"], fontsize=6)
    axin.tick_params(labelsize=6, length=2, pad=1.5)
    axin.axhline(0, color="k", lw=0.6)
    axin.set_ylim(min(ys) * 1.35, 2)
    axin.set_title(r"$\Delta V_{\mathrm{PI}}/V_{\mathrm{PI}}$ at $\beta=0.03$",
                   fontsize=6.2, pad=2)

    fig.tight_layout()
    return fs.savefig(fig, "fig3_lifshitz_shift")


# ----------------------------------------------------------------------------
# Fig. 4 -- (a) PINN training loss;  (b) fixed-step RK4 step study
# ----------------------------------------------------------------------------
def figure4():
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(fs.COL_DOUBLE, 2.7))

    # ---- (a) training loss curves --------------------------------------
    reg = [
        ("stable", "stable", fs.PALETTE["blue"], "o"),
        ("growing", "growing", fs.PALETTE["vermillion"], "s"),
        ("pullin", "pull-in", fs.PALETTE["green"], "^"),
    ]
    lbfgs_marks = []
    for tag, lab, c, mk in reg:
        with open(os.path.join(LOGS, f"history_{tag}.json")) as fh:
            h = json.load(fh)
        step = np.asarray(h["step"], float)
        tot = np.asarray(h["L_total"], float)
        phase = h["phase"]
        axa.semilogy(step, tot, "-", marker=mk, color=c, lw=1.2, ms=4,
                     label=lab)
        # first L-BFGS step (Adam -> L-BFGS handover)
        if "lbfgs" in phase:
            i = phase.index("lbfgs")
            lbfgs_marks.append(step[i])

    axa.grid(True, which="both")
    if lbfgs_marks:
        xm = float(np.mean(lbfgs_marks))
        axa.axvline(xm, ls=":", color=fs.PALETTE["black"], lw=0.9)
        axa.text(xm, axa.get_ylim()[1], " Adam$\\,\\to\\,$L-BFGS", fontsize=6.8,
                 va="top", ha="left", rotation=90, color="#333333")
    axa.set_xlabel("optimizer iteration")
    axa.set_ylabel(r"total loss  $\mathcal{L}_{\mathrm{total}}$")
    axa.set_title("(a)  PINN training", fontsize=8.5, loc="left")
    axa.legend(loc="upper right", fontsize=7.2)

    # ---- (b) fixed-step RK4 step study near pull-in --------------------
    rows = list(csv.DictReader(open(os.path.join(RESULTS, "rk4_stepstudy.csv"))))
    h = np.array([float(r["h"]) for r in rows])
    linf = np.array([float(r["linf_xi_[0,0.98tau*]"]) for r in rows])
    tau_err = np.array([float(r["tau_PI_err"]) for r in rows])

    axb.loglog(h, linf, "-o", color=fs.PALETTE["blue"], lw=1.2, ms=4,
               label=r"$L_\infty(\xi)$ on $[0,0.98\,\tau^{*}]$")
    axb.loglog(h, tau_err, "-s", color=fs.PALETTE["vermillion"], lw=1.2, ms=4,
               label=r"$|\tau_{\mathrm{PI}}-\tau^{*}|$")

    # 4th-order reference slope guide
    href = np.array([h.min(), h.max()])
    c4 = linf[np.argmax(h)] / h.max() ** 4
    axb.loglog(href, c4 * href ** 4, ":", color=fs.PALETTE["black"], lw=1.0,
               label=r"$\propto h^{4}$")

    axb.grid(True, which="both")
    axb.set_xlabel(r"fixed step size  $h$")
    axb.set_ylabel(r"error  (dimensionless)")
    axb.set_title("(b)  fixed-step RK4 near pull-in", fontsize=8.5, loc="left")
    axb.legend(loc="lower right", fontsize=6.8)
    # honest note: every RK4 run overshoots the pole (xi>1, unphysical)
    axb.text(0.03, 0.05,
             "all $h$: RK4 overshoots the\npole ($\\xi>1$, unphysical)",
             transform=axb.transAxes, fontsize=6.5, va="bottom", ha="left",
             color=fs.PALETTE["vermillion"])

    fig.tight_layout()
    return fs.savefig(fig, "fig4_diagnostics")


def main():
    fs.apply_style()
    written = []
    written += figure1()
    written += figure2()
    written += figure3()
    written += figure4()

    print("\nGenerated figures (figures/):")
    for p in written:
        size = os.path.getsize(p)
        print(f"  {os.path.basename(p):32s} {size/1024:8.1f} KiB   {p}")


if __name__ == "__main__":
    main()
