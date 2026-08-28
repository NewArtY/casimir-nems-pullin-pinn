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
    """Trajectories (top row) over their own pointwise error (bottom row).

    The error used to live in an inset floating over the trajectory; giving it
    a dedicated, x-shared strip under each panel removes every overlap and
    lets the three errors be compared on one common decade range.
    """
    regimes = [
        ("stable", "stable", "traj_stable.npz"),
        ("growing", "growing", "traj_growing.npz"),
        ("pullin", "pull-in", "traj_pullin.npz"),
    ]
    # tau_PI for the pull-in case comes from the DOP853 reference verification.
    with open(os.path.join(RESULTS, "verification_summary.json")) as fh:
        tau_pi = json.load(fh)["reference"]["tau_star"]

    ref_c = fs.PALETTE["black"]
    pinn_c = fs.PALETTE["vermillion"]
    err_c = fs.PALETTE["green"]

    data = [_load(f) for _, _, f in regimes]
    errs = [np.abs(d["xi_pinn"] - d["xi_ref"]) for d in data]
    # common error decades, set by the data: top decade above the worst
    # deviation, five decades of dynamic range below it.
    e_hi = 10.0 ** np.ceil(np.log10(max(e.max() for e in errs)))
    e_lo = e_hi * 1e-5

    fig, axes = plt.subplots(2, 3, figsize=(fs.COL_DOUBLE, 3.55), sharex="col",
                             gridspec_kw={"height_ratios": [2.4, 1.0]})
    labels = ["(a)", "(b)", "(c)"]
    # anchor of the parameter block inside each panel, chosen to sit in the
    # empty half of that particular trajectory (axes fractions).
    ptxt_pos = {"stable": (0.34, 0.62, "left", "center"),
                "growing": (0.60, 0.34, "left", "center"),
                "pullin": (0.05, 0.74, "left", "top")}

    for k, ((key, title, _f), lab) in enumerate(zip(regimes, labels)):
        ax, axe = axes[0, k], axes[1, k]
        d, err = data[k], errs[k]
        t, xr, xp = d["t"], d["xi_ref"], d["xi_pinn"]
        a, b, z = float(d["alpha"]), float(d["beta"]), float(d["zeta"])

        # In the pull-in panel the stored arrays stop at the PINN training
        # horizon 0.98 tau_*, so the reference is redrawn from the separately
        # integrated run that goes all the way to the barrier; the network is
        # still drawn only where it was trained.
        if key == "pullin":
            full = _load("traj_pullin_ref_full.npz")
            ax.plot(full["t"], full["xi"], "-", color=ref_c, lw=1.3, zorder=3)
        else:
            ax.plot(t, xr, "-", color=ref_c, lw=1.3, zorder=3)
        ax.plot(t, xp, "--", color=pinn_c, lw=1.3, zorder=4)
        # sparse open markers to show PINN is a distinct solution
        step = max(1, len(t) // 12)
        ax.plot(t[::step], xp[::step], "o", mfc="none", mec=pinn_c,
                mew=0.9, ms=3.5, zorder=5)

        ax.grid(True)
        ax.margins(x=0.02, y=0.08)
        ax.set_title(f"{lab}  {title}", fontsize=8.5, loc="left")

        ptxt = (rf"$\alpha={a:.3g}$" "\n"
                rf"$\beta={b:.3g}$" "\n"
                rf"$\zeta={z:.2g}$")
        px, py, pha, pva = ptxt_pos[key]
        ax.text(px, py, ptxt, transform=ax.transAxes, va=pva, ha=pha,
                fontsize=7.5, linespacing=1.35)

        if key == "pullin":
            # the barrier xi = 1 is a reference level, not a data curve
            ax.axhline(1.0, ls="-", color=fs.PALETTE["grey"], lw=0.9, zorder=1)
            ax.text(0.05, 1.0, r"pull-in barrier  $\xi=1$", fontsize=6.8,
                    transform=ax.get_yaxis_transform(), ha="left", va="bottom",
                    color="#555555")
            ax.axvline(tau_pi, ls=":", color=fs.PALETTE["blue"], lw=1.1, zorder=2)
            ax.text(tau_pi, 0.06, r"$\tau_{\mathrm{PI}}$", color=fs.PALETTE["blue"],
                    fontsize=7.5, ha="right", va="bottom", rotation=90)
            ax.set_ylim(0.0, 1.12)
            ax.set_xlim(0.0, tau_pi * 1.02)
            axe.axvline(tau_pi, ls=":", color=fs.PALETTE["blue"], lw=1.1,
                        zorder=2)
        else:
            ax.set_ylim(bottom=0.0)

        # ---- error strip, sharing the x axis of the panel above ---------
        axe.semilogy(t, np.maximum(err, e_lo), "-", color=err_c, lw=0.8)
        axe.set_ylim(e_lo, e_hi)
        axe.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=4))
        axe.yaxis.set_minor_locator(mticker.NullLocator())
        axe.grid(True, which="major")
        axe.set_xlabel(r"scaled time  $\tau$")

    axes[0, 0].set_ylabel(r"displacement  $\xi(\tau)=x/d$")
    axes[1, 0].set_ylabel(r"$|\xi_{\mathrm{PINN}}-\xi_{\mathrm{ref}}|$",
                          fontsize=7.5)

    # single legend (top, spanning)
    handles = [
        Line2D([], [], color=ref_c, lw=1.3, ls="-", label="reference (DOP853)"),
        Line2D([], [], color=pinn_c, lw=1.3, ls="--", marker="o", mfc="none",
               label="PINN"),
        Line2D([], [], color=err_c, lw=0.8, ls="-", label="PINN error"),
        Line2D([], [], color=fs.PALETTE["blue"], lw=1.1, ls=":",
               label=r"$\tau_{\mathrm{PI}}$ (pull-in time)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.015), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.935), h_pad=0.6)
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

    fig, ax = plt.subplots(figsize=(fs.COL_SINGLE, 3.15))

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

    # endpoint markers.  clip_on=False keeps the stars whole where they sit
    # exactly on the beta = 0 and alpha = 0 spines.
    ax.plot([alpha_c0], [0.0], "*", color=fs.PALETTE["orange"], ms=9,
            mec="k", mew=0.4, zorder=9, clip_on=False)
    ax.plot([0.0], [beta_star], "*", color=fs.PALETTE["orange"], ms=9,
            mec="k", mew=0.4, zorder=9, clip_on=False)
    # Endpoint callouts: short leaders only.  The alpha_c label sits inside the
    # stable grey wedge, the beta* label on the mid-viridis green above the
    # boundary; black type clears 7:1 contrast on both.
    ax.annotate(r"$\alpha_c=4/27$", xy=(alpha_c0, 0.0),
                xytext=(0.116, 0.0080), ha="right", va="center", fontsize=7.5,
                zorder=9,
                arrowprops=dict(arrowstyle="->", lw=0.7,
                                shrinkA=2.0, shrinkB=3.0))
    ax.annotate(rf"$\beta^{{*}}={beta_star:.4f}$", xy=(0.0, beta_star),
                xytext=(0.030, beta_star - 0.0062), ha="left", va="center",
                fontsize=7.5, zorder=9,
                arrowprops=dict(arrowstyle="->", lw=0.7,
                                shrinkA=2.0, shrinkB=3.0))

    # region labels, both placed clear of the three boundary lines
    ax.text(0.150, 0.0790, "PULL-IN", color="w", fontsize=8, fontweight="bold",
            ha="center", va="center")
    ax.text(0.150, 0.0715, r"($\tau_{\mathrm{PI}}$ map, $\zeta=0.1$)",
            color="w", fontsize=6.6, ha="center", va="center")
    ax.text(0.042, 0.0200, "STABLE", color="#444444", fontsize=8,
            fontweight="bold", ha="center", va="center")

    ax.set_xlim(ag.min(), ag.max())
    ax.set_ylim(bg.min(), bg.max())
    ax.set_xlabel(r"electrostatic number  $\alpha \propto V^{2}$")
    ax.set_ylabel(r"Casimir number  $\beta$")

    leg_handles = [
        Line2D([], [], color=fs.PALETTE["black"], lw=1.4, ls="-",
               label="static fold (analytic)"),
        Line2D([], [], color=fs.PALETTE["purple"], lw=0, marker="s", ms=3.0,
               mec="k", mew=0.3, label="PINN boundary"),
        Line2D([], [], color=fs.PALETTE["vermillion"], lw=1.3, ls="--",
               label=rf"dynamic, $\zeta={zeta:g}$"),
        Line2D([], [], color=fs.PALETTE["blue"], lw=1.2, ls=":",
               label=rf"dynamic, $\zeta={zeta_cmp:g}$"),
    ]
    # The legend lives outside the axes: inside, it can only sit on the
    # boundaries it is describing.
    ax.legend(handles=leg_handles, loc="lower center",
              bbox_to_anchor=(0.5, 1.005), ncol=2, fontsize=6.6,
              frameon=False, labelspacing=0.3, columnspacing=1.2,
              handlelength=1.9, borderaxespad=0.0)

    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fs.savefig(fig, "fig2_phase_diagram")


# ----------------------------------------------------------------------------
# Fig. 3 -- Lifshitz thermal shift of the pull-in boundary
# ----------------------------------------------------------------------------
def figure3():
    """Thermal (Lifshitz) correction to the pull-in boundary.

    (a) the T = 0 fold in the (alpha,beta) plane with the four device points
        and a zoom inset on the beta-axis intercept, the only place where the
        three temperatures are separated by more than the line width;
    (b) the induced relative shift of the pull-in voltage versus beta, which
        is what the corrected Lifshitz coefficient actually changes.
    """
    L = _load("lifshitz_boundaries.npz")
    with open(os.path.join(RESULTS, "device_table.json")) as fh:
        devs = json.load(fh)["devices"]

    tags = ("T0", "T100", "T300")
    col = {"T0": fs.PALETTE["blue"], "T100": fs.PALETTE["vermillion"],
           "T300": fs.PALETTE["green"]}
    lsty = {"T0": "-", "T100": "--", "T300": "-."}
    tlab = {"T0": "0 K", "T100": "100 K", "T300": "300 K"}

    a0, b0 = L["alpha_c_T0"], L["beta_c_T0"]
    bstar = {t: float(L["beta_star_" + t]) for t in tags}

    def branch(tag):
        """Fold branch of one temperature, closed on its exact intercept."""
        return (np.append(L["alpha_c_" + tag], 0.0),
                np.append(L["beta_c_" + tag], float(L["beta_star_" + tag])))

    d_nm = float(L["d"]) * 1e9
    alpha_c0 = 4.0 / 27.0                      # beta = 0 endpoint, T-independent

    fig, (axa, axb) = plt.subplots(2, 1, figsize=(fs.COL_SINGLE, 5.0))

    # ---- (a) T = 0 pull-in boundary in the (alpha,beta) plane ----------
    axa.grid(True)
    axa.plot(*branch("T0"), color=col["T0"], ls="-", lw=1.5, zorder=4,
             label=r"static fold, $T=0$")

    # device operating points A-D: each device sits at its beta on the fold,
    # alpha = alpha_c(beta) from device_table.  Label offsets (data units)
    # keep the two nearly degenerate points A and D readable.
    off = {"A": (0.013, -0.0050), "B": (0.008, 0.0016),
           "C": (0.000, 0.0115), "D": (-0.011, 0.0036)}
    da = np.array([dv["alpha_c_T0"] for dv in devs])
    db = np.array([dv["beta"] for dv in devs])
    axa.plot(da, db, "D", color=fs.PALETTE["black"], ms=4.2, mec="w", mew=0.6,
             zorder=7, label="device points")
    for dv in devs:
        dx, dy = off[dv["name"]]
        axa.annotate(dv["name"], xy=(dv["alpha_c_T0"], dv["beta"]),
                     xytext=(dv["alpha_c_T0"] + dx, dv["beta"] + dy),
                     fontsize=7.5, fontweight="bold", color=fs.PALETTE["black"],
                     ha="center", va="center", zorder=8)

    # beta = 0 endpoint: purely electrostatic, hence temperature independent.
    # Device C (beta = 8e-5) is drawn on top of this marker, so the star is
    # kept large enough to stay visible underneath the diamond.
    axa.plot([alpha_c0], [0.0], "*", color=fs.PALETTE["orange"], ms=11,
             mec="k", mew=0.4, zorder=5)
    axa.text(0.1560, 0.0005, r"$\alpha_c=4/27$" "\n" r"($T$-indep.)",
             ha="left", va="bottom", fontsize=6.8)

    axa.set_xlim(-0.005, 0.198)
    axa.set_ylim(-0.0065, 0.094)
    axa.set_xlabel(r"electrostatic number  $\alpha \propto V^{2}$")
    axa.set_ylabel(r"Casimir number  $\beta$")
    axa.set_title(r"(a)  pull-in boundary and device points", fontsize=8.5,
                  loc="left")
    axa.legend(loc="lower left", fontsize=6.8, borderpad=0.3,
               handletextpad=0.5)

    # zoom inset on the beta-axis intercept beta*(T): the three temperatures
    # are resolved only here (0.0819 -> 0.0799 between 0 K and 300 K).
    axin = axa.inset_axes([0.455, 0.455, 0.51, 0.47])
    for t in tags:
        axin.plot(*branch(t), color=col[t], ls=lsty[t], lw=1.1, label=tlab[t])
        axin.plot([0.0], [bstar[t]], "o", color=col[t], ms=3.4, mec="k",
                  mew=0.3, zorder=5)
    axin.set_xlim(-0.0004, 0.0062)
    axin.set_ylim(0.0788, 0.0827)
    axin.set_xticks([0.0, 0.003, 0.006])
    axin.set_yticks([0.080, 0.081, 0.082])
    axin.tick_params(labelsize=6.0, length=2, pad=1.5)
    axin.grid(True, alpha=0.25)
    axin.legend(loc="upper right", fontsize=6.0, labelspacing=0.15,
                handlelength=2.4, handletextpad=0.4, borderpad=0.2,
                borderaxespad=0.25)
    axin.set_title(r"zoom: $\beta^{*}(T)$", fontsize=6.5, pad=2)
    axa.indicate_inset_zoom(axin, edgecolor="#555555", lw=0.6, alpha=0.8)

    # ---- (b) relative pull-in voltage shift versus beta ----------------
    def rel_shift(tag):
        """-dV_PI/V_PI in percent on the beta support of the T-boundary."""
        b, a = L["beta_c_" + tag], L["alpha_c_" + tag]
        g = np.linspace(0.0, float(b.max()), 500)
        s = np.sqrt(np.interp(g, b, a) / np.interp(g, b0, a0)) - 1.0
        # V_PI vanishes exactly at beta*(T): close the curve on that pole
        return np.append(g, bstar[tag]), np.append(-100.0 * s, 100.0)

    axb.grid(True, which="both")
    curves = {}
    for t in ("T100", "T300"):
        g, y = rel_shift(t)
        curves[t] = (g, y)
        axb.semilogy(g, y, lsty[t], color=col[t], lw=1.4,
                     label=rf"$T={tlab[t].split()[0]}$ K")

    # numeric callout at beta = 0.03 (quoted in the text)
    b_ref = 0.03
    y_ref = {}
    for t in ("T100", "T300"):
        g, y = curves[t]
        y_ref[t] = float(np.interp(b_ref, g, y))
    axb.plot([b_ref, b_ref], [1e-3, y_ref["T300"]], ":", color="#555555",
             lw=0.8, zorder=2)
    for t in ("T100", "T300"):
        axb.plot([b_ref], [y_ref[t]], "o", color=col[t], ms=4, mec="k",
                 mew=0.3, zorder=6)
    # the numbers themselves go into the empty upper-left corner
    # The axis already carries the minus sign, so the callout quotes the
    # magnitude: printing "Delta V/V = -0.24%" beside an axis labelled
    # "-Delta V/V" makes a reader resolve a double negative.
    axb.text(0.045, 0.760, r"voltage reduction at $\beta=0.03$:",
             transform=axb.transAxes,
             fontsize=6.6, ha="left", va="center", color="#333333")
    for k, t in enumerate(("T100", "T300")):
        axb.text(0.045, 0.700 - 0.058 * k,
                 rf"${y_ref[t]:.2f}$%   ({tlab[t]})",
                 transform=axb.transAxes, fontsize=6.6, ha="left", va="center",
                 color=col[t])

    # device beta coordinates as a rug along the top axis
    trb = axb.get_xaxis_transform()
    lab_ha = {"A": "right", "D": "left", "B": "center", "C": "left"}
    lab_dx = {"A": -0.0012, "D": 0.0012, "B": 0.0, "C": 0.0014}
    for dv in devs:
        nm = dv["name"]
        axb.plot([dv["beta"]] * 2, [0.935, 1.0], "-", color=fs.PALETTE["black"],
                 lw=0.8, transform=trb, clip_on=False, zorder=6)
        axb.text(dv["beta"] + lab_dx[nm], 0.875, nm, transform=trb,
                 fontsize=7, fontweight="bold", ha=lab_ha[nm], va="center")

    ymax = max(curves["T100"][1].max(), curves["T300"][1].max())
    ylo = 10.0 ** np.floor(np.log10(np.interp(0.05 * bstar["T0"],
                                              *curves["T100"])))
    axb.set_xlim(0.0, bstar["T0"])
    axb.set_ylim(ylo, 3.0 * ymax)
    axb.set_xlabel(r"Casimir number  $\beta$")
    axb.set_ylabel(r"$-\Delta V_{\mathrm{PI}}/V_{\mathrm{PI}}$  (%)")
    axb.set_title(r"(b)  thermal shift of the pull-in voltage", fontsize=8.5,
                  loc="left")
    axb.text(0.052, 0.30 * ymax, r"$V_{\mathrm{PI}}\!\to\!0$" "\n"
             r"at $\beta^{*}(T)$", fontsize=6.4, ha="left", va="center",
             color="#333333")
    # the electrostatic limit carries no temperature dependence at all
    axb.text(0.045, 0.585, r"both curves $\to 0$ as $\beta \to 0$",
             transform=axb.transAxes, fontsize=6.4, ha="left", va="center",
             color="#555555")
    axb.legend(loc="lower right", fontsize=6.8, title=rf"$d={d_nm:.0f}$ nm",
               title_fontsize=6.8, borderpad=0.3, handletextpad=0.5)

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
    # The three runs hand over to L-BFGS at DIFFERENT iterations (config
    # "adam_iters": 1500 / 2000 / 3000), so the handover cannot be one shared
    # vertical line: each curve carries its own tick, in its own colour, at its
    # own last Adam step.
    handles_a = []
    for tag, lab, c, mk in reg:
        with open(os.path.join(LOGS, f"history_{tag}.json")) as fh:
            h = json.load(fh)
        step = np.asarray(h["step"], float)
        tot = np.asarray(h["L_total"], float)
        phase = h["phase"]
        n_adam = int(h["config"]["adam_iters"])
        axa.semilogy(step, tot, "-", marker=mk, color=c, lw=1.2, ms=4,
                     zorder=4)
        # handover tick: last Adam iterate, i.e. the point the optimizer was
        # switched at.  A tall bar marker stays visible over the line marker.
        i_last = max(i for i, p in enumerate(phase) if p == "adam")
        axa.plot([step[i_last]], [tot[i_last]], marker="|", ms=12, mew=1.6,
                 color=c, zorder=6)
        handles_a.append(Line2D([], [], color=c, lw=1.2, marker=mk, ms=4,
                                label=rf"{lab}  ($n_{{\mathrm{{Adam}}}}="
                                      rf"{n_adam:d}$)"))

    handles_a.append(Line2D([], [], color=fs.PALETTE["black"], lw=0,
                            marker="|", ms=9, mew=1.6,
                            label=r"Adam$\,\to\,$L-BFGS handover"))

    axa.grid(True, which="major")
    axa.grid(True, which="minor", alpha=0.10, lw=0.4)
    axa.set_xlabel("optimizer iteration")
    axa.set_ylabel(r"total loss  $\mathcal{L}_{\mathrm{total}}$")
    axa.set_title("(a)  PINN training", fontsize=8.5, loc="left")
    axa.legend(handles=handles_a, loc="upper right", fontsize=6.6,
               labelspacing=0.28, handletextpad=0.6)

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

    axb.grid(True, which="major")
    axb.grid(True, which="minor", alpha=0.10, lw=0.4)
    axb.set_xlabel(r"fixed step size  $h$")
    axb.set_ylabel(r"error  (dimensionless)")
    axb.set_title("(b)  fixed-step RK4 near pull-in", fontsize=8.5, loc="left")
    axb.legend(loc="lower right", fontsize=6.8)
    # Honest note: every RK4 run overshoots the pole (xi>1, unphysical).  It is
    # a statement about all step sizes, not about one series, so it is set in
    # neutral ink and parked in the empty band between the two curves.
    axb.text(0.04, 0.60,
             "all $h$: RK4 overshoots the\npole ($\\xi>1$, unphysical)",
             transform=axb.transAxes, fontsize=6.5, va="center", ha="left",
             color="#333333")

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
