"""The zero-frequency bound on the thermal Casimir pressure of ideal mirrors.

Appendix A proves, for ideal mirrors, that the exact Matsubara pressure obeys

    |P(a, T)| <= |P_0(a)| + |P_cl(a, T)|,

where P_0 is the zero-temperature pressure and P_cl the n = 0 Matsubara term
(the classical limit).  In units of |P_0|, with tau = 4 pi a / lambda_T,

    R(tau) = |P| / |P_0| = tau [H(0)/2 + sum_{n>=1} H(n tau)] / I ,
    H(z)   = int_z^inf x^2/(e^x - 1) dx ,   I = int_0^inf H = pi^4/15 ,

and the bound reads R <= 1 + c(tau), c = 15 zeta(3) tau / pi^4, the factor
of Eq. (lifshitz) of the main text.  H is decreasing, so the sum over n >= 1
is at most the integral, which is the proof; this module evaluates R on a wide
range of tau as a check, together with the low-temperature asymptote
R - 1 ~ (16/3)(a/lambda_T)^4 of Eq. (A9).

Run:  python -m src.matsubara_bound
"""

from __future__ import annotations

import numpy as np

ZETA3 = 1.2020569031595942854
I_TOT = np.pi ** 4 / 15.0


def H(z):
    """int_z^inf x^2/(e^x-1) dx = sum_k e^{-kz} (z^2/k + 2z/k^2 + 2/k^3)."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    out = np.empty_like(z)
    for i, zi in enumerate(z):
        if zi == 0.0:
            out[i] = 2.0 * ZETA3
            continue
        K = int(np.ceil(45.0 / zi)) + 5
        k = np.arange(1, K + 1, dtype=float)
        e = np.exp(-k * zi)
        out[i] = np.sum(e * (zi ** 2 / k + 2 * zi / k ** 2 + 2 / k ** 3))
    return out


def ratio(tau):
    n_max = int(np.ceil(45.0 / tau)) + 5
    n = np.arange(1, n_max + 1, dtype=float)
    return tau * (0.5 * H(0.0)[0] + np.sum(H(n * tau))) / I_TOT


def main():
    print(f"{'a/lambda_T':>11} {'R-1':>12} {'c(tau)':>12} {'(R-1)/c':>10}"
          f" {'(16/3)(a/l)^4':>14}")
    worst = 0.0
    for tau in np.geomspace(0.05, 40.0, 25):
        R = ratio(tau)
        c = 15.0 * ZETA3 * tau / np.pi ** 4
        x = tau / (4 * np.pi)
        worst = max(worst, (R - 1) / c)
        print(f"{x:11.4e} {R - 1:12.4e} {c:12.4e} {(R - 1) / c:10.6f}"
              f" {16 / 3 * x ** 4:14.4e}")
    print(f"\nlargest (R-1)/c over the scan: {worst:.6f}  (bound: <= 1)")
    if worst > 1.0:
        raise SystemExit("bound violated")


if __name__ == "__main__":
    main()
