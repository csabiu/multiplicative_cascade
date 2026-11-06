"""Exact digital genus of the 2D conservative (random-permutation) cascade.

Implements Result 1 of Sabiu, "Exact topology of conservative multiplicative
cascades" (J. Phys. A, JPhysA-125505): the ensemble-mean bulk digital Euler
density of the level-j excursion set {rho > nu} under the (4,8) adjacency
convention (4-connected foreground, 8-connected background),

    chi_j / A = p1 - pEh - pEv + pF ,

as an exact finite recursion over the multinomial atoms of the log-density
(no grid, no interpolation, no simulation), together with the matching
measurement on simulated fields and the finite-grid conversion.

    >>> from cascade_genus import genus_theory, genus_measured
    >>> th = genus_theory([0.42, 0.30, 0.20, 0.08], J=8, thresholds=[0.0, 1.0])
    >>> th['chi']

Conventions: J = depth = total number of multiplication steps (the field has
2^J cells per side); `multifrac(..., levels=J-1)` generates the same field.
"""
import itertools
from math import factorial
import numpy as np

N = 4                     # b=2, d=2 children per parent
_DEC = 9                  # rounding of log-density keys in the memo


def log_increments(weights):
    w = np.asarray(weights, float)
    w = w / w.sum()
    return np.log(N * w)   # c_i = ln(b^d p_i); -inf for a zero weight


def atoms(C, m):
    """Exact (position, probability) atoms of the m-level log-density."""
    pos, pr, fm = [], [], factorial(m)
    for a in range(m + 1):
        for b in range(m - a + 1):
            for c in range(m - a - b + 1):
                d = m - a - b - c
                pos.append(a * C[0] + b * C[1] + c * C[2] + d * C[3])
                pr.append(fm / (factorial(a) * factorial(b) * factorial(c) * factorial(d)) / N ** m)
    return np.array(pos), np.array(pr)


class GenusTheory:
    """Exact evaluator of p1, pE (=pEh=pEv), pF and chi/A at depth J."""

    def __init__(self, weights, J):
        self.C = tuple(log_increments(weights))
        self.J = J
        self.ac = {m: atoms(np.array(self.C), m) for m in range(J + 1)}

    def __call__(self, t):
        """Evaluate at log-density threshold t (= ln nu, unit-mean density)."""
        C, ac, J = self.C, self.ac, self.J
        memo = {}

        def ev(f, x):                       # f: functional tree, x: log-density
            key = (f, round(x, _DEC))
            if key in memo:
                return memo[key]
            op = f[0]
            if op == 'psi':
                v = 1.0 if x > t else 0.0
            elif op == 'S':                 # m-level descendant average
                m, g = f[1], f[2]
                pos, pr = ac[m]
                v = sum(p * ev(g, x + a) for a, p in zip(pos, pr))
            elif op == 'P1':                # one child, Psi_1
                v = sum(ev(f[1], x + c) for c in C) / N
            elif op == 'P2':                # sibling pair without replacement, Psi_2
                s = [ev(f[1], x + c) for c in C]
                S = sum(s)
                v = (S * S - sum(u * u for u in s)) / (N * (N - 1))
            elif op == 'F4':                # all four children, Phi_4
                v = 1.0
                for c in C:
                    v *= ev(f[1], x + c)
            memo[key] = v
            return v

        def inner(k, g):
            pos, pr = ac[k]
            return sum(p * ev(g, a) for a, p in zip(pos, pr))

        def G(j, g):                        # adjacent pair, eq. (6)
            dn = 2 ** j - 1
            return sum((2 ** k / dn) * inner(k, ('P2', ('S', j - 1 - k, g) if j - 1 - k > 0 else g))
                       for k in range(j))

        def F(j, g):                        # 2x2 plaquette, eq. (8)
            if j == 1:
                v = 1.0
                for c in C:
                    v *= ev(g, c)
                return v
            L = 2 ** j
            pe = (L // 2) / (L - 1)
            po = (L // 2 - 1) / (L - 1)
            return pe * pe * inner(j - 1, ('F4', g)) + 2 * pe * po * G(j - 1, ('P2', g)) + po * po * F(j - 1, ('P1', g))

        posJ, prJ = ac[J]
        p1 = float(prJ[posJ > t].sum())
        pE = G(J, ('psi',))
        pF = F(J, ('psi',))
        return p1, pE, pF, p1 - 2 * pE + pF

    def gap_thresholds(self, lo=-np.inf, hi=np.inf, n_max=None):
        """Midpoints between consecutive distinct atoms of the level-J law."""
        u = np.unique(np.round(np.sort(self.ac[self.J][0]), _DEC))
        mids = 0.5 * (u[:-1] + u[1:])
        mids = mids[(mids >= lo) & (mids <= hi)]
        if n_max is not None and len(mids) > n_max:
            mids = mids[np.linspace(0, len(mids) - 1, n_max).round().astype(int)]
        return mids


def genus_theory(weights, J, thresholds):
    """Exact p1, pE, pF, chi (bulk densities) at the given ln(nu) thresholds."""
    th = GenusTheory(weights, J)
    out = np.array([th(t) for t in np.atleast_1d(thresholds)])
    return dict(p1=out[:, 0], pE=out[:, 1], pF=out[:, 2], chi=out[:, 3])


def finite_grid_chi(p1, pE, pF, Nside):
    """E[chi_{NxN}]/N^2 from the bulk densities, eq. (5): boundary factors do not cancel."""
    f = 1.0 - 1.0 / Nside
    return p1 - 2 * f * pE + f * f * pF


def euler_counts(on):
    """(V, E, F) counts of a boolean 2D array under the (4,8) convention."""
    V = int(on.sum())
    E = int((on[:, :-1] & on[:, 1:]).sum() + (on[:-1, :] & on[1:, :]).sum())
    F = int((on[:-1, :-1] & on[:-1, 1:] & on[1:, :-1] & on[1:, 1:]).sum())
    return V, E, F


def euler_characteristic(on):
    """Digital Euler characteristic, (4,8) convention: two diagonal pixels give 2."""
    V, E, F = euler_counts(on)
    return V - E + F


def bulk_densities(on):
    """Per-configuration (bulk) p1, pEh, pEv, pF of a boolean array."""
    p1 = on.mean()
    pEh = (on[:, :-1] & on[:, 1:]).mean()
    pEv = (on[:-1, :] & on[1:, :]).mean()
    pF = (on[:-1, :-1] & on[:-1, 1:] & on[1:, :-1] & on[1:, 1:]).mean()
    return p1, pEh, pEv, pF


def genus_measured(weights, J, thresholds, nreal=120, seed0=1, generator=None):
    """Bulk p1, pE, pF, chi (mean and standard error over nreal cascade realizations)."""
    if generator is None:
        import os, sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
        from multifrac import multifrac
        def generator(seed):
            f, _ = multifrac(weights, dim=2, size=2, levels=J - 1, verbose=False, seed=seed)
            return f
    thresholds = np.atleast_1d(thresholds)
    acc = np.empty((nreal, len(thresholds), 4))
    for r in range(nreal):
        f = generator(seed0 + r)
        lr = np.log(f / f.mean())
        for i, t in enumerate(thresholds):
            p1, pEh, pEv, pF = bulk_densities(lr > t)
            acc[r, i] = (p1, 0.5 * (pEh + pEv), pF, p1 - pEh - pEv + pF)
    mean = acc.mean(0)
    se = acc.std(0, ddof=1) / np.sqrt(nreal) if nreal > 1 else np.zeros_like(mean)
    keys = ('p1', 'pE', 'pF', 'chi')
    out = {k: mean[:, i] for i, k in enumerate(keys)}
    out.update({k + 'e': se[:, i] for i, k in enumerate(keys)})
    return out


def enumerate_depth2(weights, thresholds):
    """Exhaustive ensemble mean at J=2 over all 24^5 permutation configurations."""
    w = np.asarray(weights, float)
    w = w / w.sum()
    perms = np.array(list(itertools.permutations(range(4))))
    P = perms.shape[0]
    idx = np.array(list(itertools.product(range(P), repeat=4)))
    res = np.zeros((len(thresholds), 4))
    for a in range(P):
        rootw = w[perms[a]]
        field = np.empty((len(idx), 4, 4))
        for k in range(4):
            cw = w[perms[idx[:, k]]]
            r0, c0 = 2 * (k // 2), 2 * (k % 2)
            field[:, r0:r0 + 2, c0:c0 + 2] = (rootw[k] * cw).reshape(-1, 2, 2)
        with np.errstate(divide='ignore'):
            lr = np.log(field * 16.0)
        for i, t in enumerate(thresholds):
            on = lr > t
            p1 = on.mean(axis=(1, 2))
            pEh = (on[:, :, :-1] & on[:, :, 1:]).mean(axis=(1, 2))
            pEv = (on[:, :-1, :] & on[:, 1:, :]).mean(axis=(1, 2))
            pF = (on[:, :-1, :-1] & on[:, :-1, 1:] & on[:, 1:, :-1] & on[:, 1:, 1:]).mean(axis=(1, 2))
            res[i] += (p1.sum(), 0.5 * (pEh + pEv).sum(), pF.sum(), (p1 - pEh - pEv + pF).sum())
    res /= P ** 5
    return dict(p1=res[:, 0], pE=res[:, 1], pF=res[:, 2], chi=res[:, 3])


if __name__ == '__main__':
    import time
    w = [0.42, 0.30, 0.20, 0.08]
    # convention check
    d = np.zeros((3, 3), bool); d[0, 0] = d[1, 1] = True
    print('two diagonal pixels: chi =', euler_characteristic(d), '(4,8 convention gives 2)')
    th = GenusTheory(w, 2)
    thr = th.gap_thresholds()
    t0 = time.time()
    ex = genus_theory(w, 2, thr)
    en = enumerate_depth2(w, thr)
    for k in ('p1', 'pE', 'pF', 'chi'):
        print(f'J=2 exhaustive enumeration vs exact recursion, {k}: max|diff| = {np.abs(ex[k]-en[k]).max():.1e}')
    print(f'[{time.time()-t0:.0f}s]')
