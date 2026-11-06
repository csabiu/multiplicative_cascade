"""Exact 3D digital genus of the conservative octree cascade, (6,26) convention.

chi_j / V = p1 - 3 pE + 3 pF - pC, each term an exact finite recursion over the
multinomial atoms (8 categories). Companion of cascade_genus.py (2D); see the
appendix of the J. Phys. A paper.
"""
from math import factorial, comb
import numpy as np

N = 8
_DEC = 9


def log_increments(weights):
    w = np.asarray(weights, float)
    w = w / w.sum()
    return np.log(N * w)


def atoms(C, m):
    pos, pr, fm, counts = [], [], factorial(m), [0] * N

    def rec(i, left):
        if i == N - 1:
            counts[i] = left
            p = fm
            for c in counts:
                p //= factorial(c)
            pos.append(sum(c * ci for c, ci in zip(counts, C)))
            pr.append(p / N ** m)
            return
        for c in range(left + 1):
            counts[i] = c
            rec(i + 1, left - c)
    rec(0, m)
    return np.array(pos), np.array(pr)


def _e4(s):
    """Fourth elementary symmetric polynomial of the values s (Newton's identities)."""
    p = [sum(v ** k for v in s) for k in range(1, 5)]
    e1 = p[0]
    e2 = (e1 * p[0] - p[1]) / 2
    e3 = (e2 * p[0] - e1 * p[1] + p[2]) / 3
    e4 = (e3 * p[0] - e2 * p[1] + e1 * p[2] - p[3]) / 4
    return e4


class GenusTheory3D:
    def __init__(self, weights, J):
        self.C = tuple(log_increments(weights))
        self.J = J
        self.ac = {m: atoms(np.array(self.C), m) for m in range(J + 1)}

    def gap_thresholds(self, lo=-np.inf, hi=np.inf, n_max=None):
        u = np.unique(np.round(np.sort(self.ac[self.J][0]), _DEC))
        mids = 0.5 * (u[:-1] + u[1:])
        mids = mids[(mids >= lo) & (mids <= hi)]
        if n_max is not None and len(mids) > n_max:
            mids = mids[np.linspace(0, len(mids) - 1, n_max).round().astype(int)]
        return mids

    def __call__(self, t):
        C, ac, J = self.C, self.ac, self.J
        memo = {}

        def ev(f, x):
            key = (f, round(x, _DEC))
            if key in memo:
                return memo[key]
            op = f[0]
            if op == 'psi':
                v = 1.0 if x > t else 0.0
            elif op == 'S':
                pos, pr = ac[f[1]]
                v = sum(p * ev(f[2], x + a) for a, p in zip(pos, pr))
            elif op == 'P1':
                v = sum(ev(f[1], x + c) for c in C) / N
            elif op == 'P2':
                s = [ev(f[1], x + c) for c in C]
                S = sum(s)
                v = (S * S - sum(u * u for u in s)) / (N * (N - 1))
            elif op == 'P4':
                v = _e4([ev(f[1], x + c) for c in C]) / comb(N, 4)
            elif op == 'F8':
                v = 1.0
                for c in C:
                    v *= ev(f[1], x + c)
            memo[key] = v
            return v

        def inner(k, g):
            pos, pr = ac[k]
            return sum(p * ev(g, a) for a, p in zip(pos, pr))

        def wrapS(m, g):
            return ('S', m, g) if m > 0 else g

        def G(j, g):
            dn = 2 ** j - 1
            return sum((2 ** k / dn) * inner(k, ('P2', wrapS(j - 1 - k, g))) for k in range(j))

        def pe_po(j):
            L = 2 ** j
            return (L // 2) / (L - 1), (L // 2 - 1) / (L - 1)

        def F(j, g):
            if j == 1:
                return ev(('P4', g), 0.0)
            pe, po = pe_po(j)
            return pe * pe * inner(j - 1, ('P4', g)) + 2 * pe * po * G(j - 1, ('P2', g)) + po * po * F(j - 1, ('P1', g))

        def Cb(j, g):
            if j == 1:
                return ev(('F8', g), 0.0)
            pe, po = pe_po(j)
            return (pe ** 3 * inner(j - 1, ('F8', g)) + 3 * pe * pe * po * G(j - 1, ('P4', g))
                    + 3 * pe * po * po * F(j - 1, ('P2', g)) + po ** 3 * Cb(j - 1, ('P1', g)))

        posJ, prJ = ac[J]
        psi = ('psi',)
        p1 = float(prJ[posJ > t].sum())
        pE, pF, pC = G(J, psi), F(J, psi), Cb(J, psi)
        return p1, pE, pF, pC, p1 - 3 * pE + 3 * pF - pC


def chi626(on):
    """(6,26) digital Euler characteristic of a boolean 3D array."""
    n1 = on.sum()
    n2 = (on[:-1] & on[1:]).sum() + (on[:, :-1] & on[:, 1:]).sum() + (on[:, :, :-1] & on[:, :, 1:]).sum()
    n4 = ((on[:-1, :-1] & on[1:, :-1] & on[:-1, 1:] & on[1:, 1:]).sum()
          + (on[:-1, :, :-1] & on[1:, :, :-1] & on[:-1, :, 1:] & on[1:, :, 1:]).sum()
          + (on[:, :-1, :-1] & on[:, 1:, :-1] & on[:, :-1, 1:] & on[:, 1:, 1:]).sum())
    n8 = (on[:-1, :-1, :-1] & on[1:, :-1, :-1] & on[:-1, 1:, :-1] & on[:-1, :-1, 1:]
          & on[1:, 1:, :-1] & on[1:, :-1, 1:] & on[:-1, 1:, 1:] & on[1:, 1:, 1:]).sum()
    return int(n1 - n2 + n4 - n8)
