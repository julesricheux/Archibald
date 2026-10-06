"""
Radojcic et al. (2014) — résistance de coques à bouchain vif en pré-planing /
planing (séries USCG, TUNS), Fn_vol 0.6 - 3.5.  Polish Maritime Research 21(2).

Modèle « simple » : R/Δ|Δ=100000 lb = A Fn³ + B Fn² + C Fn + D, avec A..D
fonctions de L/∇^(1/3) ; modèle « complexe » : réseau de neurones / polynôme à
116 termes (L/∇^(1/3), L/B, LCG/L, beta).  Domaine : 3.9 <= L/∇^(1/3) <= 6.9,
2.5 <= L/B <= 4.7, 0.27 <= LCG/L <= 0.41, 12° <= beta <= 24°.
=> les dériveurs/skiffs (L/∇^(1/3) ~ 7.5-9) sont en EXTRAPOLATION.

!! Les coefficients (annexes 1-2) sont des images dans le PDF consulté : non
   transcrits ici.  Renseigner RADOJCIC_SIMPLE_COEFFS (fonctions de L/∇^(1/3))
   pour activer la méthode.  La correction d'échelle de frottement (Δ ≠ 100000 lb)
   est, elle, implémentée ci-dessous.
"""
import archibald.numpy as np
from .planing_common import speed_ms, Fn_vol, friction_coefficient

RADOJCIC_SIMPLE_COEFFS = None     # dict {"A": f(slender), "B": ..., "C": ..., "D": ...}
LB_PER_N = 1. / 4.4482216
DELTA_STD_N = 100000. * 4.4482216


def _R_over_Delta_std(Fn, slender):
    if RADOJCIC_SIMPLE_COEFFS is None:
        raise NotImplementedError("Coefficients de Radojcic (2014) non renseignés (annexe 1).")
    c = RADOJCIC_SIMPLE_COEFFS
    return c["A"](slender) * Fn ** 3 + c["B"](slender) * Fn ** 2 + c["C"](slender) * Fn + c["D"](slender)


def _geom(volume, Lwl, Aws, S_over_V23, Lk_over_L):
    # propriétés hydrostatiques à l'assiette courante (boucle ouverte)
    S23 = S_over_V23 if S_over_V23 is not None else Aws / volume ** (2. / 3.)
    return S23, (Lk_over_L if Lk_over_L is not None else 1.)


def compute_Rtotal_radojcic(stw, rho, g, nu, volume, Lwl, Aws=None, S_over_V23=None, Lk_over_L=None,
                            weight=None, **kw):
    """
    R total (N) = Δ [ (R/Δ)_std + correction de frottement géosim ].
    S_over_V23 (S/∇^(2/3)) et Lk_over_L : fournis (modèles de l'article ou CFD/géométrie).
    """
    S_over_V23, Lk_over_L = _geom(volume, Lwl, Aws, S_over_V23, Lk_over_L)
    V = speed_ms(stw)
    Fn = Fn_vol(V, volume, g)
    slender = Lwl / volume ** (1. / 3.)
    Delta = weight if weight is not None else rho * g * volume
    r_std = _R_over_Delta_std(Fn, slender)
    # carène géosim de référence (même Fn_vol, même forme) : Δ_std
    vol_std = DELTA_STD_N / (rho * g)
    lam = (volume / vol_std) ** (1. / 3.)
    V_std = V / np.sqrt(lam)
    S_std = S_over_V23 * vol_std ** (2. / 3.)
    Lw_std = Lk_over_L * Lwl / lam
    Rf_std = 0.5 * rho * friction_coefficient(V_std, Lw_std, nu) * S_std * V_std ** 2
    Rf = 0.5 * rho * friction_coefficient(V, Lk_over_L * Lwl, nu) * (S_over_V23 * volume ** (2. / 3.)) * V ** 2
    return Delta * ((r_std * DELTA_STD_N - Rf_std) / DELTA_STD_N) + Rf


def compute_Rf_radojcic(stw, rho, g, nu, volume, Lwl, Aws=None, S_over_V23=None, Lk_over_L=None, **kw):
    S_over_V23, Lk_over_L = _geom(volume, Lwl, Aws, S_over_V23, Lk_over_L)
    V = speed_ms(stw)
    return 0.5 * rho * friction_coefficient(V, Lk_over_L * Lwl, nu) * (S_over_V23 * volume ** (2. / 3.)) * V ** 2


def compute_Rw_radojcic(stw, rho, g, nu, volume, Lwl, Aws=None, S_over_V23=None, Lk_over_L=None, **kw):
    """Résiduaire = total - frottement (même décomposition que dsyhs)."""
    return (compute_Rtotal_radojcic(stw, rho, g, nu, volume, Lwl, Aws, S_over_V23, Lk_over_L, **kw)
            - compute_Rf_radojcic(stw, rho, g, nu, volume, Lwl, Aws, S_over_V23, Lk_over_L))
