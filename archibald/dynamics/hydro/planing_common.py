"""
Outils communs aux méthodes de pré-planing / planing.

AUCUNE résolution ici : toutes les méthodes sont évaluées en boucle ouverte à
l'assiette (trim, enfoncement, gîte...) fournie par les propriétés hydrostatiques ;
l'équilibre est fermé à l'extérieur (IPOPT).

Conventions (identiques à dsyhs.py / holtrop.py) :
  - SI partout, `stw` en noeuds en entrée des compute_R*.
  - Toutes les fonctions sont lisses et n'utilisent que des opérations
    disponibles dans archibald.numpy (compatibles CasADi, déroulées).
"""
import archibald.numpy as np
import archibald.toolbox.units as u
from archibald.dynamics.hydro.common import Cf_hull


# ------------------------------------------------------------------ Froude
def speed_ms(stw):
    return stw * u.kt


def Fn_vol(V, volume, g=9.81):
    """Froude volumique  V / sqrt(g ∇^(1/3))."""
    return V / np.sqrt(g * np.softplus(volume, beta=1e3) ** (1. / 3.))


def Cv_beam(V, b, g=9.81):
    """Coefficient de vitesse de Savitsky  V / sqrt(g b)."""
    return V / np.sqrt(g * b)


def Fn_length(V, L, g=9.81):
    return V / np.sqrt(g * L)


# --------------------------------------------------------- fonctions lisses
def smoothstep(x, x0, x1):
    """0 pour x<=x0, 1 pour x>=x1, raccord C1 (3t²-2t³)."""
    t = np.fmax(0., np.fmin(1., (x - x0) / (x1 - x0)))
    return t * t * (3. - 2. * t)


def sigmoid(x):
    return 0.5 * (1. + np.tanh(0.5 * x))


def smooth_min(a, b, k):
    """min(a,b) lisse (log-sum-exp), k [1/unité] grand = plus net."""
    m = np.fmin(a, b)  # décalage pour la stabilité numérique
    return m - np.log(np.exp(-k * (a - m)) + np.exp(-k * (b - m))) / k


def blend(w, low, high):
    return (1. - w) * low + w * high


# ------------------------------------------------------------ frottement
def friction_coefficient(V, Lw, nu=1.19e-6, dCf=0.):
    """Cf ITTC57 sur une longueur mouillée Lw ; dCf = surcote de rugosité."""
    Re = np.softplus(V, beta=1e3) * Lw / nu
    return Cf_hull(Re + 10.) + dCf
