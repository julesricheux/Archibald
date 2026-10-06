"""
Théorie des tranches 2D+T (Wagner / Zarnick) — BOUCLE OUVERTE.

Évalue, pour une assiette et un état cinématique donnés, les efforts
hydrodynamiques sur la coque. Aucune résolution interne (ni équilibre, ni
accélérations) : IPOPT / la dynamique globale assemblent les torseurs.

Chaque section (plan fixe dans l'eau) donne un effort vertical par unité de longueur
      f_hs  = rho g A(h) r_h(xi)                 flottabilité (relaxée près du tableau)
      f_dyn = m'(h) w_dot* + (dm'/dh) w²         masse ajoutée de Wagner (Zarnick 1978)
avec h = immersion de la quille, w = vitesse d'immersion relative, m' = ½ρπc²,
c = min(kf h/tan(beta), b/2) (lisse).  Traînée de pression = Σ f · pente locale.

Entrées : sections (prismatic_sections ou vos offsets : xi,b,beta,zk), xcg_tr, zcg_base,
état (zeta = hauteur du G / flottaison [m], trim [DEG] cabré >0, trim_rate [deg/s]), vitesses
(zeta_rate, trim_rate), U (stw en noeuds via compute_*).

Sorties (dict strip_wrench) :
    Fz_dyn, My_dyn   effort/moment dynamiques (SANS flottabilité), My au G, cabrant +
    Fz_hs,  My_hs    flottabilité cohérente avec ce modèle (à comparer à VOTRE hydrostatique ;
                     c'est l'un OU l'autre dans le bilan, pas les deux)
    Rp_dyn, Rp_hs    traînée de pression (dynamique / trim-drag hydrostatique)
    Rf, Sw           frottement et surface mouillée (ITTC57 sur longueur équivalente)
    A33, A35, A55    masses ajoutées : F = Fz_dyn - A33 zeta_acc - A35 trim_acc ;
                     M = My_dyn - A35 zeta_acc - A55 trim_acc  (à assembler en dynamique ; trim_acc en rad/s²)

!! Modèle de recherche : kf et l_relief sont des paramètres à CALIBRER ; pas de gerbes
   ni de chine mouillé fin. Voir validation_planing.py.
"""
import archibald.numpy as np
from .planing_common import (speed_ms, smoothstep, sigmoid, smooth_min,
                             friction_coefficient)


# ------------------------------------------------------------- géométrie
def prismatic_sections(L, b, beta_deg, n=30, rocker=0.):
    """Coque prismatique (xi depuis le tableau, b, beta, zk) ; rocker = flèche [m]."""
    xi = [L * i / (n - 1) for i in range(n)]
    zk = [rocker * (2. * x / L - 1.) ** 2 for x in xi]
    zk = [z - zk[0] for z in zk]
    return dict(xi=xi, b=[b] * n, beta=[beta_deg] * n, zk=zk)


def prepare_sections(sec):
    """Pré-calcule pentes, courbures et poids trapèze (constantes de géométrie)."""
    if "w" in sec:
        return sec
    xi, zk = sec["xi"], sec["zk"]
    n = len(xi)
    d1, d2, wt = [], [], []
    for i in range(n):
        lo, hi = max(i - 1, 0), min(i + 1, n - 1)
        d1.append((zk[hi] - zk[lo]) / (xi[hi] - xi[lo]))
        wt.append((xi[hi] - xi[lo]) / 2.)
    for i in range(n):
        lo, hi = max(i - 1, 0), min(i + 1, n - 1)
        d2.append((d1[hi] - d1[lo]) / (xi[hi] - xi[lo]))
    out = dict(sec)
    out.update(dzk=d1, d2zk=d2, w=wt)
    return out


# ---------------------------------------------------- section (par station)
def _section(h, b, beta_deg, rho, kf, k_sp=300.):
    tb = np.tan(beta_deg * np.pi / 180.)
    sb = np.sin(beta_deg * np.pi / 180.)
    hp = np.softplus(h, beta=k_sp)
    dhp = sigmoid(k_sp * h)
    hc = 0.5 * b * tb
    hq = smooth_min(hp, hc, 40. / b)
    A = hq ** 2 / tb + b * (hp - hq)
    P = 2. * hq / sb + 2. * (hp - hq)
    a = kf * hp / tb
    kc = 40. / b
    c = smooth_min(a, 0.5 * b, kc)
    wgt = sigmoid(kc * (0.5 * b - a))
    mp = 0.5 * rho * np.pi * c ** 2
    dmp = rho * np.pi * c * wgt * kf / tb * dhp
    return A, mp, dmp, P


# ---------------------------------------------------------------- torseur
def strip_wrench(stw, rho, g, nu, zeta, trim, sections, xcg_tr, zcg_base,
                 zeta_rate=0., trim_rate=0., kf=1.0, l_relief=0., dCf=0.):
    U = np.softplus(speed_ms(stw), beta=1e3) + 1e-6
    sec = prepare_sections(sections)
    trim = trim * (np.pi / 180.)             # ANGLES EN DEGRES en entrée (comme `ie`, savitsky)
    trim_rate = trim_rate * (np.pi / 180.)   # [deg/s] -> [rad/s]
    ct, st = np.cos(trim), np.sin(trim)
    Fd = Md = Fh = Mh = Rpd = Rph = 0.
    A33 = A35 = A55 = Sw = zf = 0.
    for i in range(len(sec["xi"])):
        xi, zk, dz, d2z, wt = sec["xi"][i], sec["zk"][i], sec["dzk"][i], sec["d2zk"][i], sec["w"][i]
        dxdxi = ct - dz * st
        a = (xi - xcg_tr) * ct - (zk - zcg_base) * st          # bras horizontal / G
        q = (xi - xcg_tr) * st + (zk - zcg_base) * ct
        zE = zeta + q
        s = (st + dz * ct) / dxdxi                             # pente locale de la quille
        w = -(zeta_rate + a * trim_rate) + U * s               # vitesse d'immersion relative
        wdot0 = q * trim_rate ** 2 + 2. * U * trim_rate - U ** 2 * d2z
        A, mp, dmp, P = _section(-zE, sec["b"][i], sec["beta"][i], rho, kf)
        rh = smoothstep(xi, 0., l_relief) if l_relief > 0 else 1.
        dx = wt * dxdxi
        fd = mp * wdot0 + dmp * w ** 2
        fh = rho * g * A * rh
        Fd += fd * dx; Md += fd * a * dx; Rpd += fd * s * dx
        Fh += fh * dx; Mh += fh * a * dx; Rph += fh * s * dx
        A33 += mp * dx; A35 += mp * a * dx; A55 += mp * a * a * dx
        Sw += P * dx; zf += P * dx * zE
    Lw = np.fmax(1e-3, Sw / sec["b"][0])
    Rf = 0.5 * rho * friction_coefficient(U, Lw, nu, dCf) * U ** 2 * Sw
    z_fr = zf / np.fmax(1e-9, Sw)
    Mf = -Rf * (zeta - z_fr)                                   # force aft sous le G : piqueur
    return dict(Fz_dyn=Fd, My_dyn=Md + Mf, Fz_hs=Fh, My_hs=Mh, Rp_dyn=Rpd, Rp_hs=Rph,
                Rf=Rf, Sw=Sw, A33=A33, A35=A35, A55=A55)


# ------------------------------------------------ interface type compute_R*
def _w(stw, rho, g, nu, kw):
    return strip_wrench(stw, rho, g, nu, kw["zeta"], kw["trim"], kw["strip_sections"],
                        kw["xcg_tr"], kw["zcg_base"], kw.get("zeta_rate", 0.),
                        kw.get("trim_rate", 0.), kw.get("kf", 1.0), kw.get("l_relief", 0.),
                        kw.get("dCf", 0.))


def compute_Rf_strip(stw, rho, g, nu, **kwargs):
    return _w(stw, rho, g, nu, kwargs)["Rf"]


def compute_Rw_strip(stw, rho, g, nu, **kwargs):
    """Pression : dynamique + trim-drag hydrostatique (hydrostatic_drag=False pour l'exclure)."""
    o = _w(stw, rho, g, nu, kwargs)
    return o["Rp_dyn"] + (o["Rp_hs"] if kwargs.get("hydrostatic_drag", True) else 0.)


def compute_Fz_strip(stw, rho, g, nu, **kwargs):
    """Effort vertical dynamique (sans flottabilité) ; include_hydrostatic=True pour l'inclure."""
    o = _w(stw, rho, g, nu, kwargs)
    return o["Fz_dyn"] + (o["Fz_hs"] if kwargs.get("include_hydrostatic", False) else 0.)


def compute_My_strip(stw, rho, g, nu, **kwargs):
    o = _w(stw, rho, g, nu, kwargs)
    return o["My_dyn"] + (o["My_hs"] if kwargs.get("include_hydrostatic", False) else 0.)
