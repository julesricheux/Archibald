# -*- coding: utf-8 -*-
"""
SAVITSKY 1964 / SAVITSKY & BROWN 1976 PLANING METHOD FUNCTIONS -- DYNAMIC COMPONENTS ONLY

No hydrostatic component is computed here :
  - the buoyancy-like lift term of Savitsky's C_L0 (the one in lambda^2.5 / Cv^2, which carries g)
    is removed : only the dynamic lift  C_L0 = 0.012 tau^1.1 lambda^0.5  remains ;
  - the centre of pressure is the high speed (purely dynamic) limit  l_cp = 0.75 lambda b.
The hydrostatic force, moment and trim drag come from the hydrostatics module ;
Fz and My below are ADDED to them in the equilibrium (IPOPT).

Open-loop evaluation : nothing is solved here. The attitude (trim, sinkage) and the wetted
geometry at that attitude (Lwl = wetted keel length, Bwl = beam at the chines) are inputs.

Conventions : SI units, speeds in knots, ALL ANGLES IN DEGREES (like `ie`).
Resistance components (to be summed), each from its own independent function :
    Rf  = Df / cos(tau)          frictional   (depends on the wetted geometry and Vm only)
    Rw  = L_dyn tan(tau)         pressure     (depends on the dynamic lift only)
    Rsp                          whisker spray (viscous, 2007 extension)
    Rtr = 0                      dry transom
Rf + Rw is Savitsky's  D = Delta tan(tau) + Df / cos(tau)  (the vertical friction component
Df sin(tau) is carried by the pressure force, which is why Rf has the 1/cos(tau)).
Hydrodynamic wrench   : Fz, My       (NOT resistances, never to be summed with them).

Range of validity of the original method (total lift) :
0.6 <= Cv <= 13, 2 deg <= trim <= 15 deg, lambda <= 4, deadrise <= 30 deg.
The dynamic-only lift is the Cv -> infinity limit ; at low Cv it under-predicts the lift
unless the hydrostatics module supplies the missing buoyancy.
"""

#%% DEPENDENCIES

import archibald.numpy as np
import archibald.toolbox.units as u

from archibald.dynamics.hydro.common import Cf_hull, transom_resistance

#%% CONSTANTS

_DEG = np.pi / 180.   # [rad/deg]
TAU_MIN = 0.3         # Smooth lower bound of the trim angle [deg]
LAMBDA_MIN = 0.05     # Smooth lower bound of the wetted length ratio [-]
TAU_SPRAY_MIN = 2.0   # Lower bound of the trim angle for the whisker spray model [deg]

#%% INDIVIDUAL COEFFICIENTS

def tau(
        trim,
        **kwargs,
    ):
    r"""Effective trim angle, smoothly bounded below to avoid the singularity at zero trim.

    .. math::
        \tau = \tau_{min} + \text{softplus}\left(\text{trim} - \tau_{min}\right)

    Parameters
    ----------
    trim : float, Trim angle, bow up positive [deg].

    Returns
    -------
    float, Effective trim angle tau [deg].
    """
    return TAU_MIN + np.softplus(trim - TAU_MIN, beta=5.)


def lambda_(
        trim,
        Lwl,
        Bwl,
        deadrise,
        rise_factor=1.,
        lam=None,
        **kwargs,
    ):
    r"""Mean wetted length to chine beam ratio, deduced from the wetted keel length.

    Purely geometric : Lwl is the wetted keel length supplied by the hydrostatics module at the
    current attitude (no hydrostatic force is involved). According to
    :cite:`savitsky1964hydrodynamic`, :math:`L_K = \lambda b + \frac{b\tan\beta}{2\pi\tan\tau}`,
    hence:

    .. math::
        \lambda = \frac{f_{rise}\,L_{WL}}{b} - \frac{\tan\beta}{2\pi\tan\tau}

    Parameters
    ----------
    trim : float, Trim angle, bow up positive [deg].
    Lwl : float, Wetted keel length at the current attitude [m].
    Bwl : float, Beam at the chines [m].
    deadrise : float, Deadrise angle [deg].
    rise_factor : float, Dynamic water rise-up factor applied to Lwl [-] (calibration).
    lam : float, Optional imposed value of lambda (external variable) [-].

    Returns
    -------
    float, Mean wetted length ratio lambda [-].
    """
    if lam is not None:
        return lam

    _tau = tau(trim)
    lam_raw = (rise_factor * Lwl) / Bwl \
        - np.tan(deadrise * _DEG) / (2. * np.pi * np.tan(_tau * _DEG))

    return LAMBDA_MIN + np.softplus(lam_raw - LAMBDA_MIN, beta=20.)


def C_L0(
        trim,
        lam,
        **kwargs,
    ):
    r"""Dynamic lift coefficient of the flat plate (zero deadrise).

    Savitsky's :math:`C_{L_0} = \tau^{1.1}(0.012\lambda^{0.5} + 0.0055\lambda^{2.5}/C_V^2)`
    without its second (gravity dependent, hydrostatic) term:

    .. math::
        C_{L_0} = 0.012\,\tau^{1.1}\,\lambda^{0.5}

    Parameters
    ----------
    trim : float, Trim angle [deg].
    lam : float, Mean wetted length ratio [-].

    Returns
    -------
    float, Dynamic flat plate lift coefficient [-].
    """
    _tau = tau(trim)
    return 0.012 * _tau ** 1.1 * np.sqrt(lam)


def C_Lbeta(
        trim,
        lam,
        deadrise,
        **kwargs,
    ):
    r"""Dynamic lift coefficient of the deadrise surface.

    According to :cite:`savitsky1964hydrodynamic`:

    .. math::
        C_{L_\beta} = C_{L_0} - 0.0065\,\beta\,C_{L_0}^{0.6} \quad (\geq 0, \text{smooth floor})

    Parameters
    ----------
    trim : float, Trim angle [deg].
    lam : float, Mean wetted length ratio [-].
    deadrise : float, Deadrise angle beta [deg].

    Returns
    -------
    float, Dynamic deadrise lift coefficient [-].
    """
    _CL0 = C_L0(trim=trim, lam=lam)
    # smooth floor at zero : the empirical formula turns negative outside its validity range
    return np.softplus(_CL0 - 0.0065 * deadrise * _CL0 ** 0.6, beta=2000.)


def lcp(
        lam,
        Bwl,
        **kwargs,
    ):
    r"""Centre of pressure of the dynamic lift, measured from the transom along the keel line.

    Savitsky's :math:`l_{cp} = \lambda b\,(0.75 - 1/(5.21 C_V^2/\lambda^2 + 2.39))` moves aft at
    low speed because of the hydrostatic lift ; the purely dynamic limit is kept:

    .. math::
        l_{cp} = 0.75\,\lambda\,b

    Parameters
    ----------
    lam : float, Mean wetted length ratio [-].
    Bwl : float, Beam at the chines [m].

    Returns
    -------
    float, Centre of pressure abscissa from the transom [m].
    """
    return 0.75 * lam * Bwl


def V_m_ratio(
        trim,
        lam,
        deadrise,
        **kwargs,
    ):
    r"""Mean bottom velocity to speed ratio.

    According to :cite:`savitsky1964hydrodynamic`:

    .. math::
        \frac{V_m}{V} = \sqrt{1 - \frac{C_{L_\beta}}{\lambda\cos\tau}}

    Parameters
    ----------
    trim : float, Trim angle [deg].
    lam : float, Mean wetted length ratio [-].
    deadrise : float, Deadrise angle beta [deg].

    Returns
    -------
    float, Ratio Vm / V [-].
    """
    _tau = tau(trim)
    _CLb = C_Lbeta(trim=trim, lam=lam, deadrise=deadrise)
    return np.sqrt(np.fmax(0.05, 1. - _CLb / (lam * np.cos(_tau * _DEG))))


def L_K(
        trim,
        lam,
        Bwl,
        deadrise,
        **kwargs,
    ):
    r"""Wetted keel length.

    .. math::
        L_K = \lambda\,b + \frac{b\tan\beta}{2\pi\tan\tau}

    Returns
    -------
    float, Wetted keel length [m].
    """
    _tau = tau(trim)
    return lam * Bwl + Bwl * np.tan(deadrise * _DEG) / (2. * np.pi * np.tan(_tau * _DEG))


def L_C(
        trim,
        lam,
        Bwl,
        deadrise,
        **kwargs,
    ):
    r"""Wetted chine length.

    .. math::
        L_C = \lambda\,b - \frac{b\tan\beta}{2\pi\tan\tau}

    Returns
    -------
    float, Wetted chine length [m].
    """
    _tau = tau(trim)
    return lam * Bwl - Bwl * np.tan(deadrise * _DEG) / (2. * np.pi * np.tan(_tau * _DEG))


def alpha_spray(
        trim,
        deadrise,
        **kwargs,
    ):
    r"""Angle of the stagnation / spray root line relative to the keel.

    According to Wagner (1932) wave-rise factor of pi/2 (:cite:`savitsky2007whisker`):

    .. math::
        \alpha = \arctan\left(\frac{\pi}{2}\frac{\tan\tau}{\tan\beta}\right)

    Smoothly bounded to 45 deg (the empirical spray model is meant for deep-V hulls at small
    trim) and evaluated with a trim of at least 2 deg (validity range of the model).

    Parameters
    ----------
    trim : float, Trim angle [deg].
    deadrise : float, Deadrise angle [deg].

    Returns
    -------
    float, Spray root line angle alpha [deg].
    """
    # The spray model is only valid for trim >= 2 deg ; below it A_as ~ 1 / sin(2 alpha) diverges.
    _tau = TAU_SPRAY_MIN + np.softplus(tau(trim) - TAU_SPRAY_MIN, beta=5.)
    a = np.arctan(np.pi / 2. * np.tan(_tau * _DEG) / np.tan(deadrise * _DEG)) / _DEG
    return 45. - np.softplus(45. - a, beta=0.5)


def A_spray(
        trim,
        Bwl,
        deadrise,
        **kwargs,
    ):
    r"""Whisker spray wetted area (forward of the stagnation line).

    .. math::
        A_{as} = \frac{b^2}{4\,\sin(2\alpha)\cos\beta}

    !! Trigonometric form read from secondary sources (see compute_Rsp_savitsky).

    Returns
    -------
    float, Whisker spray area [m²].
    """
    _alpha = alpha_spray(trim=trim, deadrise=deadrise)
    return Bwl ** 2 / (4. * np.sin(2. * _alpha * _DEG) * np.cos(deadrise * _DEG))


def phi_spray(
        trim,
        deadrise,
        **kwargs,
    ):
    r"""Angle between the spray viscous force and the keel line.

    .. math::
        \varphi = \arctan\left(\frac{\tan\theta}{\cos\beta}\right), \qquad \theta = 2\alpha

    (the whisker spray leaves the stagnation line at about twice its angle).

    Returns
    -------
    float, Spray force angle phi [deg].
    """
    _alpha = alpha_spray(trim=trim, deadrise=deadrise)
    return np.arctan(np.tan(2. * _alpha * _DEG) / np.cos(deadrise * _DEG)) / _DEG


def blount_fox_M_prime(
        bf_M,
        bf_K=0.5,
        **kwargs,
    ):
    r"""Blount & Fox semi-planing correction factor.

    .. math::
        M' = K\,(M - 1) + 1

    Parameters
    ----------
    bf_M : float, Blount & Fox factor M [-] (user supplied).
    bf_K : float, Correlation factor [-] (0 = Savitsky, 0.5 = realistic hump, 1 = Blount & Fox).

    Returns
    -------
    float, Corrected factor M' [-].
    """
    return bf_K * (bf_M - 1.) + 1.


def L_dynamic(
        stw,
        rho,
        trim,
        Lwl,
        Bwl,
        deadrise,
        rise_factor=1.,
        lam=None,
        **kwargs,
    ):
    r"""Dynamic vertical lift (no hydrostatic part).

    .. math::
        L = C_{L_\beta}\,\tfrac{1}{2}\rho V^2 b^2

    Parameters
    ----------
    stw : float, Speed through water [kt].
    rho : float, Water density [kg/m³].
    trim : float, Trim angle [deg].
    Lwl : float, Wetted keel length at the current attitude [m].
    Bwl : float, Beam at the chines [m].
    deadrise : float, Deadrise angle [deg].

    Returns
    -------
    float, Dynamic vertical lift [N].
    """
    Vms = np.softplus(stw * u.kt, beta=1e3) + 1e-6
    _lam = lambda_(trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, rise_factor=rise_factor, lam=lam)
    _CLb = C_Lbeta(trim=trim, lam=_lam, deadrise=deadrise)

    return _CLb * 0.5 * rho * Vms ** 2 * Bwl ** 2


def D_f(
        stw,
        rho,
        nu,
        trim,
        Lwl,
        Bwl,
        deadrise,
        rise_factor=1.,
        lam=None,
        dCf=0.,
        Aws=None,
        use_Aws=False,
        **kwargs,
    ):
    r"""Frictional force along the keel line (ITTC '57 on the mean wetted length).

    .. math::
        D_f = \tfrac{1}{2}\rho\,C_f(Re_m)\,V_m^2\,\frac{\lambda b^2}{\cos\beta}

    Parameters
    ----------
    nu : float, Kinematic viscosity [m²/s].
    dCf : float, Roughness allowance [-].
    Aws : float, Hydrostatic wetted surface [m²] (used if use_Aws is True).
    use_Aws : bool, Use the hydrostatic wetted surface instead of lambda b² / cos(beta).

    Returns
    -------
    float, Frictional force along the keel [N].
    """
    Vms = np.softplus(stw * u.kt, beta=1e3) + 1e-6
    _lam = lambda_(trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, rise_factor=rise_factor, lam=lam)
    Vm = Vms * V_m_ratio(trim=trim, lam=_lam, deadrise=deadrise)

    if use_Aws and Aws is not None:
        S, Lw = Aws, Lwl
    else:
        S, Lw = _lam * Bwl ** 2 / np.cos(deadrise * _DEG), _lam * Bwl

    Re = Vm * Lw / nu
    return 0.5 * rho * (Cf_hull(Re + 10.) + dCf) * Vm ** 2 * S


def N_normal(
        stw,
        rho,
        nu,
        trim,
        Lwl,
        Bwl,
        deadrise,
        **kwargs,
    ):
    r"""Dynamic force normal to the keel line.

    .. math::
        N = \frac{L + D_f \sin\tau}{\cos\tau}

    Returns
    -------
    float, Normal force [N].
    """
    _tau = tau(trim)
    _L = L_dynamic(stw=stw, rho=rho, trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, **kwargs)
    _Df = D_f(stw=stw, rho=rho, nu=nu, trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, **kwargs)
    return (_L + _Df * np.sin(_tau * _DEG)) / np.cos(_tau * _DEG)

#%% RESISTANCE COMPONENTS

def compute_Rf_savitsky(
        stw,
        rho,
        nu,
        trim,
        Lwl,
        Bwl,
        deadrise,
        **kwargs,
    ):
    r"""
    Calculate the frictional resistance using the Savitsky method.

    Independent of the lift : it only depends on the wetted geometry and on the mean
    bottom velocity.

    .. math::
        R_f = \frac{D_f}{\cos\tau}

    Parameters
    ----------
    stw : float, Speed through water [kt].
    rho : float, Water density [kg/m³].
    nu : float, Kinematic viscosity of water [m²/s].
    trim : float, Trim angle, bow up positive [deg].
    Lwl : float, Wetted keel length at the current attitude [m].
    Bwl : float, Beam at the chines [m].
    deadrise : float, Deadrise angle [deg].
    Optional (kwargs) : rise_factor, lam, dCf, Aws, use_Aws.

    Returns
    -------
    float, Frictional resistance [N].
    """
    _tau = tau(trim)
    _Df = D_f(stw=stw, rho=rho, nu=nu, trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, **kwargs)
    Rf = _Df / np.cos(_tau * _DEG)

    if not np.is_casadi_type(Rf):
        Rf = np.squeeze(Rf)

    return Rf


def compute_Rp_savitsky(
        stw,
        rho,
        trim,
        Lwl,
        Bwl,
        deadrise,
        bf_M=None,
        bf_K=0.5,
        **kwargs,
    ):
    r"""
    Calculate the pressure (induced) resistance of the DYNAMIC lift using the Savitsky method.

    Independent of friction : it only depends on the dynamic lift and on the trim.

    .. math::
        R_p = L_{dyn} \tan\tau

    The trim drag of the hydrostatic (buoyant) part is NOT included : it belongs to the
    hydrostatics module.

    Parameters
    ----------
    stw : float, Speed through water [kt].
    rho : float, Water density [kg/m³].
    trim : float, Trim angle, bow up positive [deg].
    Lwl : float, Wetted keel length at the current attitude [m].
    Bwl : float, Beam at the chines [m].
    deadrise : float, Deadrise angle [deg].
    bf_M : float, Optional Blount & Fox factor M (user supplied) [-].
    bf_K : float, Blount & Fox correlation factor K [-].

    Returns
    -------
    float, Pressure resistance [N].
    """
    _tau = tau(trim)
    _L = L_dynamic(stw=stw, rho=rho, trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, **kwargs)
    Rp = _L * np.tan(_tau * _DEG)

    if bf_M is not None:
        Rp = Rp * blount_fox_M_prime(bf_M, bf_K)

    if not np.is_casadi_type(Rp):
        Rp = np.squeeze(Rp)

    return Rp


# Registry name (same slot as the residuary / wave component of the other methods)
compute_Rw_savitsky = compute_Rp_savitsky


def compute_Rsp_savitsky(
        stw,
        rho,
        nu,
        trim,
        Lwl,
        Bwl,
        deadrise,
        rise_factor=1.,
        lam=None,
        dCf=0.,
        **kwargs,
    ):
    r"""
    Calculate the whisker spray (viscous) resistance, Savitsky, DeLorme & Datla (2007).

    The 1964 method only contains the friction and pressure drag aft of the stagnation
    line ; the thin spray sheet forward of it adds viscous drag (10-15 % of the total at
    planing speeds):

    .. math::
        F_s = \tfrac{1}{2}\rho V^2 C_f A_{as}, \qquad R_{sp} = F_s \cos\varphi \cos\tau

    Parameters
    ----------
    (same as compute_Rf_savitsky)

    Returns
    -------
    float, Whisker spray resistance [N].

    Notes
    -----
    !! The original paper could not be consulted. The root-line angle alpha (Wagner pi/2
       factor), F_s, phi and R_sp = F_s cos(phi) cos(tau) come from secondary sources ;
       the area A_as = b^2 / (4 sin(2 alpha) cos(beta)) and theta = 2 alpha are the readings
       consistent with the published magnitude (spray = 10-15 % of the total). Check them
       against the original (Marine Technology 44(1), 2007) before normative use.
       Cf is evaluated at the free stream speed on the mean wetted length (+ dCf).
       The spray force moment / lift are neglected.
    """
    Vms = np.softplus(stw * u.kt, beta=1e3) + 1e-6
    _lam = lambda_(trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, rise_factor=rise_factor, lam=lam)
    _tau = tau(trim)
    Cf = Cf_hull(Vms * _lam * Bwl / nu + 10.) + dCf

    Fs = 0.5 * rho * Vms ** 2 * Cf * A_spray(trim=trim, Bwl=Bwl, deadrise=deadrise)
    Rsp = Fs * np.cos(phi_spray(trim=trim, deadrise=deadrise) * _DEG) * np.cos(_tau * _DEG)

    if not np.is_casadi_type(Rsp):
        Rsp = np.squeeze(Rsp)

    return Rsp


def compute_Rtr_savitsky(
        stw,
        Ttr,
        Atr,
        rho,
        g,
        **kwargs,
    ):
    """
    Calculate transom resistance using the Holtrop-Mennen method.
    """

    Vms = stw * u.kt

    Fr_T = Vms / (np.sqrt(g * Ttr + 1e-8) + 1e-8)

    return transom_resistance(
            Vms,
            Fr_T,
            Atr,
            rho,
        )

#%% HYDRODYNAMIC WRENCH (not resistances)

def compute_Fz_savitsky(
        stw,
        rho,
        trim,
        Lwl,
        Bwl,
        deadrise,
        **kwargs,
    ):
    """
    Dynamic vertical force, upward positive [N] (no hydrostatic part : add it to the
    buoyancy supplied by the hydrostatics module).
    """
    Fz = L_dynamic(stw=stw, rho=rho, trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, **kwargs)

    if not np.is_casadi_type(Fz):
        Fz = np.squeeze(Fz)

    return Fz


def compute_My_savitsky(
        stw,
        rho,
        nu,
        trim,
        Lwl,
        Bwl,
        deadrise,
        lcg_tr,
        vcg_keel=None,
        **kwargs,
    ):
    r"""
    Dynamic pitching moment about the reduction point, bow up positive [N.m].

    .. math::
        M_y = N\,(l_{cp} - l_{cg}) - D_f\left(v_{cg} - \frac{b}{4}\tan\beta\right)

    Parameters
    ----------
    lcg_tr : float, Reduction point abscissa from the transom, along the keel [m].
    vcg_keel : float, Reduction point height above the keel, normal to it [m]
        (default : friction lever arm set to zero).
    """
    _lam = lambda_(trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, **kwargs)
    _N = N_normal(stw=stw, rho=rho, nu=nu, trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, **kwargs)
    _Df = D_f(stw=stw, rho=rho, nu=nu, trim=trim, Lwl=Lwl, Bwl=Bwl, deadrise=deadrise, **kwargs)

    f_arm = 0. if vcg_keel is None else vcg_keel - Bwl / 4. * np.tan(deadrise * _DEG)
    My = _N * (lcp(lam=_lam, Bwl=Bwl) - lcg_tr) - _Df * f_arm

    if not np.is_casadi_type(My):
        My = np.squeeze(My)

    return My