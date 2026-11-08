# -*- coding: utf-8 -*-
r"""
ADDED RESISTANCE IN WAVES OF ARBITRARY HEADING (MITTENDORF ET AL.).

Semi-empirical transfer function of the added resistance of a ship in regular
waves, with its 90 % prediction interval, from:

    Mittendorf, M.; Nielsen, U.D.; Bingham, H.B.; Liu, S. (2022) "Towards the
    Uncertainty Quantification of Semi-Empirical Formulas Applied to the Added
    Resistance of Ships in Waves of Arbitrary Heading", Ocean Engineering 251, 111040.

It extends the formula of Liu & Papanikolaou (2020) and returns the coefficient

.. math::
    C_{AW} = \frac{R_{AW}}{\rho\,g\,\zeta_a^2\,B^2 / L_{pp}}

``caw_mittendorf`` is a differentiable version of the published code (``mm_raw``
keeps its interface). The branches of the published code are kept as they are,
with ``np.where``, so the result is the same, including where the formula is
not smooth (for example at the switch of the wave frequency term at omega = 1).

``compute_Raw_mittendorf`` gives the mean added resistance in an irregular sea,
by the usual linear superposition of the regular wave response on a wave
spectrum (e.g. ITTC Recommended Procedure 7.5-02-07-02.2):

.. math::
    \bar R_{AW} = 2 \int_0^\infty \frac{R_{AW}(\omega)}{\zeta_a^2}\,S(\omega)\,d\omega
"""

import warnings

import numpy as onp

import archibald.numpy as np
import archibald.toolbox.units as u

#%% COEFFICIENTS OF THE PUBLISHED CODE

# Mean, upper and lower estimates, for Cb < 0.7 (ps) and Cb >= 0.7 (pb)
_PS = (
    [4.05946942e+00, 7.04999991e-01, 5.62274820e+01, 1.18458738e+00,
     4.05706220e+00, 6.50000001e-01, 2.25000391e+00, 1.33566557e-01,
     1.62572344e+00, 3.58496857e+00, 2.91066129e+01, 1.08675480e+01,
     9.87827505e+00, 4.80961585e+02, 2.62009924e+00, 1.23162591e+02,
     1.74527790e+00, 1.06966785e-01, 3.50000000e+00, 8.24976128e-01,
     1.25000000e-01, 1.03000141e+00, 1.20135494e+00, 7.22951370e-01,
     1.36307675e+01, 5.00000000e+00, 3.14999975e+00, 2.89714558e+00,
     5.55186351e+00],
    [4.27240317e+00, 7.05000000e-01, 5.58810360e+01, 1.05849695e+00,
     5.00000000e+00, 6.50000000e-01, 2.53759425e+00, 2.10000000e-01,
     1.50221848e+00, 3.58270417e+00, 2.70758743e+01, 8.20000054e+00,
     6.37000001e+00, 4.25000004e+02, 3.50000000e+00, 1.21801494e+02,
     2.04089706e+00, 8.35524201e-02, 3.50000000e+00, 6.39449863e-01,
     1.25000223e-01, 1.03000050e+00, 1.13419070e+00, 6.03042936e-01,
     1.23638297e+01, 5.00000000e+00, 3.14999947e+00, 1.00000004e+00,
     7.49999988e+00],
    [3.03271576e+00, 4.20000015e-01, 4.99673118e+01, 1.65946595e+00,
     3.46566078e+00, 1.08999872e+00, 3.53017296e+00, 1.20000000e-01,
     1.48347334e+00, 2.62015273e+00, 3.09243793e+01, 1.29737533e+01,
     1.07000000e+01, 7.07939677e+02, 1.95001682e+00, 9.35351064e+01,
     2.16620326e+00, 1.39999997e-01, 2.89553627e+00, 8.24992364e-01,
     1.30763658e-01, 1.03000000e+00, 1.42153913e+00, 6.31753054e-01,
     1.72855270e+01, 3.00000001e+00, 3.11414443e+00, 6.00000000e+00,
     2.00000001e+00],
)
_PB = (
    [6.98574593e+00, 4.25032242e-01, 7.09177633e+01, 1.99960254e+00,
     3.01301221e+00, 8.99994676e-01, 3.72568388e+00, 2.81612167e-01,
     1.38750811e+00, 3.86896582e+00, 1.79628922e+01, 1.14985385e+01,
     5.58870161e+00, 4.73743004e+02, 2.21408408e+00, 1.53646073e+02,
     2.43280559e+00, 5.56757940e-02, 2.41039849e+00, 8.13139297e-01,
     2.99999998e-01, 1.00776665e+00, 1.07619998e+00, 7.78406778e-01,
     1.92414808e+01, 3.02938365e+00, 3.84472339e+00, 3.12300755e+00,
     4.52696031e+00],
    [4.58710582e+00, 5.65022282e-01, 5.42786254e+01, 1.23604172e+00,
     4.99998263e+00, 6.50259977e-01, 2.25000468e+00, 2.10000000e-01,
     1.54949896e+00, 3.95366005e+00, 3.37772648e+01, 1.39999578e+01,
     6.37000001e+00, 4.25000564e+02, 3.50000000e+00, 1.18022195e+02,
     2.04252813e+00, 1.04536091e-01, 2.26795773e+00, 7.01539307e-01,
     1.25034505e-01, 1.03001307e+00, 1.44950225e+00, 6.47384026e-01,
     1.44229162e+01, 3.00000134e+00, 3.14999949e+00, 1.00000000e+00,
     7.49999997e+00],
    [3.00355039e+00, 4.20001279e-01, 5.58414825e+01, 1.32934752e+00,
     4.31245215e+00, 1.08998811e+00, 3.74999578e+00, 1.46428490e-01,
     1.79301192e+00, 3.56562762e+00, 2.76275045e+01, 1.22975415e+01,
     1.06999891e+01, 6.79271138e+02, 2.30875016e+00, 1.30001874e+02,
     2.23066085e+00, 8.32000024e-02, 2.63843128e+00, 7.83843682e-01,
     2.20011546e-01, 1.35089575e+00, 1.31843660e+00, 6.26778792e-01,
     1.52590637e+01, 3.00000000e+00, 2.40826045e+00, 5.99999925e+00,
     2.00000000e+00],
)
# Constant shift of each estimate: the upper bound is the second set, the lower bound the third
_OFFSETS = (0., 0.19967, -0.19967)
_ESTIMATES = {"mean": 0, "upper": 1, "lower": 2}

CB_SWITCH = 0.7     # block coefficient at which the coefficient sets change
G_PUBLISHED = 9.807 # gravity of the published code [m/s²]

#%% REGULAR WAVES

def _caw_set(
        p,
        add,
        wavelength,
        wave_heading,
        Lpp,
        Bwl,
        Tf,
        Ta,
        Cb,
        Le,
        Lr,
        kyy,
        Fr,
        g,
    ):
    """Added resistance coefficient for one set of coefficients ``p``, see ``caw_mittendorf``."""
    dr = wave_heading
    lambd = wavelength
    alp = np.radians(dr)
    tmax = np.maximum(Tf, Ta)
    delta = np.abs(Ta - Tf)
    U = Fr * np.sqrt(g * Lpp)
    log_BT = np.log(Bwl / tmax)

    # angles of entrance and run
    E1 = np.arctan(0.99 * Bwl * 0.5 / Le)
    E2 = np.arctan(0.99 * Bwl * 0.5 / Lr)

    a2 = np.where(
        Fr < 0.12,
        0.0072 + p[7] * Fr,
        np.maximum(Fr, 1e-12) ** p[8] * np.exp(-p[9] * Fr),
    )
    a3 = 1. + p[10] * np.arctan(delta / Lpp)

    w0 = np.sqrt(2. * np.pi * g / lambd)
    Vc = np.sqrt(g * lambd / 2. / np.pi)

    omega = (
        p[16] * kyy ** (1. / 3.) * np.sqrt(Lpp / lambd)
        * (1. - (p[17] / Cb) * (log_BT - np.log(p[18])))
        * (Cb / p[19]) ** p[20]
        * ((-p[21] * Fr ** 2 + p[22] * Fr) * np.abs(np.cos(alp)) + (p[23] * (13. + np.cos(2. * alp))) / p[24])
    )
    below = omega < 1.
    b1 = np.where(below, p[11], -p[12])
    d1 = np.where(
        below,
        p[13] * (Lpp * Cb / Bwl) ** (-p[14]),
        -p[13] * (Lpp / Bwl) ** (-p[14]) * (4. - p[15] * np.arctan2(delta, Lpp)),
    )
    frequency_term = omega ** b1 * np.exp((b1 / d1) * (1. - omega ** d1))

    # f(alpha)
    fofa = np.where(dr >= np.degrees(np.pi - E1), -np.cos(alp), 0.)

    a1 = p[2] * Cb ** p[3] * (p[4] * kyy) ** 2 * (p[5] / Cb) ** (-(1. + Fr) * np.cos(alp)) / log_BT * (1. - 2. * np.cos(alp)) / p[6]
    a90 = p[2] * Cb ** p[3] * (p[4] * kyy) ** 2 * (p[5] / Cb) ** (-(1. + Fr) * np.cos(0.5 * np.pi)) / log_BT * (1. - 2. * np.cos(0.5 * np.pi)) / p[6]

    # following waves
    fnr = (U - Vc / 2.) / np.sqrt(g * Lpp)
    a20 = np.where(
        U < Vc / 2.,
        (-0.0072 + 0.0072 * 4. / Vc * U) * (0.87 / Cb),
        np.where(
            fnr <= 0.12,
            (0.0072 + p[7] * fnr) * (0.87 / Cb),
            np.maximum(fnr, 1e-12) ** 1.5 * np.exp(-3.5 * fnr) * (0.87 / Cb) ** (1. + fnr),
        ),
    )
    a10 = p[2] * Cb ** 1.34 * (4. * kyy) ** 2 / log_BT

    # motion part: head to beam seas, and an interpolation between beam and following seas
    rawm_beam = p[0] * a90 * a2 * a3 * frequency_term
    rawm_following = p[0] * a10 * a20 * a3 * frequency_term
    rawm = np.where(
        dr < 90.,
        rawm_beam + (rawm_following - rawm_beam) * (1. - dr / 90.),
        p[0] * a1 * a2 * a3 * frequency_term,
    )

    # reflection part, four segments of the waterline
    def alpha_t(tstar):
        return 1. - np.exp(-p[25] * np.pi * (tstar / lambd - tstar / (p[26] * Lpp)))

    alpha12 = alpha_t(tmax)
    alpha34 = alpha_t(tmax * (p[27] + np.sqrt(np.abs(np.cos(alp)))) / p[28])

    def segment(E, sign):
        return (
            np.sin(E + sign * alp) ** 2
            + 2. * w0 * U / g * (np.cos(E) * np.cos(E + sign * alp) - np.cos(alp))
        )

    bow = p[1] * (Lpp / Bwl) * alpha12 * (0.87 / Cb) ** ((1. + 4. * np.sqrt(Fr)) * fofa)
    stern = -p[1] * (Lpp / Bwl) * alpha34
    rawr1 = np.where(dr >= np.degrees(E1), bow * segment(E1, -1.), 0.)
    rawr2 = np.where(dr >= np.degrees(np.pi - E1), bow * segment(E1, 1.), 0.)
    rawr3 = np.where(dr <= np.degrees(np.pi - E2), stern * segment(E2, 1.), 0.)
    rawr4 = np.where(dr <= np.degrees(E2), stern * segment(E2, -1.), 0.)

    return rawr1 + rawr2 + rawr3 + rawr4 + rawm + add


def caw_mittendorf(
        wavelength,
        wave_heading,
        Lpp,
        Bwl,
        Tf,
        Ta,
        Cb,
        Le,
        Lr,
        kyy,
        Fr,
        g=G_PUBLISHED,
        estimate="mean",
    ):
    r"""
    Added resistance coefficient in regular waves, differentiable.

    .. math::
        C_{AW} = \frac{R_{AW}}{\rho\,g\,\zeta_a^2\,B^2 / L_{pp}}

    Same formula and coefficients as the code published with the paper, for one
    wavelength. Valid for wave headings from 0 (following) to 180 deg (head seas).

    Args:
        wavelength: float, Wavelength [m].
        wave_heading: float, Wave heading, 180 for head seas, 0 for following seas [deg].
        Lpp: float, Length between perpendiculars [m].
        Bwl: float, Beam [m].
        Tf, Ta: float, Draughts at the forward and aft perpendiculars [m].
        Cb: float, Block coefficient [-].
        Le, Lr: float, Lengths of entrance (bow) and run (stern) [m].
        kyy: float, Radius of gyration in pitch divided by Lpp, in general 0.25 [-].
        Fr: float, Froude number on Lpp [-].
        g: float, Gravitational acceleration [m/s²].
        estimate: str, "mean", or a bound of the 90 % prediction interval, "upper" or "lower".

    Returns:
        float, Added resistance coefficient C_AW [-].
    """
    k = _ESTIMATES[estimate]
    args = dict(
        wavelength=wavelength, wave_heading=wave_heading, Lpp=Lpp, Bwl=Bwl, Tf=Tf, Ta=Ta,
        Cb=Cb, Le=Le, Lr=Lr, kyy=kyy, Fr=Fr, g=g,
    )
    if not np.is_casadi_type(Cb, recursive=False):
        return _caw_set(_PS[k] if Cb < CB_SWITCH else _PB[k], _OFFSETS[k], **args)
    return np.where(
        Cb < CB_SWITCH,
        _caw_set(_PS[k], _OFFSETS[k], **args),
        _caw_set(_PB[k], _OFFSETS[k], **args),
    )


def mm_raw(
        wl,
        dr,
        Lpp,
        Bwl,
        Tf,
        Ta,
        Cb,
        Le,
        Lr,
        kyy,
        Fr,
    ):
    """
    Function for computing the R_AW transfer function and its 90% prediction interval for arbitrary heading angle.

    Follows the proposed method by Mittendorf, M.; Nielsen, U.D.; Bingham, H.B.; Liu, S. (2021) "Towards the
    Uncertainty Quantification of Semi-Empirical Formulas Applied to the Added Resistance of Ships in Waves of
    Arbitrary Heading" Ocean Engineering Vol. 251 No. 111040.

    Interface of the published code, numeric only. See ``caw_mittendorf`` for a
    differentiable version.

    >INPUT
    wl - wavelength [m]
    dr - wave angle [deg]
    Lpp - length per perpendicular [m]
    Bwl - beam [m]
    Tf - draught at fp [m]
    Ta - draught at ap [m]
    Cb - block coefficient [-]
    Le - length of run (bow) [m]
    Lr - length of run (stern) [m]
    kyy - norm. radius of gyration (kyy/Lpp), in general 0.25 [-]
    Fr - froude number [-]

    >OUTPUT
    array[0] - added resistance coefficient (mean)
    array[1] - added resistance coefficient (upper bound of the 90 % interval)
    array[2] - added resistance coefficient (lower bound of the 90 % interval)
    array[3] - intrinsic wave frequency
    array[4] - normalized wave frequency(w/sqrt(g/Lpp))
    array[5] - lambda/Lpp
    """
    if dr > 180 or dr < 0:
        warnings.warn('Method is only valid in range [0,180]deg.')
        dr = onp.nan
    g = G_PUBLISHED

    raw_arr = [
        [
            float(caw_mittendorf(lambd, dr, Lpp, Bwl, Tf, Ta, Cb, Le, Lr, kyy, Fr, g=g, estimate=estimate))
            for lambd in wl
        ]
        for estimate in ("mean", "upper", "lower")
    ]

    w = onp.sqrt(onp.divide(2 * onp.pi * g, wl))
    wbar = onp.divide(w, onp.sqrt(g / Lpp))
    ll = onp.divide(wl, Lpp)

    # exception for intersecting bounds in following seas
    if onp.any(onp.greater(raw_arr[2], raw_arr[1])):
        warnings.warn('Uncertainty bounds did intersect, fallback to simplified version')
        spread = max(raw_arr[1]) - min(raw_arr[2])
        raw_arr1 = onp.add(raw_arr[0], spread * 0.5)
        raw_arr2 = onp.add(raw_arr[0], -spread * 0.5)
        return [raw_arr[0], raw_arr1, raw_arr2, w, wbar, ll]
    return [raw_arr[0], raw_arr[1], raw_arr[2], w, wbar, ll]

#%% IRREGULAR WAVES

def wave_spectrum(
        omega,
        swh,
        tp,
        gamma=1.,
        **kwargs,
    ):
    r"""
    Wave spectrum of a sea state, JONSWAP or Pierson-Moskowitz.

    .. math::
        S(\omega) = A_\gamma\,\frac{5}{16}\,H_s^2\,\omega_p^4\,\omega^{-5}
            \exp\left(-\frac{5}{4}\left(\frac{\omega_p}{\omega}\right)^4\right)
            \gamma^{\exp\left(-\frac{(\omega - \omega_p)^2}{2\,\sigma^2\,\omega_p^2}\right)}

    with :math:`\omega_p = 2\pi/T_p`, :math:`A_\gamma = 1 - 0.287\ln\gamma`, and
    :math:`\sigma = 0.07` below the peak, 0.09 above (DNV-RP-C205, JONSWAP spectrum).
    gamma = 1 gives the Pierson-Moskowitz (two-parameter ITTC) spectrum, whose
    zeroth moment is exactly :math:`H_s^2/16`. With gamma > 1, :math:`A_\gamma`
    is an approximate normalisation.

    Args:
        omega: float, Wave frequency [rad/s].
        swh: float, Significant wave height Hs [m].
        tp: float, Peak period [s].
        gamma: float, Peak enhancement factor, 1 to 7, 3.3 for the mean JONSWAP [-].

    Returns:
        float, Spectral density [m².s/rad].
    """
    wp = 2. * np.pi / tp
    sigma = np.where(omega <= wp, 0.07, 0.09)
    peak = gamma ** np.exp(-(omega - wp) ** 2 / (2. * sigma ** 2 * wp ** 2))
    A_gamma = 1. - 0.287 * np.log(gamma)
    return A_gamma * 5. / 16. * swh ** 2 * wp ** 4 * omega ** -5 * np.exp(-1.25 * (wp / omega) ** 4) * peak


# Quadrature of the spectrum in omega / omega_p. Below 0.3 the Pierson-Moskowitz
# spectrum is below 1e-50 of its peak; above 5 lies 0.2 % of its energy.
X_MIN, X_MAX = 0.3, 5.


def compute_Raw_mittendorf(
        stw,
        rho,
        g,
        Lbp,
        Bwl,
        T,
        Cb,
        swh=None,
        tp=None,
        wave_heading=180.,
        Le=None,
        Lr=None,
        kyy=None,
        Tf=None,
        Ta=None,
        gamma=1.,
        estimate="mean",
        n_omega=241,
        **kwargs,
    ):
    r"""
    Mean added resistance in irregular waves, from Mittendorf et al. (2022).

    .. math::
        \bar R_{AW} = 2\,\frac{\rho\,g\,B^2}{L_{pp}}
            \int_0^\infty C_{AW}(\omega)\,S(\omega)\,d\omega

    with :math:`C_{AW}` from :func:`caw_mittendorf` at the deep water wavelength
    :math:`\lambda = 2\pi g/\omega^2`, and :math:`S` from :func:`wave_spectrum`.
    The waves are long crested, from a single heading. Zero in calm water
    (``swh`` None or 0).

    The method was fitted on merchant ships (Cb about 0.5 to 0.85): for a
    sailing yacht it is an extrapolation.

    Args:
        stw: float, Speed through water [kt].
        rho: float, Water density [kg/m³].
        g: float, Gravitational acceleration [m/s²].
        Lbp: float, Length between perpendiculars [m].
        Bwl: float, Waterline beam [m].
        T: float, Draught [m], used for Tf and Ta when they are not given.
        Cb: float, Block coefficient [-].
        swh: float, Significant wave height Hs [m].
        tp: float, Peak period Tp [s].
        wave_heading: float, Wave heading relative to the ship, 180 for head seas,
            0 for following seas, from 0 to 180 [deg].
        Le, Lr: float, Lengths of entrance (bow) and run (stern) [m].
        kyy: float, Radius of gyration in pitch [m]. Default 0.25 Lbp.
        Tf, Ta: float, Draughts at the forward and aft perpendiculars [m]. Default T.
        gamma: float, JONSWAP peak enhancement factor, 1 for Pierson-Moskowitz [-].
        estimate: str, "mean", "upper" or "lower" bound of the 90 % prediction interval.
        n_omega: int, Number of frequencies of the quadrature.

    Returns:
        float, Mean added resistance [N].
    """
    if swh is None or (isinstance(swh, (int, float)) and swh == 0):
        return 0. * stw
    missing = [name for name, value in (("tp", tp), ("Le", Le), ("Lr", Lr)) if value is None]
    if missing:
        raise ValueError(f"compute_Raw_mittendorf needs {missing} when a sea state (swh) is given.")
    if not np.is_casadi_type(wave_heading, recursive=False) and not 0. <= wave_heading <= 180.:
        raise ValueError(f"wave_heading should be within [0, 180] deg, not {wave_heading}.")

    Lpp = Lbp
    Tf = T if Tf is None else Tf
    Ta = T if Ta is None else Ta
    kyy_ratio = 0.25 if kyy is None else kyy / Lpp
    Fr = stw * u.kt / np.sqrt(g * Lpp)

    wp = 2. * np.pi / tp
    x = onp.linspace(X_MIN, X_MAX, n_omega)
    weights = onp.full(n_omega, x[1] - x[0])
    weights[[0, -1]] *= 0.5

    integral = 0.
    for xi, wi in zip(x, weights):
        omega = xi * wp
        caw = caw_mittendorf(
            wavelength=2. * np.pi * g / omega ** 2,
            wave_heading=wave_heading,
            Lpp=Lpp, Bwl=Bwl, Tf=Tf, Ta=Ta, Cb=Cb, Le=Le, Lr=Lr,
            kyy=kyy_ratio, Fr=Fr, g=g, estimate=estimate,
        )
        integral = integral + wi * caw * wave_spectrum(omega, swh=swh, tp=tp, gamma=gamma)

    return 2. * rho * g * Bwl ** 2 / Lpp * wp * integral

if __name__=="__main__":
    # from raw_formula import mm_raw
    from matplotlib import pyplot as plt
    # import numpy as np 
    
    # #kvlcc2
    # Lpp = 320
    # Bwl = 58
    # t = 20.80
    # Cb = 0.808
    # Le = 60
    # Lr = 60
    # kyy = 0.25
    # Fr = 0.142
    # lambd = np.linspace(0.19*Lpp, 2.6*Lpp, 100)
    
    # for angle in [180,120,90,30,0]:
    #     arr = mm_raw(lambd, angle, Lpp, Bwl, t, t, Cb, Le, Lr, kyy, Fr)
    #     plt.plot(arr[5],arr[0], label= str(angle)+' deg.')
    #     plt.fill_between(arr[5], arr[1],arr[2], alpha=0.1)
    # plt.legend()
    # plt.grid()
    # plt.xlim((0,2.7))
    # plt.title('KVLCC-2; Fn=0.142')
    # plt.ylabel(r'$C_{AW}$ [-]')
    # plt.xlabel(r'$\lambda/Lpp$ [-]')
    # plt.show()
    
    #Scb-84
    Lpp = 178.
    t = 11.57
    Bwl= 32.26
    Cb = 0.84
    Le = 35
    Lr = 45
    kyy=0.25
    Fr = 0.099
    lambd = np.linspace(0.19*Lpp, 2.6*Lpp, 100)
    
    for angle in [180,120,90,30,0]:
        arr = mm_raw(lambd, angle, Lpp, Bwl, t,t,Cb, Le,Lr, kyy,Fr)
        plt.plot(arr[4],arr[0], label= str(angle)+' deg.')
        plt.fill_between(arr[4], arr[1],arr[2], alpha=0.1)
    plt.legend()
    plt.grid()
    plt.xlim((1.5,4.5))
    plt.title('SCb-84; Fn=0.1')
    plt.ylabel(r'$C_{AW}$ [-]')
    plt.xlabel(r'$\omega/ \sqrt{g/L_{pp}}$ [-]')
    plt.show()
     
    # #dtc
    # Lpp = 355
    # dr = 60
    # t = 14.5
    # Bwl= 51
    # Cb = 0.661
    # Le= 112
    # Lr = 112
    # Fr = 0.139
    # kyy = 0.27
    # lambd = np.linspace(0.19*Lpp, 2.6*Lpp, 100)
    
    # for angle in [180,120,90,30,0]:
    #     arr = mm_raw(lambd,angle, Lpp, Bwl, t,t,Cb, Le,Lr, kyy,Fr)
    #     plt.plot(arr[3],arr[0], label= str(angle)+' deg.')
    #     plt.fill_between(arr[3], arr[1],arr[2], alpha=0.1)
    # plt.legend()
    # plt.grid()
    # plt.xlim((0.2,1))
    # plt.title('DTC; Fn=0.139')
    # plt.ylabel(r'$C_{AW}$ [-]')
    # plt.xlabel(r'$\omega$ [-]')
    # plt.show()
    
    #%%
    
    #neoliner origin
    stw = 10.
    Lpp = 136
    t = 5.5
    Bwl= 24
    Cb = 0.613
    Le= Lpp/3.
    Lr = Lpp/3.
    Fr = stw*.5144 / np.sqrt(9.8066 * Lpp)
    kyy = 0.25
    lambd = np.linspace(0.11*Lpp, 3.*Lpp, 100)
    
    for angle in [45, 30, 15, 0]:
    # for angle in [180,120,90,30,0]:
        arr = mm_raw(lambd, angle, Lpp, Bwl, t, t, Cb, Le, Lr, kyy, Fr)
        plt.plot(arr[3], arr[0], label= str(angle)+' deg.')
        plt.fill_between(arr[3], arr[1], arr[2], alpha=0.1)
    plt.legend()
    # plt.grid()
    plt.title('Neoliner Origin; Fn=0.139')
    plt.ylabel(r'$C_{AW}$ [-]')
    plt.xlabel(r'$\omega$ [-]')
    plt.show()
