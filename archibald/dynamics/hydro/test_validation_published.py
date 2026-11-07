# -*- coding: utf-8 -*-
"""
Hydrodynamic models against numbers printed in their sources.

The plots and the full comparisons are in ``examples/03 - Validation``, with the references.
These tests only keep the numbers that must not move.
"""
import math

import numpy as onp
import pytest
from scipy.integrate import trapezoid

import archibald.toolbox.units as u
from archibald.dynamics.hydro import dsyhs, holtrop, insel_molland, jamaluddin, michell, mittendorf, miyazawa, savitsky
from archibald.geometry.propeller import BSeriesPropeller


@pytest.fixture(autouse=True)
def default_floating_point_errors():
    """``test_all_operations_run.py`` calls np.seterr(all="raise") when collected, for the whole session."""
    with onp.errstate(all="warn", under="ignore"):
        yield


#%% HOLTROP & MENNEN (1982), example of p. 170

SHIP = dict(
    Lwl=205., Lbp=205., Bwl=32., T=10., volume=37500., Aws=7381.45, Atr=16., Abt=20., hB=4., Cx=0.980,
    Cwp=0.750, Cp=0.5833, Cb=37500. / (205. * 32. * 10.), ie=12.08, lcb=-0.75,
    rho=1025., g=9.81, nu=1.1883e-6, stw=25.,
)


@pytest.mark.parametrize("function, published, tolerance", [
    (holtrop.c_7, 0.1561, 1e-3),
    (holtrop.c_1, 1.398, 1e-3),
    (holtrop.c_3, 0.02119, 1e-3),  # (0.56 Abt)^1.5 instead of 0.56 Abt^1.5 gave 0.01586
    (holtrop.c_2, 0.7595, 1e-3),
    (holtrop.c_5, 0.9592, 1e-3),
    (holtrop.lambda_, 0.6513, 1e-3),
    (holtrop.L_R, 81.385, 1e-3),
    (holtrop.C_P1, 0.5477, 1e-3),
    (holtrop.c_15, -1.69385, 1e-3),
    (holtrop.m_1, -2.1274, 2e-3),  # The code divides by T + 0.1, the paper by T
])
def test_holtrop_coefficients_of_the_example(function, published, tolerance):
    assert float(function(**SHIP)) == pytest.approx(published, rel=tolerance)


def test_holtrop_relative_rotative_efficiency_of_the_example():
    assert float(holtrop.eta_R_single(**SHIP, Ae_Ao=0.7393)) == pytest.approx(0.9931, rel=1e-3)


def test_holtrop_bulb_of_the_example():
    assert float(holtrop.compute_Rb_holtrop(**SHIP, Bulbchoice=1)) == pytest.approx(49., rel=0.02)
    assert float(holtrop.p_b(SHIP["T"], SHIP["hB"], SHIP["Abt"])) == pytest.approx(0.6261, rel=1e-3)
    assert float(holtrop.Fr_i(SHIP["stw"], SHIP["T"], SHIP["hB"], SHIP["Abt"], SHIP["g"])) == pytest.approx(1.5084, rel=1e-3)


def test_holtrop_wave_resistance_of_the_example():
    # m4 of 1984 replaces m2 of 1982, published 557.11 kN
    assert float(holtrop.compute_Rw_holtrop(**SHIP)) == pytest.approx(557.11e3, rel=0.02)


def test_holtrop_twin_screw_relative_rotative_efficiency():
    # eta_R = 0.9737 + 0.111 (Cp - 0.0225 lcb) - 0.06325 P/D, p. 169
    args = dict(Cp=0.6, lcb=0.)
    low = float(holtrop.eta_R_twin(P=0.8, D=1., **args))
    high = float(holtrop.eta_R_twin(P=1.2, D=1., **args))
    assert high - low == pytest.approx(-0.06325 * 0.4)


def test_holtrop_open_stern_wake():
    # w = 0.3 Cb + 10 CV Cb - 0.1, p. 169
    args = dict(stw=25., Lbp=205., Bwl=32., T=10., volume=37500., Cp=0.5833, Cb=0.5716, Cstern=10., lcb=-0.75, rho=1025., g=9.81)
    CV = holtrop.C_V(stw=25., Lbp=205., Bwl=32., T=10., volume=37500., Cp=0.5833, Cstern=10., lcb=-0.75, rho=1025., g=9.81)
    assert float(holtrop.w_single_open_stern(**args)) == pytest.approx(0.3 * 0.5716 + 10. * float(CV) * 0.5716 - 0.1)


def test_holtrop_twin_screw_wake_runs():
    wake = holtrop.w_twin(
        stw=25., Lbp=205., Bwl=32., T=10., volume=37500., D=6., Cp=0.5833, Cb=0.5716, Cstern=10., lcb=-0.75, rho=1025., g=9.81,
    )
    assert 0. < float(wake) < 0.5


#%% SAVITSKY: S&B (1976) example as computed by OpenPlaning (MIT)

OPEN_PLANING = dict(stw=13.07 / u.kt, b=7.315, deadrise=15., tau=2.878945, L_K=28.69256)


def input_trim(tau):
    """Inverse of the soft floor of ``savitsky.tau``."""
    return math.log(math.exp(tau) - 1.)


@pytest.mark.parametrize("name, published", [
    ("L_C", 17.67617),
    ("lambda_", 3.199428),
    ("x_s", 11.0164),
    ("A_wetted", 175.5762),
])
def test_savitsky_geometry_with_open_planing(name, published):
    value = getattr(savitsky, name)(
        trim=input_trim(OPEN_PLANING["tau"]), Lwl=OPEN_PLANING["L_K"], Bwl=OPEN_PLANING["b"], deadrise=OPEN_PLANING["deadrise"],
    )
    assert float(value) == pytest.approx(published, rel=1e-4)


def test_savitsky_mean_bottom_velocity_with_open_planing():
    trim = input_trim(OPEN_PLANING["tau"])
    lam = savitsky.lambda_(trim=trim, Lwl=OPEN_PLANING["L_K"], Bwl=OPEN_PLANING["b"], deadrise=OPEN_PLANING["deadrise"])
    ratio = savitsky.V_m_ratio(trim=trim, lam=lam, deadrise=OPEN_PLANING["deadrise"])
    assert 13.07 * float(ratio) == pytest.approx(12.96912, rel=1e-4)


def test_savitsky_added_resistance_with_open_planing():
    Raw = savitsky.compute_Raw_savitsky(
        stw=OPEN_PLANING["stw"], rho=1025.87, g=9.8066, trim=OPEN_PLANING["tau"], Bwl=OPEN_PLANING["b"], deadrise=OPEN_PLANING["deadrise"],
        swh=1.402, Loa=24.38, displacement=827400.,
    )
    assert float(Raw) == pytest.approx(29716.1, rel=1e-3)


#%% DSYHS

def test_dsyhs_interpolation_of_the_resistance_table():
    table = dsyhs._keunig_coefs
    for i in range(8):
        key = f"a{i}"
        for Fr, value in zip(table["Fr"], table[key]):
            assert float(dsyhs._keunig_interpolators[key](Fr)) == pytest.approx(float(value), rel=1e-6, abs=1e-7)


def test_dsyhs_added_resistance_at_the_nodes_of_its_table():
    rho, g, Lwl, swh, volume, kyy = 1025., 9.81, 10., 1., 3., 2.5
    term = 100. * volume ** (1. / 3.) / Lwl * kyy / Lwl
    for row in dsyhs._wave_coefs[::7]:
        for k, Fr in enumerate([0.15, 0.25, 0.35, 0.45, 0.60]):
            a, b = row[2 + 2 * k], row[3 + 2 * k]
            value = dsyhs.compute_Raw_dsyhs(
                rho=rho, g=g, Lwl=Lwl, swh=swh, volume=volume, kyy=kyy, T1=row[0] / math.sqrt(g / Lwl), mu=row[1], Fr=Fr,
            )
            assert float(value) == pytest.approx(a * term ** b * rho * g * Lwl * swh ** 2 / 100., rel=1e-5, abs=1e-5)


#%% MICHELL

WIGLEY = dict(Lwl=1., Bwl=0.1, T=0.0625, Cp=2. / 3., Cx=2. / 3., rho=1000., g=9.81)


def michell_integral(Fn):
    """The original integral of Michell (1898) for the Wigley hull, on dense grids."""
    L, B, T, rho, g = 1., 0.1, 0.0625, 1000., 9.81
    U = Fn * math.sqrt(g * L)
    k0 = g / U ** 2
    w = onp.linspace(0., math.sqrt(math.sqrt(100. / (k0 * T)) + 1.), 1501)
    l = 1. + w ** 2
    x = onp.linspace(-L / 2., L / 2., 1501)
    z = onp.linspace(-T, 0., 801)
    amplitude = [
        (
            trapezoid(-B / 2. * 8. * x / L ** 2 * onp.sin(li * k0 * x), x)
            * trapezoid((1. - (z / T) ** 2) * onp.exp(li ** 2 * k0 * z), z)
        ) ** 2
        for li in l
    ]
    return 4. * rho * g ** 2 / (math.pi * U ** 2) * trapezoid(onp.array(amplitude) * 2. * l ** 2 / onp.sqrt(l + 1.), w)


@pytest.mark.parametrize("Fn", [0.25, 0.40, 0.60])
def test_michell_against_the_integral_on_the_wigley_hull(Fn):
    Rw = michell.compute_Rw_michell(stw=Fn * math.sqrt(9.81) / u.kt, **WIGLEY)
    # The module truncates the integral earlier, which costs up to 0.5 % above Fn = 0.2
    assert float(Rw) == pytest.approx(michell_integral(Fn), rel=0.006)


def test_michell_two_coincident_hulls_have_four_times_the_resistance():
    R = michell.compute_Rw_matrix_michell(
        stw=0.4 * math.sqrt(9.81) / u.kt, Lwl=[1., 1.], Bwl=[0.1, 0.1], T=[0.0625] * 2, Cp=[2. / 3.] * 2, Cx=[2. / 3.] * 2,
        x=[0., 0.], y=[0., 0.], rho=1000., g=9.81,
    )
    assert float(sum(map(sum, R))) == pytest.approx(4. * float(R[0][0]))


#%% MULTIHULL INTERACTION: Iqbal & Trimulyono, KAPAL 11(3), 2014

@pytest.mark.parametrize("SL, published", [(0.2, 1.4047), (0.3, 1.3942), (0.4, 1.3882)])
def test_jamaluddin_form_factor_of_table_5(SL, published):
    Lwl, volume = 19.963, 19.443
    base = 3.03 * (Lwl / volume ** (1. / 3.)) ** (-0.40)
    assert base * (1. + float(jamaluddin.viscous_interference_ratio(Lwl, volume, SL))) == pytest.approx(published, abs=5e-4)


@pytest.mark.parametrize("Fn, SL, published", [
    (0.19, 0.2, 0.627), (0.19, 0.4, 0.241), (0.28, 0.2, 1.456), (0.28, 0.3, 1.023), (0.37, 0.4, 0.777),
    (0.47, 0.3, 0.935), (0.56, 0.2, 0.830), (0.65, 0.4, 0.591),
])
def test_jamaluddin_wave_interference_of_table_6(Fn, SL, published):
    assert float(jamaluddin.tau_jamaluddin(Fn, SL)) == pytest.approx(published, abs=1e-3)


def test_miyazawa_factor():
    dRf = miyazawa.compute_dRf_miyazawa(Bwl=[2., 2.], y=[-2., 2.], Rf=[100., 100.])
    assert float(dRf[0]) == pytest.approx(100. * (2. / 4.) ** 2)


def test_insel_molland_reads_its_table_at_the_nodes():
    L, g = 20., 9.81
    dRw = insel_molland.compute_dRw_insel_molland(
        stw=0.4 * math.sqrt(g * L) / u.kt, Lwl=[L, L], y=[-3., 3.], g=g, Rw=[100., 100.],
        fn_grid=[0.3, 0.4, 0.5], sl_grid=[0.2, 0.3, 0.4], tau_table=[[1., 1., 1.], [1., 1.25, 1.], [1., 1., 1.]],
    )
    assert float(dRw[0][1]) == pytest.approx(25.)


#%% ADDED RESISTANCE, KVLCC2 at Fn = 0.142, potential code of Hizir et al. (2019), Table 3

@pytest.mark.parametrize("ratio, potential", [(0.5, 2.457), (1.2, 6.038)])
def test_mittendorf_near_a_potential_code(ratio, potential):
    mean, upper, lower = mittendorf.mm_raw([ratio * 320.], 180., 320., 58., 20.8, 20.8, 0.8098, 60., 60., 0.25, 0.142)[:3]
    assert float(mean[0]) == pytest.approx(potential, rel=0.15)
    assert float(lower[0]) < potential < float(upper[0])


#%% B-SERIES

def performance(blades, area_ratio, pitch_ratio, J):
    propeller = BSeriesPropeller(blades_number=blades, blade_area_ratio=area_ratio, diameter=1., pitch_ratio=pitch_ratio)
    KT = onp.asarray(propeller.compute_kt(J), dtype=float).item()
    KQ = onp.asarray(propeller.compute_kq(J), dtype=float).item()
    return KT, KQ, J * KT / (2. * math.pi * KQ)


@pytest.mark.parametrize("exponent, J, efficiency", [(2, 0.64, 0.61), (4, 0.74, 0.67)])
def test_b_series_on_the_mit_notes(exponent, J, efficiency):
    # MIT 2.611 notes, B5-75, P/D = 1, at the intersection with KT = 0.544 J^n, read to 2 digits
    KT, _, eta = performance(5, 0.75, 1.0, J)
    assert KT == pytest.approx(0.544 * J ** exponent, rel=0.05)
    assert eta == pytest.approx(efficiency, abs=0.01)


def test_b_series_on_the_holtrop_example():
    # P/D = 1.0367 gives the printed KTs = 0.18802 with the viscous correction. The paper prints KQo = 0.033275.
    KT, KQ, _ = performance(4, 0.7393, 1.0367, 0.7184)
    assert KT + 0.000956 * 0.3 * 1.0367 * 3.065 * 4. / 8. == pytest.approx(0.18802, rel=1e-3)
    assert KQ == pytest.approx(0.033275, rel=0.005)
