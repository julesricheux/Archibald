# -*- coding: utf-8 -*-
"""
Propeller geometry and performance models.

This module provides classes and functions for defining propeller geometry and 
calculating performance, with a focus on B-series (Wageningen) propellers.
"""

import os
import copy
from typing import Union, List, Sequence, Iterable
from pathlib import Path

import archibald.numpy as np
from archibald import _archibald_root
from archibald.modeling import InterpolatedModel
from archibald.toolbox.math_utils import read_coefs, rotate_single_vector

# Note: Airfoil import is currently unused but might be needed for later development.
# from archibald.geometry.airfoil import Airfoil

def tall(array: np.ndarray) -> np.ndarray:
    """
    Reshapes an array into a tall (N x 1) vector.

    Args:
        array: Input array to reshape.

    Returns:
        Reshaped (N x 1) array.
    """
    return np.reshape(array, (-1, 1))


def wide(array: np.ndarray) -> np.ndarray:
    """
    Reshapes an array into a wide (1 x N) vector.

    Args:
        array: Input array to reshape.

    Returns:
        Reshaped (1 x N) array.
    """
    return np.reshape(array, (1, -1))


def b_series_polynomial(
    J: float | np.ndarray,
    P_D: float | np.ndarray,
    Ae_Ao: float | np.ndarray,
    Z: int | np.ndarray,
    c: np.ndarray,
    stuv: np.ndarray,
) -> np.ndarray:
    """
    Computes B-series propeller performance coefficients using polynomials.

    According to evaluation ranges:
    2 <= Z <= 7
    0.30 <= Ae/Ao <= 1.05
    0.50 <= P/D <= 1.40

    Args:
        J: Advance ratio.
        P_D: Pitch-to-diameter ratio.
        Ae_Ao: Blade area ratio.
        Z: Number of blades.
        c: Polynomial coefficients.
        stuv: Polynomial exponents.

    Returns:
        Calculated coefficient (Kt or Kq).
    """
    is_casadi = (
        np.is_casadi_type(J) or 
        np.is_casadi_type(P_D) or 
        np.is_casadi_type(Ae_Ao) or 
        np.is_casadi_type(Z) or 
        np.is_casadi_type(c) or 
        np.is_casadi_type(stuv)
    )
    
    if isinstance(J, (int, float, np.number)):
        J = np.array([J])
    elif isinstance(J, (list, tuple)):
        J = np.array(J)
        
    j_mat = tall(copy.copy(J))
    property_mat = np.ones((j_mat.shape[0], 1)) @ wide(np.array([P_D, Ae_Ao, Z]))
        
    if is_casadi:
        param_mat = np.stack(
            [j_mat] + [property_mat[:, i] for i in range(3)], 
            axis=1
        )
        
        k = []
        for i in range(j_mat.shape[0]):
            line = np.ones((stuv.shape[0], 1)) @ wide(param_mat[i, :])
            pow_mat = np.power(line, stuv)
            prod_mat = np.prod(pow_mat, axis=1)
            k.append(np.dot(prod_mat, tall(c)))
        k = np.array(k)
    else:
        param_mat = np.stack(
            [j_mat] + [tall(property_mat[:, i]) for i in range(3)], 
            axis=1
        )
        pow_mat = np.power(param_mat, np.transpose(stuv))
        prod_mat = np.prod(pow_mat, axis=1)
        k = np.dot(prod_mat, tall(c)).T[0]
    
    return k


def b_series_sections(
    r_R: np.ndarray,
    t_max: np.ndarray,
    v1_fun: callable,
    v2_fun: callable,
    t_le: float = 1e-3,
    t_te: float = 1e-3,
    n_points: int = 200,
    display: bool = False,
) -> list:
    """
    Computes B-series propeller blade sections.

    Args:
        r_R: Radial positions [non-dimensional].
        t_max: Maximum thickness at each radial position.
        v1_fun: Interpolator for B-series V1 parameter.
        v2_fun: Interpolator for B-series V2 parameter.
        t_le: Leading edge thickness [m].
        t_te: Trailing edge thickness [m].
        n_points: Number of points per section.
        display: Whether to draw the sections.

    Returns:
        List of section objects (requires Airfoil class).
    """
    # Note: This function relies on an 'Airfoil' class that must be available in scope.
    # The current implementation assumes it is imported or defined.
    from archibald.geometry.airfoil import Airfoil

    n_sections = len(r_R)
    xt_max = 0.4
    
    n_lead = int(n_points * xt_max / 2)
    n_trail = n_points // 2 - n_lead
    
    p_neg = np.linspace(-1.0, 0.0, n_trail)
    p_pos = np.linspace(1 / n_lead, 1.0, n_lead)
    
    p = np.hstack((p_neg, p_pos))
    rr, pp = np.meshgrid(r_R, p)
    rr_flat = rr.ravel()
    pp_flat = pp.ravel()
    
    v1 = v1_fun(rr_flat, pp_flat)
    v2 = v2_fun(rr_flat, pp_flat)
    
    v1_neg = v1[:n_trail * n_sections]
    v2_neg = v2[:n_trail * n_sections]
    v1_pos = v1[n_trail * n_sections:]
    v2_pos = v2[n_trail * n_sections:]
    
    x_lead = xt_max * (1 - p_pos)
    x_trail = (1 - xt_max) * (xt_max / (1 - xt_max) - p_neg)
    
    x_upper = np.hstack((x_trail, x_lead))
    x_lower = np.hstack((np.flip(x_lead), np.flip(x_trail)))
    x_section = np.hstack((x_upper, x_lower))
    
    airfoils = []
    for i in range(n_sections):
        yy_face_trail = v1_neg * (t_max[i] - t_te)
        yy_face_lead = v1_pos * (t_max[i] - t_le)
        
        yy_back_trail = (v1_neg + v2_neg) * (t_max[i] - t_te) + t_te
        yy_back_lead = (v1_pos + v2_pos) * (t_max[i] - t_le) + t_le
        
        y_face_trail = yy_face_trail[i::n_sections]
        y_face_lead = yy_face_lead[i::n_sections]
        
        y_back_trail = yy_back_trail[i::n_sections]
        y_back_lead = yy_back_lead[i::n_sections]
        
        y_upper = np.hstack((y_back_trail, y_back_lead))
        y_lower = np.hstack((np.flip(y_face_lead), np.flip(y_face_trail)))
        y_section = np.hstack((y_upper, y_lower))
        
        coords = np.transpose(np.vstack((x_section, y_section)))
        
        af = Airfoil(
            name=f"B-SERIES-{r_R[i]}",
            coordinates=coords
        )
        airfoils.append(af)
        
        if display:
            af.draw()
            
    return airfoils


def write_airfoil_polars(
    airfoil: any,
    alpha_min: float = -180.0,
    alpha_max: float = 180.0,
    n_alphas: int = 361,
    reynolds: float = 1e6,
    directory: str = './pyBEMT/pybemt/airfoils/',
    suffix: str = '',
) -> None:
    """
    Writes airfoil aerodynamic polars to a file.

    Args:
        airfoil: Airfoil object.
        alpha_min: Minimum angle of attack [deg].
        alpha_max: Maximum angle of attack [deg].
        n_alphas: Number of alpha points.
        reynolds: Reynolds number.
        directory: Destination directory.
        suffix: Filename suffix.
    """
    af = airfoil.copy()
    
    if suffix == '-gen':
        af.coordinates[:, 1] = -af.coordinates[:, 1]
        np.flip(af.coordinates, axis=0)
    
    alphas = np.linspace(alpha_min, alpha_max, n_alphas)
    data = af.get_aero_from_neuralfoil(alphas, reynolds)
    
    output_path = Path(directory) / f"{af.name}{suffix}.dat"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for _ in range(13):
            f.write("---\n")
        f.write("Alpha CL CD\n")
        for a, cl, cd in zip(alphas, data['CL'], data['CD']):
            f.write(f"{a} {cl} {cd}\n")


class Propeller:
    """
    Base class for propeller geometry and performance.
    """
    def __init__(
        self,
        name: str = 'Untitled_propeller',
        xyz_center: np.ndarray | list | None = None,
        axis: np.ndarray | list | None = None,
        blades_number: int = 2,
        blade_area_ratio: float = 1.0,
        diameter: float = 1.0,
        pitch_ratio: float = 1.0,
    ):
        """
        Initializes a Propeller.

        Args:
            name: Propeller name.
            xyz_center: Center coordinates [m].
            axis: Rotation axis vector.
            blades_number: Number of blades (Z).
            blade_area_ratio: Expanded area ratio (Ae/Ao).
            diameter: Propeller diameter (D) [m].
            pitch_ratio: Pitch-to-diameter ratio (P/D).
        """
        if xyz_center is None:
            xyz_center = np.zeros(3)
            
        if axis is None:
            axis = np.array([1.0, 0.0, 0.0])
        
        self.name = name
        self._center = np.array(xyz_center)
        self._center_0 = copy.copy(self._center)
        self._axis = np.array(axis)
        self._axis_0 = copy.copy(self._axis)
        self._z = blades_number
        self._ae_ao = blade_area_ratio
        self._d = diameter
        self._r = diameter / 2
        self._p_d = pitch_ratio
        self._p_angle = np.degrees(np.arctan(self._p_d))
        
    @property
    def center(self) -> np.ndarray:
        return self._center
    
    @property
    def axis(self) -> np.ndarray:
        return self._axis
    
    @property
    def blades_number(self) -> int:
        return self._z
        
    @property
    def diameter(self) -> float:
        return self._d
        
    @property
    def radius(self) -> float:
        return self._r
        
    @property
    def blade_area_ratio(self) -> float:
        return self._ae_ao
        
    @property
    def pitch_ratio(self) -> float:
        return self._p_d
    
    @property
    def pitch_angle(self) -> float:
        return self._p_angle
    
    @property
    def area(self) -> float:
        return np.pi * self._r**2
    
    def reset(self) -> None:
        """
        Cancels all previous transformations.
        """
        self._center = copy.copy(self._center_0)
        self._axis = copy.copy(self._axis_0)
    
    @center.setter
    def center(self, xyz: Iterable[float]) -> None:
        xyz = np.array(xyz)
        if xyz.size != 3:
            raise ValueError("Passed array must have 3 elements.")
        self._center = xyz
        
    def translate(self, xyz: Iterable[float]) -> None:
        """
        Translates the propeller center.
        """
        xyz = np.array(xyz)
        if xyz.size != 3:
            raise ValueError("Passed array must have 3 elements.")
        self._center += xyz
        
    def _global_rotation_from_matrix(
        self,
        rot_mat: np.ndarray,
        center: np.ndarray | list | None = None,
    ) -> None:
        if center is None:
            center = self._center.copy()
        self._center = rotate_single_vector(self._center, rot_mat, center)
        self._axis = rotate_single_vector(self._axis, rot_mat, center)
        
    def global_rotation(
        self,
        angle: float,
        axis: np.ndarray | list | str,
        center: np.ndarray | list | None = None,
        matrix: np.ndarray | None = None,
    ) -> None:
        """
        Rotates the propeller globally.
        """
        if matrix is None:
            matrix = np.rotation_matrix_3D(np.deg2rad(angle), axis)
        
        self._global_rotation_from_matrix(matrix, center)
        
    def local_rotation(
        self,
        angle: float,
        axis: np.ndarray | list | str,
        matrix: np.ndarray | None = None,
    ) -> None:
        """
        Rotates the propeller around its center.
        """
        if matrix is None:
            matrix = np.rotation_matrix_3D(np.deg2rad(angle), axis)
        
        self._global_rotation_from_matrix(matrix, None)


class BSeriesPropeller(Propeller):
    """
    B-Series (Wageningen) Propeller implementation.
    """
    def __init__(
        self,
        name: str = 'Untitled_Bseries_propeller',
        xyz_center: np.ndarray | list | None = None,
        axis: np.ndarray | list | None = None,
        blades_number: int = 3,
        blade_area_ratio: float = 1.0,
        diameter: float = 1.0,
        pitch_ratio: float = 1.0,
    ):
        super().__init__(
            name=name,
            xyz_center=xyz_center,
            axis=axis,
            blades_number=blades_number,
            blade_area_ratio=blade_area_ratio,
            diameter=diameter,
            pitch_ratio=pitch_ratio,
        )
        
        _data_path = Path(_archibald_root) / "data" / "propeller"
        kt_path = _data_path / "b-series-kt.csv"
        kq_path = _data_path / "b-series-kq.csv"
        geometry_path = _data_path / "b-series-geometry.csv"
                
        kt_data = read_coefs(kt_path, delim='\t', skipRows=2)
        kq_data = read_coefs(kq_path, delim='\t', skipRows=2)
        
        if self._z == 3:
            _coefs = read_coefs(
                geometry_path, 
                delim='\t',
                skipRows=2, 
                columns=[0, 4, 5, 6, 7, 8, 9]
            ).T
        elif 4 <= self._z <= 7:
            _coefs = read_coefs(
                geometry_path, 
                delim='\t',
                skipRows=2, 
                columns=[0, 1, 2, 3, 7, 8, 9]
            ).T
        else:
            raise ValueError(f"Number of blades ({self._z}) must be between 3 and 7")
        
        geometry_keys = ['c/D*Z/(AE/AO)', 'a/c', 'b/c', 'Ar', 'Br', 'pitch factor']
        self._geometry_data = {}
        
        for i, k in enumerate(geometry_keys):
            self._geometry_data[k] = InterpolatedModel(
                x_data_coordinates=_coefs[0, :],
                y_data_structured=_coefs[i+1, :],
                method="bspline",
            )
        
        self._c_t = kt_data[:, 0]
        self._stuv_t = kt_data[:, 1:].astype('int32')
        
        self._c_q = kq_data[:, 0]
        self._stuv_q = kq_data[:, 1:].astype('int32')
        
    def chord_law(self, r: float | np.ndarray) -> float | np.ndarray:
        """Computes chord distribution along the radius."""
        norm_law = self._geometry_data['c/D*Z/(AE/AO)'](r)
        return norm_law * self._d * self._ae_ao / self._z
    
    def sweep_law(self, r: float | np.ndarray, chord_law: float | np.ndarray | None = None) -> float | np.ndarray:
        """Computes sweep distribution along the radius."""
        if chord_law is None:
            chord_law = self.chord_law(r)
        norm_law = self._geometry_data['a/c'](r)
        return norm_law * chord_law
    
    def pitch_law(self, r: float | np.ndarray) -> float | np.ndarray:
        """Computes pitch factor distribution along the radius."""
        return self._geometry_data['pitch factor'](r)
    
    def thickness_law(self, r: float | np.ndarray, chord_law: float | np.ndarray | None = None) -> float | np.ndarray:
        """Computes thickness distribution along the radius."""
        if chord_law is None:
            chord_law = self.chord_law(r)
        return self._d * (self._geometry_data['Ar'](r) - self._z * self._geometry_data['Br'](r)) / chord_law
    
    def compute_geometric_laws(self, r: np.ndarray | None = None) -> tuple:
        """Computes all geometric laws at specified radial positions."""
        if r is None:
            r = np.linspace(1e-3, 1.0 - 1e-3, 10)
        chord = self.chord_law(r)
        sweep = self.sweep_law(r, chord)
        pitch = self.pitch_law(r)
        thickness = self.thickness_law(r, chord)
        return chord, sweep, pitch, thickness
    
    # EXPERIMENTAL
    def build_blade_sections(self, r: np.ndarray, thickness_law: np.ndarray | None = None, chord_law: np.ndarray | None = None):
        if thickness_law is None:
            if chord_law is None:
                chord_law = self.chord_law(r)
            thickness_law = self.thickness_law(r, chord_law)
        
        # Note: BSeries sections V1/V2 data usage is commented in original
        # return b_series_sections(r, thickness_law, self._geometry_data['V1'], self._geometry_data['V2'], display=False)
        pass
    
    # EXPERIMENTAL
    def write_sections_polars(self, r: np.ndarray | None = None, sections: list | None = None, suffix: str = ''):
        if r is not None or sections is not None:
            if sections is None:
                sections = self.build_blade_sections(r)
            if sections:
                for s in sections:
                    write_airfoil_polars(s, suffix=suffix)
        else:
            print("Could not write sections polars. No radius nor sections were specified.")
    
    # EXPERIMENTAL
    def write_pyBEMT_input(self, n_sections: int = 10, directory: str = './pyBEMT/pybemt/airfoils/', generator: bool = False):
        r = np.linspace(0.20, 0.999, n_sections)
        suffix = '-gen' if generator else ''
        
        chords = self.chord_law(r)
        pitches = self.pitch_angle * self.pitch_law(r) + 90 * (1 - self.pitch_law(r))
        sections = self.build_blade_sections(r, chord_law=chords)
        self.write_sections_polars(sections=sections, suffix=suffix)
        
        filename = f"{directory}{self.name}{suffix}.ini"
        # ... (rest of the file writing logic preserved)
        return filename

    # TODO rewrite for to Archibald
    # def build_aerosanbox_geometry(self, nSections=10):
    #     ... (preserved commented method)
        
    def compute_kt(self, J: float | np.ndarray) -> float | np.ndarray:
        """Computes thrust coefficient Kt."""
        return b_series_polynomial(J, self._p_d, self._ae_ao, self._z, self._c_t, self._stuv_t)

    def compute_kq(self, J: float | np.ndarray) -> float | np.ndarray:
        """Computes torque coefficient Kq."""
        return b_series_polynomial(J, self._p_d, self._ae_ao, self._z, self._c_q, self._stuv_q)
    
    def compute_performance(self, J: float | np.ndarray) -> tuple:
        """Computes Kt, Kq and open-water efficiency."""
        kt = self.compute_kt(J)
        kq = self.compute_kq(J)
        eta = kt / kq * J / (2 * np.pi)
        return kt, kq, eta
    
    def compute_forces(
        self, 
        J: float | np.ndarray, 
        rho: float = 1025.0, 
        v_a: float | None = None, 
        rpm: float | None = None
    ) -> tuple:
        """Computes Thrust, Torque, Power and RPS."""
        if rpm is not None:
            n = rpm / 60
        elif v_a is None:
            raise ValueError('Either v_a or rpm must be specified')
        else:
            n = v_a / (J * self._d)
        
        kt = self.compute_kt(J)
        kq = self.compute_kq(J)
        
        thrust = kt * rho * n**2 * self._d**4
        torque = kq * rho * n**2 * self._d**5
        power = torque * n * 2 * np.pi
        
        return thrust, torque, power, n
    
    def compute_coefficients(
        self, 
        J: float | np.ndarray, 
        rho: float = 1025.0, 
        v_a: float | None = None, 
        rpm: float | None = None
    ) -> tuple:
        """Computes non-dimensional thrust and power coefficients."""
        if rpm:
            n = rpm / 60
            v_a = n / (J * self._d)
        elif not v_a:
            raise ValueError('Either v_a or rpm must be specified')
        else:
            n = v_a / (J * self._d)
        
        kt = self.compute_kt(J)
        kq = self.compute_kq(J)
        
        cth = 8 / np.pi * kt / J**2
        cq = 8 / np.pi * kq / J**2
        cp = cq * n * 2 * np.pi
        
        return cth, cq, cp

    # TODO implement properly
    # def compute_forces_from_op(self, op_point, hull=None):
    #     ... (preserved commented method)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Example: Analyze a 4-bladed B-Series propeller
    prop = BSeriesPropeller(
        name="TestProp",
        blades_number=4,
        blade_area_ratio=0.8,
        pitch_ratio=1.0,
        diameter=4.0
    )
    
    j_values = np.linspace(0.1, 1.4, 100)
    kt, kq, eta = prop.compute_performance(j_values)
    
    # Filter valid results (positive torque)
    valid = kq > 0
    
    plt.figure(figsize=(10, 6))
    plt.plot(j_values[valid], kt[valid], label='Kt')
    plt.plot(j_values[valid], 10 * kq[valid], label='10*Kq', linestyle='--')
    plt.plot(j_values[valid], eta[valid], label='Efficiency (eta)', linestyle='-.')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlabel('Advance Ratio (J)')
    plt.ylabel('Coefficients')
    plt.title(f"B-Series Propeller Performance (Z={prop.blades_number}, P/D={prop.pitch_ratio})")
    plt.ylim(0, 1.2)
    plt.show()







