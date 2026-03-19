"""
SAILING OPERATING POINT DESCRIPTION

Created 18/10/2023
Last update: 18/10/2024

@author: Jules Richeux
@contributors: -

Citation:
    Adapted from:         operating_point module in AeroSandbox
    Author:               Peter D Sharpe
    Date of Retrieval:    18/10/2024

"""

#%% DEPENDENCIES

from typing import Tuple, Union, Dict, List
from archibald2.tools.string_formatting import trim_string
import inspect

from archibald2.tools.env_utils import grad_wind
from archibald2.environment.environment import Environment
import archibald2.tools.units as u

import archibald2.numpy as np


#%% FUNCTIONS

def compute_AW(tws, twa, V):
    """
    Computes apparent wind from true wind.

    Parameters
    ----------
    tws : float. True wind speed in m/s
    twa : float. True wind angle in deg
    V : float. Boat speed in m/s

    Returns
    -------
    Apparent wind speed in m/s
    Apparent wind angle in deg

    """
    
    # Convert inputs to numpy arrays for vectorized operations
    tws = np.asarray(tws)
    twa = np.asarray(twa)
    V = np.asarray(V)
    
    # Calculate true wind components
    TW_x = tws * np.cosd(twa)
    TW_y = tws * np.sind(twa)
    
    # Boat speed components (assuming boat is moving along x-axis)
    SW_x = V
    SW_y = np.zeros_like(V)
    
    # Apparent wind components
    AW_x = TW_x + SW_x
    AW_y = TW_y + SW_y
    
    # Apparent wind speed and angle
    aws = np.sqrt(AW_x**2 + AW_y**2)
    awa = np.arctan2(AW_y, AW_x)
    
    return aws, awa*u.deg


#%% CLASSES

class OperatingPoint():
    def __init__(self,
                 environment: Environment = Environment(),
                 stw: float = 1., # kts
                 tws0: float = 1., # kts
                 twa: float = 0., # deg
                 z0: float = 10., # m
                 a: float = 0.12, # Hellmann coefficient
                 heel: float = 0., # deg
                 trim: float = 0., # deg
                 leeway: float = 0., # deg
                 immersion: float = 0., # m
                 p: float = 0.,
                 q: float = 0.,
                 r: float = 0.,
                 ):
        """
        An object that represents the instantaneous aerodynamic flight conditions of an aircraft.

        Args:
            atmosphere: The atmosphere object (of type asb.Atmosphere). Defaults to sea level conditions.

            tws: The flight velocity, expressed as a true airspeed. [m/s]

            twa: The angle of attack. [degrees]

            beta: The sideslip angle. (Reminder: convention that a positive beta implies that the oncoming air comes
            from the pilot's right-hand side.) [degrees]

            p: The roll rate about the x_b axis. [rad/sec]

            q: The pitch rate about the y_b axis. [rad/sec]

            r: The yaw rate about the z_b axis. [rad/sec]

        """
        self.environment = environment
        self._stw = stw * u.knot
        self._tws0 = tws0 * u.knot
        self._twa = twa
        
        aws0 = np.sqrt(tws0**2 + stw**2 + 2*tws0*stw*np.cosd(twa))
        awa0 = np.arccosd((tws0*np.cosd(twa) + stw)/aws0)
        
        self._aws0 = aws0 * u.knot
        self._awa0 = awa0
        
        self.z0 = z0
        self.a = a
        
        self.beta = 0.
        
        self.heel = heel
        self.trim = trim
        self.leeway = leeway
        self.immersion = immersion
        
        self.p = p
        self.q = q
        self.r = r
    
    @property
    def stw(self):
        """
        Returns speed through water in knots.
        
        """
        return self._stw / u.knot
    
    @property
    def tws0(self):
        """
        Returns true wind speed in knots at reference height z0.
        
        """
        return self._tws0 / u.knot
    
    @property
    def twa(self):
        """
        Returns true wind angle in degrees.
        
        """
        return self._twa
    
    def _tws(self, z=None):
        """
        Returns true wind speed at height z in m/s.
        
        """
        if z is None:
            return self._tws0
        
        return grad_wind(self._tws0, z, self.z0, self.a)
    
    def _aws(self, z=None):
        """
        Returns apparent wind speed at height z in m/s.
        
        """
        tws = self._tws(z)
        twa = self._twa
        stw = self._stw
        
        return np.sqrt(tws**2 + stw**2 + 2*tws*stw*np.cosd(twa))
    
    def awa(self, z=None):
        """
        Returns apparent wind angle in degree at height z in degrees.
        
        """
        tws = self._tws(z)
        twa = self._twa
        stw = self._stw
        
        return np.arccosd((tws*np.cosd(twa) + stw)/self._aws(z))
    
    @property
    def awa0(self):
        """
        Returns apparent wind angle in degrees at reference height z0.
        
        """
        return self.awa(z=self.z0)
    
    def tws(self, z=None):
        """
        Returns true wind speed at height z in knots.
        
        """
        return self._tws(z) / u.knot
    
    def aws(self, z=None):
        """
        Returns apparent wind speed in knots at height z in knots.
        
        """
        return self._aws(z) / u.knot
    
    @property
    def aws0(self):
        """
        Returns apparent wind speed in knots at reference height z0.s.
        
        """
        return self.aws(z=self.z0)

    @property
    def state(self) -> Dict[str, Union[float, np.ndarray]]:
        """
        Returns the state variables of this OperatingPoint instance as a Dict.

        Keys are strings that give the name of the variables.
        Values are the variables themselves.

        """
        return {
            "environment"   : self.environment,
            "STW"           : self._stw,
            "TWS0"          : self._tws0,
            "TWA"           : self._twa,
            "z0"            : self.z0,
            "a"             : self.a,
            "p"             : self.p,
            "q"              : self.q,
            "r"              : self.r,
        }

    def get_new_instance_with_state(self,
                                    new_state: Union[
                                        Dict[str, Union[float, np.ndarray]],
                                        List, Tuple, np.ndarray
                                    ] = None
                                    ):
        """
        Creates a new instance of the OperatingPoint class from the given state.

        Args:
            new_state: The new state to be used for the new instance. Ideally, this is represented as a Dict in identical format to the `state` of a OperatingPoint instance.

        Returns: A new instance of this same OperatingPoint class.

        """

        ### Get a list of all the inputs that the class constructor wants to see
        init_signature = inspect.signature(self.__class__.__init__)
        init_args = list(init_signature.parameters.keys())[1:]  # Ignore 'self'

        ### Create a new instance, and give the constructor all the inputs it wants to see (based on values in this instance)
        new_op_point: __class__ = self.__class__(**{
            k: getattr(self, k)
            for k in init_args
        })

        ### Overwrite the state variables in the new instance with those from the input
        new_op_point._set_state(new_state=new_state)

        ### Return the new instance
        return new_op_point

    def _set_state(self,
                   new_state: Union[
                       Dict[str, Union[float, np.ndarray]],
                       List, Tuple, np.ndarray
                   ] = None
                   ):
        """
        Force-overwrites all state variables with a new set (either partial or complete) of state variables.

        Warning: this is *not* the intended public usage of OperatingPoint instances.
        If you want a new state yourself, you should instantiate a new one either:
            a) manually, or
            b) by using OperatingPoint.get_new_instance_with_state()

        Hence, this function is meant for PRIVATE use only - be careful how you use this!
        """
        ### Set the default parameters
        if new_state is None:
            new_state = {}

        try:  # Assume `value` is a dict-like, with keys
            for key in new_state.keys():  # Overwrite each of the specified state variables
                setattr(self, key, new_state[key])

        except AttributeError:  # Assume it's an iterable that has been sorted.
            self._set_state(
                self.pack_state(new_state))  # Pack the iterable into a dict-like, then do the same thing as above.

    def unpack_state(self,
                     dict_like_state: Dict[str, Union[float, np.ndarray]] = None
                     ) -> Tuple[Union[float, np.ndarray]]:
        """
        'Unpacks' a Dict-like state into an array-like that represents the state of the OperatingPoint.

        Args:
            dict_like_state: Takes in a dict-like representation of the state.

        Returns: The array representation of the state that you gave.

        """
        if dict_like_state is None:
            dict_like_state = self.state
        return tuple(dict_like_state.values())

    def pack_state(self,
                   array_like_state: Union[List, Tuple, np.ndarray] = None
                   ) -> Dict[str, Union[float, np.ndarray]]:
        """
        'Packs' an array into a Dict that represents the state of the OperatingPoint.

        Args:
            array_like_state: Takes in an iterable that must have the same number of entries as the state vector of the OperatingPoint.

        Returns: The Dict representation of the state that you gave.

        """
        if array_like_state is None:
            return self.state
        if not len(self.state.keys()) == len(array_like_state):
            raise ValueError(
                "There are a differing number of elements in the `state` variable and the `array_like` you're trying to pack!")
        return {
            k: v
            for k, v in zip(
                self.state.keys(),
                array_like_state
            )
        }

    def __repr__(self) -> str:

        title = f"{self.__class__.__name__} instance:"

        def makeline(k, v):
            name = trim_string(str(k).strip(), length=10).rjust(10)
            item = trim_string(str(v).strip(), length=120).ljust(120)

            line = f"{name}: {item}"

            return line

        state_variables_title = "\tState variables:"

        state_variables = "\n".join([
            "\t\t" + makeline(k, v)
            for k, v in self.state.items()
        ])

        return "\n".join([
            title,
            state_variables_title,
            state_variables,
        ])

    def __getitem__(self, index: int) -> "OperatingPoint":
        """
        Indexes one item from each attribute of an OperatingPoint instance.
        Returns a new OperatingPoint instance.

        Args:
            index: The index that is being called; e.g.,:
                >>> first_dyn = op_point[0]

        Returns: A new OperatingPoint instance, where each attribute is subscripted at the given value, if possible.

        """

        def get_item_of_attribute(a):
            try:
                return a[index]
            except TypeError as e:  # object is not subscriptable
                return a
            except IndexError as e:  # index out of range
                raise IndexError("A state variable could not be indexed, since the index is out of range!")
            except NotImplementedError as e:
                raise TypeError(f"Indices must be integers or slices, not {index.__class__.__name__}")

        new_instance = self.get_new_instance_with_state()

        for k, v in new_instance.__dict__.items():
            setattr(new_instance, k, get_item_of_attribute(v))

        return new_instance

    def __len__(self):
        length = 1
        for v in self.state.values():
            if np.length(v) == 1:
                pass
            elif length == 1:
                length = np.length(v)
            elif length == np.length(v):
                pass
            else:
                raise ValueError("State variables are appear vectorized, but of different lengths!")
        return length

    def air_dynamic_pressure(self, z = None):
        """
        Dynamic pressure of the working fluid
        Returns:
            float: Dynamic pressure of the working fluid. [Pa]
        """
        if z == None:
            z = self.z0
        
        awsZ = self._aws(z)
        
        return 0.5 * self.environment.air.density * awsZ ** 2

    def water_dynamic_pressure(self):
        """
        Dynamic pressure of the working fluid
        Returns:
            float: Dynamic pressure of the working fluid. [Pa]
        """        
        return 0.5 * self.environment.water.density * self._stw ** 2

    def air_reynolds(self, reference_length, z=None):
        """
        Computes a Reynolds number with respect to a given reference length.
        :param reference_length: A reference length you choose [m]
        :return: Reynolds number [unitless]
        """
        density = self.environment.air.density
        viscosity = self.environment.air.dynamic_viscosity

        return density * self.aws(z) * reference_length / viscosity

    def water_reynolds(self, reference_length):
        """
        Computes a Reynolds number with respect to a given reference length.
        :param reference_length: A reference length you choose [m]
        :return: Reynolds number [unitless]
        """
        density = self.environment.water.density
        viscosity = self.environment.water.dynamic_viscosity

        return density * self._stw * reference_length / viscosity

    def convert_axes(self,
                     x_from: Union[float, np.ndarray],
                     y_from: Union[float, np.ndarray],
                     z_from: Union[float, np.ndarray],
                     from_axes: str,
                     to_axes: str,
                     ) -> Tuple[float, float, float]:
        """
        Converts a vector [x_from, y_from, z_from], as given in the `from_axes` frame, to an equivalent vector [x_to,
        y_to, z_to], as given in the `to_axes` frame.

        Both `from_axes` and `to_axes` should be a string, one of:
            * "geometry"
            * "body"
            * "wind"
            * "stability"

        This whole function is vectorized, both over the vector and the OperatingPoint (e.g., a vector of
        `OperatingPoint.alpha` values)

        Wind axes rotations are taken from Eq. 6.7 in Sect. 6.2.2 of Drela's Flight Vehicle Aerodynamics textbook,
        with axes corrections to go from [D, Y, L] to true wind axes (and same for geometry to body axes).

        Args:
            x_from: x-component of the vector, in `from_axes` frame.
            y_from: y-component of the vector, in `from_axes` frame.
            z_from: z-component of the vector, in `from_axes` frame.
            from_axes: The axes to convert from.
            to_axes: The axes to convert to.

        Returns: The x-, y-, and z-components of the vector, in `to_axes` frame. Given as a tuple.

        """
        if from_axes == to_axes:
            return x_from, y_from, z_from

        if from_axes == "geometry":
            x_b = -x_from
            y_b = -y_from
            z_b = z_from
        elif from_axes == "underway":
            x_b = x_from
            y_b = y_from
            z_b = z_from
        elif from_axes == "wind":
            sa = np.sind(self.awa0)
            ca = np.cosd(self.awa0)
            x_b = ca * x_from - sa * y_from
            y_b = sa * x_from + ca * y_from
            z_b = z_from
        elif to_axes == "stability":
            sa = np.sind(self.awa0)
            ca = np.cosd(self.awa0)
            x_b = ca * x_from - sa * y_from
            y_b = sa * x_from + ca * y_from
            z_b = z_from
        else:
            raise ValueError("Bad value of `from_axes`!")

        if to_axes == "geometry":
            x_to = -x_b
            y_to = -y_b
            z_to = z_b
        elif to_axes == "underway":
            x_to = x_b
            y_to = y_b
            z_to = z_b
        elif to_axes == "wind":
            sa = np.sind(self.awa0)
            ca = np.cosd(self.awa0)
            x_to = ca * x_b - sa * y_b
            y_to = sa * x_b + ca * y_b
            z_to = z_from
        elif to_axes == "stability":
            sa = np.sind(self.awa0)
            ca = np.cosd(self.awa0)
            x_to = ca * x_b + sa * y_b
            y_to = -sa * x_b + ca * y_b
            z_to = z_b
        else:
            raise ValueError("Bad value of `to_axes`!")

        return np.array([x_to, y_to, z_to])

    def compute_rotation_matrix_wind_to_geometry(self) -> np.ndarray:
        """
        Computes the 3x3 rotation matrix that transforms from wind axes to geometry axes.

        Returns: a 3x3 rotation matrix.

        """

        alpha_rotation = np.rotation_matrix_3D(
            angle=np.radians(-self.alpha),
            axis="y",
        )
        beta_rotation = np.rotation_matrix_3D(
            angle=np.radians(self.beta),
            axis="z",
        )
        axes_flip = np.rotation_matrix_3D(
            angle=np.pi,
            axis="y",
        )
        # Since in geometry axes, X is downstream by convention, while in wind axes, X is upstream by convention.
        # Same with Z being up/down respectively.

        r = axes_flip @ alpha_rotation @ beta_rotation  # where "@" is the matrix multiplication operator

        return r

    def compute_rotation_velocity_geometry_axes(self, points):
        # Computes the effective velocity-due-to-rotation at a set of points.
        # Input: a Nx3 array of points
        # Output: a Nx3 array of effective velocities
        angular_velocity_vector_geometry_axes = np.array([
            -self.p,
            self.q,
            -self.r
        ])  # signs convert from body axes to geometry axes

        a = angular_velocity_vector_geometry_axes
        b = points

        rotation_velocity_geometry_axes = np.stack([
            a[1] * b[:, 2] - a[2] * b[:, 1],
            a[2] * b[:, 0] - a[0] * b[:, 2],
            a[0] * b[:, 1] - a[1] * b[:, 0]
        ], axis=1)

        rotation_velocity_geometry_axes = -rotation_velocity_geometry_axes  # negative sign, since we care about the velocity the WING SEES, not the velocity of the wing.

        return rotation_velocity_geometry_axes


if __name__ == '__main__':
    # op_point = OperatingPoint()
    
    import archibald2 as arb
    
    opti = arb.Opti()

    z = opti.variable(init_guess=2., lower_bound=0.)
    
    op_point = OperatingPoint(
              stw=10., # kts
              tws0=10., # kts
              twa=45., # deg
              z0=10., # m
              a=0.12,
              )

    obj = (op_point.tws(z) - 11.) ** 2

    # opti.subject_to(
    #     aero["CL"] == 0.5
    # )

    opti.minimize(obj)
    
    sol = opti.solve()
    
    print(sol(z))
