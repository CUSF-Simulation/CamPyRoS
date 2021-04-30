"""
Contains the classes and functions for the core trajectory simulation. SI units unless stated otherwise.
Notes
-----
Known issues:
- Unsure about the use of "dx" in "scipy.misc.derivative(self.mass_model.mass, time, dx=1)" when calculating mdot
- Possible inconsistency in the definition of the launch site coordinate system, and whether the origin is at alt=0 or alt=launch_site.alt. I haven't thoroughly checked for this inconsistency yet.
Coordinate systems:
- Body (x_b, y_b, z_b)
    - Origin on rocket
    - Rotates with the rocket.
    - y points east and z north at take off (before rail alignment is accounted for) x up.
    - x is along the "long" axis of the rocket.
- Launch site (x_l, y_l, z_l):
    - Origin has the launch site's longitude and latitude, but is at altitude = 0.
    - Rotates with the Earth.
    - z points up (normal to the surface of the Earth).
    - y points East (tangentially to the surface of the Earth).
    - x points South (tangentially to the surface of the Earth).      
- Inertial (x_i, y_i, z_i):
    - Origin at centre of the Earth.
    - Does not rotate.
    - z points to North from the centre of Earth.
    - x aligned with launch site at start .
    - y defined from x and z (so it is a right hand coordinate system).
"""

import csv
import warnings
import os
import sys
import json
import requests
import os.path
import time
import numpy as np
import pandas as pd

import scipy.interpolate as interpolate
import scipy.misc
import scipy.integrate as integrate
from scipy.spatial.transform import Rotation
import numexpr as ne

from datetime import date, timedelta
from ambiance import Atmosphere

from .constants import r_earth, ang_vel_earth, f
from .transforms import (
    pos_l2i,
    pos_i2l,
    vel_l2i,
    vel_i2l,
    direction_l2i,
    direction_i2l,
    i2airspeed,
    i2lla,
    pos_i2alt,
)

from .wind import Wind

__copyright__ = """
    Copyright 2021 Jago Strong-Wright & Daniel Gibbons
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

    Modifcations:
    - Richard Zhang: Mach varying parachute drag coefficients
"""

# print("""<name tbc>  Copyright (C) 2021  Jago Strong-Wright & Daniel Gibbons
#    This program comes with ABSOLUTELY NO WARRANTY; for details see licence.txt""")


def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    """A one line warning format
    Args:
        message ([type]): [description]
        category ([type]): [description]
        filename ([type]): [description]
        lineno ([type]): [description]
        file ([type], optional): [description]. Defaults to None.
        line ([type], optional): [description]. Defaults to None.
    Returns:
        [type]: [description]
    """
    return "%s:%s: %s:%s\n" % (filename, lineno, category.__name__, message)


warnings.formatwarning = warning_on_one_line


class Parachute:
    def __init__(
        self, main_s, main_c_d, drogue_s, drogue_c_d, main_alt, attach_distance=0.0
    ):
        """
        Object holding the parachute information
        Note
        ----
        The parachute model does not currently simulate the full rotational dynamics of the rocket.
        Instead it orientates the rocket such that it is "rear first into the wind" (as intuition would suggest).
        This is due to problems trying to model the parachute exerting torque on the body, possibly because it has to flip
        the rocket over at apogee
        Args:
            main_s (float): Area of main chute (m^2)
            drogue_s (float): Area of the main parachute (m^2)
            main_alt (float): Altitude at which the main parachute deploys (m)
            parachute_Cd: pandas.dataframe object, read from Parachute_data.CSV
            attach_distance (float, optional): Distance between the rocket nose tip and the parachute attachment point (m). Defaults to 0.0.
            main_cd_data (tuple): tuple of 2 np arrays, of mach number and Main Cd
            dro_cd_data (tuple): tuple of 2 np arrays, of mach number and Drogue Cd
        Attributes:
            main_s (float): Area of main chute (m^2)
            drogue_s (float): Area of the main parachute (m^2)
            main_alt (float): Altitude at which the main parachute deploys (m)
            attach_distance (float, optional): Distance between the rocket nose tip and the parachute attachment point (m). Defaults to 0.0.
            main_cd_data (tuple): tuple of 2 np arrays, of mach number and Main Cd
            dro_cd_data (tuple): tuple of 2 np arrays, of mach number and Drogue Cd
            dro_cd(Mach) (interp1d object): give Drogue Cd based on Mach
            main_cd(Mach) (interp1d object): give Main Cd based on Mach
        """
        self.main_s = main_s
        self.drogue_s = drogue_s
        self.main_alt = main_alt
        self.attach_distance = attach_distance

        # get Drogue and Main Cd data, note bound error. Support tuple/list of np array(s)
        if isinstance(main_c_d, float) or isinstance(main_c_d, int):
            self.variable_main_c_d = False
            self.main_c_d = main_c_d
        elif len(main_c_d) == 2:  # list, tuple, np.array
            self.variable_main_c_d = True
            self.main_c_d = interpolate.interp1d(
                main_c_d[0],
                main_c_d[1],
                copy=True,
                bounds_error=False,
                fill_value=(0, main_c_d[0][-1]),
            )
        if isinstance(drogue_c_d, float) or isinstance(drogue_c_d, int):
            self.variable_drogue_c_d = False
            self.drogue_c_d = drogue_c_d
        elif len(drogue_c_d) == 2:  # list, tuple, np.array
            self.variable_drogue_c_d = True
            self.drogue_cd = interpolate.interp1d(
                drogue_c_d[0],
                drogue_c_d[1],
                copy=True,
                bounds_error=False,
                fill_value=(0, drogue_c_d[0][-1]),
            )

    def get(self, alt, mach):
        """Returns the drag coefficient and area of the parachute, given the current altitude and Mach Number.
        I.e., it checks if the main or drogue parachute is open, and returns the relevant values.
        Args:
            alt (float): Current altitude (m)
            mach (float): Mach number
        Returns:
            float, float: Drag coefficient, parachute area (m^2)
        """
        # if Mach < 0 or Mach > 7:
        #    print("Mach Number is {}, out of range!".format(Mach))
        if alt < self.main_alt:
            s = self.main_s
            if self.variable_main_c_d == False:
                c_d = self.main_c_d
            elif self.variable_main_c_d == True:
                c_d = self.main_c_d(mach)
        else:
            s = self.drogue_s
            if self.variable_main_c_d == False:
                c_d = self.drogue_c_d
            elif self.variable_main_c_d == True:
                c_d = self.drogue_cd(mach)
        return c_d, s


class LaunchSite:
    """Object for holding launch site information.
    Args:
        rail_length (float): Length of the launch rail (m)
        rail_yaw (float): Yaw angle of the launch rail (deg), using a right-hand rotation rule out the launch frame z-axis. "rail_yaw = 0" points South, "rail_yaw = 90" points East.
        rail_pitch (float): Pitch angle of the launch rail (deg). "rail_pitch = 0" points up.
        alt (float): Launch site altitude (m)
        longi (float): Launch site longitude (deg)
        lat (float): Launch site latitude (deg)
        variable_wind (bool, optional): Whether to use real wind data or not. If True, wind data will be downloaded before use. Defaults to True.
        default_wind (array, optional): Wind vector to use if 'variable_wind = False', [x_l, y_l, z_l] (m/s). Defaults to [0,0,0].
        wind_data_loc (str, optional): Directory to store wind data files in. Defaults to "data/wind/gfs".
        run_date (str, optional): Date to collect real wind data for, in the format "YYYYMMDD". Defaults to the current date.
        forcast_time (str, optional): Forcast run time, must be "00", "06", "12" or "18". Defaults to "00".
        forcast_plus_time (str, optional): Hours forcast forward from forcast time, must be three digits between 000 and 123 (?). Defaults to "000".
        fast_wind (bool, optional): ???. Defaults to False.
    Attributes:
        rail_length (float): Length of the launch rail (m)
        rail_yaw (float): Yaw angle of the launch rail (deg), using a right-hand rotation rule out the launch frame z-axis. "rail_yaw = 0" points South, "rail_yaw = 90" points East.
        rail_pitch (float): Pitch angle of the launch rail (deg). "rail_pitch = 0" points up.
        alt (float): Launch site altitude (m)
        longi (float): Launch site longitude (deg)
        lat (float): Launch site latitude (deg)
        wind (Wind): Wind object containing wind data.
    """

    def __init__(
        self,
        rail_length,
        rail_yaw,
        rail_pitch,
        alt,
        longi,
        lat,
        variable_wind=True,
        default_wind=[0, 0, 0],
        launch_datetime=(date.today()-timedelta(days=2)).strftime("%Y%m%d %H:%M"),
        cache_Wind=False
    ):
        self.rail_length = rail_length
        self.rail_yaw = rail_yaw
        self.rail_pitch = rail_pitch
        self.alt = alt + 1e-5
        self.longi = longi
        self.lat = lat
        self.wind = Wind(
            variable=variable_wind,
            default=default_wind,
            datetime=launch_datetime,
            cache=cache_Wind
        )


class Rocket:
    """Rocket object to contain rocket data and run rocketry simulations.
    Args:
        mass_model (MassModel): MassModel object containing all the data on mass and moments of inertia.
        motor (Motor): Motor object containing information on the rocket engine.
        aero (AeroData): AeroData object containg data on aerodynamic coefficients and the centre of pressure.
        launch_site (LaunchSite): LaunchSite object contaning launch site and wind information.
        h (float, optional): Integration time step (if using a fixed time step by setting "variable = False"). Defaults to 0.01.
        variable (bool, optional): If True, a variable time step is use for the integration. If "False" then the input for "h" is used as the time step. Defaults to True.
        rtol (float, optional): Relative error tolerance for integration. Defaults to 1e-7.
        atol (float, optional): Absolute error tolerance for integration. Defaults to 1e-14.
        parachute (Parachute, optional): Parachute object, containing parachute data. Defaults to Parachute(0,0,0,0,0,0).
        alt_poll_interval (int, optional): How often to check for parachute opening. Defaults to 1.
        thrust_vector (array, optional): Direction of thrust in body coordinates. Defaults to np.array([1,0,0]).
        errors (dict, optional): Multiplication factors for the gravity, pressure, density and speed of sound. Used in the statistics model. Defaults to {"gravity":1.0,"pressure":1.0,"density":1.0,"speed_of_sound":1.0}.
    Attributes:
        mass_model (MassModel): MassModel object containing all the data on mass and moments of inertia.
        motor (Motor): Motor object containing information on the rocket engine.
        aero (AeroData): AeroData object containg data on aerodynamic coefficients and the centre of pressure.
        launch_site (LaunchSite): LaunchSite object contaning launch site and wind information.
        h (float): Integration time step (if using a fixed time step by setting "variable = False").
        variable (bool): If True, a variable time step is use for the integration. If "False" then the input for "h" is used as the time step.
        rtol (float): Relative error tolerance for integration.
        atol (float): Absolute error tolerance for integration.
        parachute (Parachute): Parachute object, containing parachute data.
        alt_poll_interval (int): ???. Defaults to 1.
        thrust_vector (array): Direction of thrust in body coordinates.
        env_vars (dict): Multiplication factors for the gravity, pressure, density and speed of sound. Used in the statistics model.
        time(array): Time since engine ignition (s).
        pos_i (array): Position in inertial coordinates [x_i, y_i, z_i] (m).
        vel_i (array): Velocity in inertial coordinates [x_i, y_i, z_i] (m/s).
        w_b (array): Angular velocity in body coordiates [x_b, y_b, z_b] (rad/s).
        b2i (scipy.spatial.transform.Rotation): Body-to-inertial coordinate rotation matrix.
        i2b (scipy.spatial.transform.Rotation): Inertial-to-body coordinate rotation matrix.
        alt (float): Rocket altitude (m).
        on_rail (bool): True if the rocket is still on the rail, False if the rocket is off the rail.
        burn_out (bool): False if engine is still firing, True if the engine has finished firing.
        alt_record(float) : "Current" altitude of the rocket used to check for parahute opening.
        alt_poll_watch_interval (float) : How often to check if the parachute needs to be openes (s).
        alt_poll_watch (float): Last polled time.
    """

    def __init__(
        self,
        mass_model,
        motor,
        aero,
        launch_site,
        h=0.01,
        variable=True,
        rtol=1e-7,
        atol=1e-14,
        parachute=Parachute(0, 0, 0, 0, 0, 0),
        alt_poll_interval=1,
        thrust_vector=np.array([1, 0, 0]),
        errors={"gravity": 1.0, "pressure": 1.0, "density": 1.0, "speed_of_sound": 1.0},
    ):
        self.launch_site = launch_site
        self.motor = motor
        self.thrust_vector = np.array(thrust_vector)
        self.aero = aero
        self.mass_model = mass_model

        self.time = 0
        self.h = h

        self.variable_time = variable
        self.atol = atol
        self.rtol = rtol

        # Get the additional bit due to the angling of the rail
        rail_rotation = Rotation.from_euler(
            "yz", [self.launch_site.rail_pitch, self.launch_site.rail_yaw], degrees=True
        )

        # Initialise the rocket's orientation - store it in a scipy.spatial.transform.Rotation object
        xb_l = rail_rotation.apply([0, 0, 1])
        yb_l = rail_rotation.apply([0, 1, 0])
        zb_l = rail_rotation.apply([-1, 0, 0])

        xb_i = direction_l2i(
            xb_l, self.launch_site, self.time
        )  # xb should point up, and zl points up
        yb_i = direction_l2i(
            yb_l, self.launch_site, self.time
        )  # y for the body is aligned with y for the launch site (both point East)
        zb_i = direction_l2i(
            zb_l, self.launch_site, self.time
        )  # xl points South, zb should point North

        b2imat = np.zeros([3, 3])
        b2imat[:, 0] = xb_i
        b2imat[:, 1] = yb_i
        b2imat[:, 2] = zb_i
        self.b2i = Rotation.from_matrix(b2imat)

        self.b2i = (
            self.b2i
        )  # Body-to-Inertial Rotation - you can apply it to a vector with self.b2i.apply(vector)
        self.i2b = self.b2i.inv()  # Inertial-to-Body Rotation

        # Initialise angular positions and angular velocities
        self.pos_i = pos_l2i(
            np.array([0, 0, launch_site.alt]), launch_site, 0
        )  # Position in inertial coordinates - defining the launch site origin as at an altitude of zero
        self.vel_i = vel_l2i(
            [0, 0, 0], launch_site, 0
        )  # Velocity in intertial coordinates

        self.w_b = np.array([0, 0, 0])  # Angular velocity in body coordinates
        self.alt = launch_site.alt  # Altitude
        self.on_rail = True
        self.burn_out = False

        self.parachute_deployed = False
        self.parachute = parachute

        self.alt_record = pos_i2alt(self.pos_i, self.time)
        self.alt_poll_watch_interval = alt_poll_interval
        self.alt_poll_watch = self.alt_poll_watch_interval

        self.thrust_vector = thrust_vector
        self.env_vars = errors

    def fdot(self, time, fn):
        """Returns the rate of change of the rocket's state array, 'fn'.
        Args:
            time (float): Time since ignition (s).
            fn (array): Rocket's current state, [pos_i[0], pos_i[1], pos_i[2], vel_i[0], vel_i[1], vel_i[2], w_b[0], w_b[1], w_b[2], xb_i[0], xb_i[1], xb_i[2], yb_i[0], yb_i[1], yb_i[2], zb_i[0],zb_i[1],zb_i[2]]
        Returns:
            array: Rate of change of fdot, i.e. [vel_i[0], vel_i[1], vel_i[2], acc_i[0], acc_i[1], acc_i[2], wdot_b[0], wdot_b[1], wdot_b[2], xbdot[0], xbdot[1], xbdot[2], ybdot[0], ybdot[1], ybdot[2], zbdot[0], zbdot[1], zbdot[2]]
        """

        # CURRENT STATUS
        # --------------
        pos_i = np.array([fn[0], fn[1], fn[2]])  # Position in inertial coordinates
        vel_i = np.array([fn[3], fn[4], fn[5]])  # Velocity in inertial coordinates
        w_b = np.array([fn[6], fn[7], fn[8]])  # Angular velocity in body coordinates
        xb = np.array(
            [fn[9], fn[10], fn[11]]
        )  # Direction of the body x-x axis (in inertial coordinates)
        yb = np.array(
            [fn[12], fn[13], fn[14]]
        )  # Direction of the body y-y axis (in inertial coordinates)
        zb = np.array(
            [fn[15], fn[16], fn[17]]
        )  # Direction of the body z-z axis (in inertial coordinates)

        b2imat = np.zeros([3, 3])
        b2imat[:, 0] = xb
        b2imat[:, 1] = yb
        b2imat[:, 2] = zb
        b2i = Rotation.from_matrix(b2imat)  # Rotation from body to inertial coordinates

        w_i = b2i.apply(w_b)  # Angular velocity in inertial coordinates

        # KEEP TRACK OF FORCES AND MOMENTS
        # --------------------------------
        # A force should be added to either F_b or F_i, but not both. F_b and F_i will be added together at the end (after doing a coordinate transform). The same applies for M_b and M_i.
        F_b = np.array([0.0, 0.0, 0.0])
        F_i = np.array([0.0, 0.0, 0.0])

        M_b = np.array([0.0, 0.0, 0.0])
        M_i = np.array([0.0, 0.0, 0.0])

        # MASS AND GEOMETRY
        # -----------------
        lat, long, alt = i2lla(pos_i, time)
        cog = self.mass_model.cog(time)
        mass = self.mass_model.mass(time)
        ixx = self.mass_model.ixx(time)
        iyy = self.mass_model.iyy(time)
        izz = self.mass_model.izz(time)

        # Is this still necessary?:
        # I keep getting some weird error where if there is any wind the time steps go to ~11s long near the ground and then it goes really far under ground, presumably in less than one whole time step so the simulation can't break
        if alt < -5000:
            alt = -5000
        elif alt > 81020:
            alt = 81020

        # LOCAL ATMOSPHERIC PROPERTIES
        # ----------------------------
        speed_of_sound = (
            Atmosphere(alt).speed_of_sound[0] * self.env_vars["speed_of_sound"]
        )
        ambient_density = Atmosphere(alt).density[0] * self.env_vars["density"]
        ambient_pressure = Atmosphere(alt).pressure[0] * self.env_vars["pressure"]

        # AERODYNAMICS
        # ------------
        v_relative_wind_i = direction_l2i(
            (
                i2airspeed(pos_i, vel_i, self.launch_site, time)
                - self.launch_site.wind.get_wind(lat, long, alt)
            ),
            self.launch_site,
            time,
        )
        v_relative_wind_b = b2i.inv().apply(v_relative_wind_i)
        air_speed = np.linalg.norm(v_relative_wind_b)
        q = 0.5 * ambient_density * air_speed ** 2  # Dynamic pressure
        mach = air_speed / speed_of_sound

        if self.parachute_deployed == True and self.parachute.main_c_d != 0:
            # Parachute forces
            CD, ref_area = self.parachute.get(alt, mach)
            F_parachute_i = -0.5 * q * ref_area * CD * v_relative_wind_i / air_speed

            # Append to list of forces
            F_i = F_i + F_parachute_i

        else:
            # Aerodynamic forces and moments from the rocket body
            alpha = np.arccos(np.dot(v_relative_wind_b / air_speed, [1, 0, 0]))
            cop = self.aero.COP(mach, abs(alpha))
            r_cop_cog_b = (cop - cog) * np.array([-1, 0, 0])

            CA = self.aero.CA(mach, abs(alpha))
            CN = self.aero.CN(mach, abs(alpha))
            FA_b = (
                CA
                * q
                * self.aero.ref_area
                * np.array([-np.sign(v_relative_wind_b[0]), 0, 0])
            )
            FN_b = (
                CN
                * q
                * self.aero.ref_area
                * np.cross(
                    [1, 0, 0], np.cross([1, 0, 0], v_relative_wind_b / air_speed)
                )
            )
            F_aero_b = FA_b + FN_b
            M_aero_b = np.cross(r_cop_cog_b, F_aero_b)

            # Aerodynamic damping moment: M = C * ρ * ω^2
            M_aerodamping_b = np.array(
                [
                    -np.sign(w_b[0])
                    * ambient_density
                    * w_b[0] ** 2
                    * self.aero.roll_damping_coefficient,
                    -np.sign(w_b[1])
                    * ambient_density
                    * w_b[1] ** 2
                    * self.aero.pitch_damping_coefficient,
                    -np.sign(w_b[2])
                    * ambient_density
                    * w_b[2] ** 2
                    * self.aero.pitch_damping_coefficient,
                ]
            )

            # Add to the forces and moments
            F_b = F_b + F_aero_b
            M_b = M_b + M_aero_b + M_aerodamping_b

        # MOTOR
        # -----
        if time < self.motor.time_array[-1]:
            thrust = (
                self.motor.thrust(time)
                + (self.motor.ambient_pressure - ambient_pressure)
                * self.motor.exit_area
            )
            r_engine_cog_b = (self.motor.pos - cog) * np.array([-1, 0, 0])
            mdot = scipy.misc.derivative(
                self.mass_model.mass, time, dx=1
            )  # Propellant mass flow rate. Not sure what I should use for 'dx' here.

            F_thrust_b = (
                thrust * self.thrust_vector / np.linalg.norm(self.thrust_vector)
            )
            M_thrust_b = np.cross(r_engine_cog_b, F_thrust_b)
            M_jetdamping_b = (
                mdot
                * (self.mass_model.cog(time) - self.motor.pos) ** 2
                * np.array([0, w_b[1], w_b[2]])
            )  # Jet damping moment - page 8 of https://apps.dtic.mil/sti/pdfs/AD0642855.pdf - we will assume that the propellant COG is the same as the rocket COG.

            # Add to the forces and moments
            F_b = F_b + F_thrust_b
            M_b = M_b + M_thrust_b + M_jetdamping_b

        else:
            if self.burn_out == False:
                print("Burnout at t={:.2f} s ".format(time))
                self.burn_out = True

        # GRAVITY
        # -------
        F_gravity_i = (
            -self.env_vars["gravity"]
            * 3.986004418e14
            * mass
            * pos_i
            / np.linalg.norm(pos_i) ** 3
        )  # F = -GMm/r^2 = μm/r^2 where μ = 3.986004418e14 for Earth
        F_i = F_i + F_gravity_i  # Add to the forces

        # ACCELERATIONS
        # -------------
        # Net force and moment in inertial coordinates
        F_i_total = F_i + b2i.apply(F_b)
        M_b_total = M_b + b2i.inv().apply(M_i)

        # Linear acceleration from F = ma
        acc_i = F_i_total / mass

        # If on the rail:
        if self.on_rail == True:
            xb_i = b2i.apply([1, 0, 0])
            xb_i = xb_i / np.linalg.norm(
                xb_i
            )  # Normalise it just in case (but this step should be unnecessary)
            acc_i = (
                np.dot(acc_i, xb_i) * xb_i
            )  # Make it so we only keep the acceleration along the body's x-direction (i.e. in the forwards direction)
            wdot_b = np.array(
                [0, 0, 0]
            )  # Assume no rotational acceleration on the rail

        else:
            # Rotational acceleration, from Euler's equations
            wdot_b = np.array(
                [
                    (M_b_total[0] + (iyy - izz) * w_b[1] * w_b[2]) / ixx,
                    (M_b_total[1] + (izz - ixx) * w_b[2] * w_b[0]) / iyy,
                    (M_b_total[2] + (ixx - iyy) * w_b[0] * w_b[1]) / izz,
                ]
            )

        # Rate of change of the rocket's direction. If a vector 'r' is rotating in the inertial frame, dr/dt = w_i x r.
        xbdot = np.cross(w_i, xb)
        ybdot = np.cross(w_i, yb)
        zbdot = np.cross(w_i, zb)

        return np.array(
            [
                vel_i[0],
                vel_i[1],
                vel_i[2],
                acc_i[0],
                acc_i[1],
                acc_i[2],
                wdot_b[0],
                wdot_b[1],
                wdot_b[2],
                xbdot[0],
                xbdot[1],
                xbdot[2],
                ybdot[0],
                ybdot[1],
                ybdot[2],
                zbdot[0],
                zbdot[1],
                zbdot[2],
            ]
        )

    def run(self, max_time=1000, debug=False, to_json=False):
        """Runs the rocket trajectory simulation. Uses the SciPy DOP853 O(h^8) integrator.
        Args:
            max_time (float, optional): Maximum time to run the simulation for (s). Defaults to 1000.
            debug (bool, optional): If True, data will be printed to the console to aid with debugging. Defaults to False.
            to_json (str, optional): Directory to export a .json file to, containing the results of the simulation. If False, no .json file will be produced. Defaults to False.
        Returns:
            pandas.DataFrame: pandas DataFrame containing the fundamental trajectory results. Most information can be derived from this in post processing.
                "time" (array): List of times that all the data corresponds to (s).
                "pos_i" (array): List of position vectors in inertial coordinates [x_i, y_i, z_i] (m).
                "vel_i" (array): List of velocity vectors in inertial coordinates [x_i, y_i, z_i] (m/s).
                "b2imat" (array): List of rotation matrices for going from the body to inertial coordinate system (i.e. a record of rocket orientation).
                "w_b" (array): List of angular velocity vectors, in body coordinates [x_b, y_b, z_b] (rad/s).
                "events" (array): List of useful events.
        """
        if debug == True:
            print("Running simulation")

        xb_i = self.b2i.as_matrix()[:, 0]
        yb_i = self.b2i.as_matrix()[:, 1]
        zb_i = self.b2i.as_matrix()[:, 2]

        # Set up the integrator. fn is the rocket's "state array" - it contains everything needed to define its current state.
        fn = [
            self.pos_i[0],
            self.pos_i[1],
            self.pos_i[2],
            self.vel_i[0],
            self.vel_i[1],
            self.vel_i[2],
            self.w_b[0],
            self.w_b[1],
            self.w_b[2],
            xb_i[0],
            xb_i[1],
            xb_i[2],
            yb_i[0],
            yb_i[1],
            yb_i[2],
            zb_i[0],
            zb_i[1],
            zb_i[2],
        ]
        integrator = integrate.DOP853(
            self.fdot, 0, fn, 1000, atol=self.atol, rtol=self.rtol
        )
        record = pd.DataFrame({})  # Set up the pandas dataframe
        c = 0  # Counter used when printing debug information

        # Integration process
        while pos_i2alt(self.pos_i, self.time) >= 0 and self.time < max_time:
            if self.variable_time == False:
                integrator.h_abs = self.h

            # Check for events, e.g. rail departure or parachute deployment
            events = self.check_phase(debug=debug)
            integrator.step()
            self.pos_i = np.array([integrator.y[0], integrator.y[1], integrator.y[2]])
            self.vel_i = np.array([integrator.y[3], integrator.y[4], integrator.y[5]])
            self.w_b = np.array([integrator.y[6], integrator.y[7], integrator.y[8]])
            b2imat = np.zeros([3, 3])

            if self.parachute_deployed == False:
                b2imat[:, 0] = np.array(
                    [integrator.y[9], integrator.y[10], integrator.y[11]]
                )  # Body x-direction
                b2imat[:, 1] = np.array(
                    [integrator.y[12], integrator.y[13], integrator.y[14]]
                )  # Body y-direction
                b2imat[:, 2] = np.array(
                    [integrator.y[15], integrator.y[16], integrator.y[17]]
                )  # Body z-direction
            else:
                lat, long, alt = i2lla(self.pos_i, self.time)
                wind_inertial = vel_l2i(
                    self.launch_site.wind.get_wind(lat, long, alt),
                    self.launch_site,
                    self.time,
                )
                v_rel_wind = self.vel_i - wind_inertial
                b2imat[:, 0] = -v_rel_wind / np.linalg.norm(
                    v_rel_wind
                )  # Body x-direction
                z = self.b2i.as_matrix()[:, 2]
                b2imat[:, 1] = np.cross(
                    z, -v_rel_wind / np.linalg.norm(v_rel_wind)
                )  # Body y-direction
                b2imat[:, 2] = z  # Body z-direction

            self.b2i = Rotation.from_matrix(b2imat)
            self.i2b = self.b2i.inv()

            self.time = integrator.t
            if self.variable_time == True:
                self.h = integrator.h_previous

            # Data to add to the pandas dataframe
            new_row = {
                "time": self.time,
                "pos_i": self.pos_i.tolist(),
                "vel_i": self.vel_i.tolist(),
                "b2imat": b2imat.tolist(),
                "w_b": self.w_b.tolist(),
                "events": events,
            }

            record = record.append(new_row, ignore_index=True)

            # Debug messages
            if c % 100 == 0 and debug == True:
                print(
                    "t={:.2f} s alt={:.2f} km (h={} s). Step number {}".format(
                        self.time,
                        pos_i2alt(self.pos_i, self.time) / 1000,
                        integrator.h_abs,
                        c,
                    )
                )
            c += 1

        # Export a JSON if required
        if to_json != False:
            # Convert the DataFrame to a dict first, the in-built Python JSON library works better than panda's does I think
            dict = record.to_dict(orient="list")

            # Now use the inbuilt json module to export it
            with open(to_json, "w+") as write_file:
                json.dump(dict, write_file)

            if debug == True:
                print("Exported JSON data to '{}'".format(to_json))

        return record

    def check_phase(self, debug=False):
        """Check what phase of flight the rocket is in, e.g. on the rail, off the rail, or with the parachute open.
        Notes:
            - Since this only checks after each time step, there may be a very short period where the rocket is orientated as if it is still on the rail, when it shouldn't be.
            - For this reason, it may look like the rocket leaves the rail at an altitude greater than the rail length.
        Args:
            debug (bool, optional): If True, a message is printed when the rocket leaves the rail. Defaults to False.
        Returns:
            list: List of events that happened in this step, for the data log.
        """
        events = []

        # Rail check
        if self.on_rail == True:
            # Check how far we've travelled - remember that the 'l' coordinate system has its origin at alt=0.
            rocket_pos_l = pos_i2l(self.pos_i, self.launch_site, self.time)
            launch_site_pos_l = np.array([0.0, 0.0, self.launch_site.alt])

            flight_distance = np.linalg.norm(rocket_pos_l - launch_site_pos_l)

            # Check if we've left the rail yet
            if flight_distance >= self.launch_site.rail_length:
                self.on_rail = False
                events.append("Cleared rail")

                if debug == True:
                    alt = pos_i2alt(self.pos_i, self.time)
                    ambient_pressure = (
                        Atmosphere(alt).pressure[0] * self.env_vars["pressure"]
                    )
                    thrust = (
                        self.motor.thrust(self.time)
                        + (self.motor.ambient_pressure - ambient_pressure)
                        * self.motor.exit_area
                    )
                    weight = 9.81 * self.mass_model.mass(self.time)

                    print(
                        "Cleared rail at t={:.2f} s with alt={:.2f} m and TtW={:.2f}".format(
                            self.time, alt, thrust / weight
                        )
                    )

        # Parachute check
        if self.parachute_deployed == False:
            if self.alt_poll_watch < self.time - self.alt_poll_watch_interval:
                current_alt = pos_i2alt(self.pos_i, self.time)
                if self.alt_record > current_alt:
                    if debug == True:
                        print(
                            "Parachute deployed at {:.2f} km at {:.2f} s".format(
                                pos_i2alt(self.pos_i, self.time) / 1000, self.time
                            )
                        )
                    events.append("Parachute deployed")
                    self.parachute_deployed = True
                    self.w_b = np.array([0, 0, 0])
                else:
                    self.alt_poll_watch = self.time
                    self.alt_record = current_alt
        return events


def from_json(directory):
    """Extract trajectory data from a .json file produced by campyros.Rocket.run(), and convert it into a pandas DataFrame.
    Args:
        directory (str): .json file directory.
    Returns:
        pandas.DataFrame: pandas DataFrame containing the fundamental trajectory results. Most information can be derived from this in post processing.
                "time" (array): List of times that all the data corresponds to (s).
                "pos_i" (array): List of position vectors in inertial coordinates [x_i, y_i, z_i] (m).
                "vel_i" (array): List of velocity vectors in inertial coordinates [x_i, y_i, z_i] (m/s).
                "b2imat" (array): List of rotation matrices for going from the body to inertial coordinate system (i.e. a record of rocket orientation).
                "w_b" (array): List of angular velocity vectors, in body coordinates [x_b, y_b, z_b] (rad/s).
                "events" (array): List of useful events.
    """
    # Import the JSON as a dict first (the in-built Python JSON library works better than panda's does I think)
    with open(directory, "r") as read_file:
        dict = json.load(read_file)

    # Now convert the dict to a pandas DataFrame
    return pd.DataFrame.from_dict(dict, orient="columns")
