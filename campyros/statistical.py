import random, os, copy, json, warnings
import numpy as np
import pandas as pd
from .main import *
from .transforms import pos_i2l, vel_i2l
from .aero import *
from .mass import *
from .motor import *

from .plot import *
from datetime import datetime
from datetime import date

try:
    import ray

    RAY_AVAILABLE = True
except:
    RAY_AVAILABLE = False
    import os, sys

    sys.path.append(os.path.dirname(__file__))
    import ray_alt as ray


__copyright__ = """

    Copyright 2021 Jago Strong-Wright

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

"""


def variable_name(**variables):
    return [x for x in variables][0]


def abs_stdev(value, percentage):
    return value * percentage


class StatisticalModel:
    """Stochastic model for the rocket flight

    Notes
    -----
    Every variable specified has a value and a standard deviation in a list (i.e. [mean,st_dev])
    Currenrly only supports aero import from rasaero file.
    Wind will currently not vary

    Parameters
    ----------
    run_file : string
        json file containing config, see stats_settings.json

    Attributes
    ----------
    launch_site_vars : dict
        Dictionary of variables for launch site object. Must contain: rail_length, rail_yaw, rail_pitch, alt, longi, lat
    mass_model_vars : dict
        Dictionary of variables for mass model object. Must contain: dry_mass, prop_mass, time_data, length, radius
    aero_file : string
        Location of RASAero data file
    aero_error : dict
        Standard deviaiton for the aero coefficients in format COP, CN and CA
    motor_base : MotorObject
        Unpeterbed motor object
    h : float, optional
        Default timestep /s, defaults to 0.05 - this doesn't really do anything
    variable_time : bool, optional
        Vary timesteps?, defaults to True
    alt_poll_interval : , optional
        Parachute altitude polling interval /s defaults to 1
    run_date : string, optional
        Date for forcast data in format YYYYMMDD, defaults to current date
    forcast_time : string, optional
        Forcast run time, must be 00,06,12 or 18, defaults to 00
    forcast_plus_time : string, optional
        Hours forcast forward from forcast time, must be three digits between 000 and 123 (?), defaults to 000
    thrust_error : float
        Standard deviation of thrust magnitude error /%
    thrust_alignment : float
        Standard deviation of thrust alignment vector error
    parachute_vars : dict
        Dictionary of variables for parachute object. Must contain: main_s, main_c_d, drogue_s, drogue_c_d, main_alt, attatch_distance
    env_vars : dictionary
        Multiplied factor for the gravity, pressure, density and speed of sound used in the model,
        defaults to {"gravity":1.0,"pressure":1.0,"density":1.0,"speed_of_sound":1.0}
    type_name : list
        The different types of errors to itterate over later
    wind_base : wind object
        Unpeterbed wind model
    """

    def __init__(self, run_file):
        with open(run_file, "r") as f:
            data = json.load(f)

        if data["name"] == "":
            self.name = "stat_model_%s" % datetime.now().strftime("%Y%m%d")
        else:
            self.name = data["name"]

        self.itterations = data["itterations"]

        self.launch_site_vars = data["launch_site"]
        self.aero_file = data["aero_file"]
        self.aero_vars = data["aero"]
        self.parachute_vars = data["parachute"]
        self.enviromental = data["enviromental"]
        self.thrust_error = data["thrust_error"]
        self.mass_vars = data["mass"]
        self.motor_data = load_motor(data["motor_file"])
        self.motor_base = Motor.from_novus(data["motor_file"], pos=data["motor_pos"])

    @ray.remote
    def run_itteration(self, id, save_loc, debug=False):
        """Runs an instance of the rocket with random errors

        Note
        ----
        Assumes gaussian errors for all variables, wind currenly not varied.

        Parameters
        ----------
        id : int
            Run number, used to name runs output
        save_loc : string
            Folder to store results
        """
        motor = copy.copy(self.motor_base)
        thrust_e = np.random.normal(1, self.thrust_error["magnitude"])
        motor.thrust_array = [t * thrust_e for t in motor.thrust_array]

        dry_mass = self.mass_vars["dry_mass"][0] * np.random.normal(
            1, self.mass_vars["dry_mass"][1]
        )
        rocket_length = self.mass_vars["rocket_length"][0] * np.random.normal(
            1, self.mass_vars["rocket_length"][1]
        )
        rocket_radius = self.mass_vars["rocket_radius"][0] * np.random.normal(
            1, self.mass_vars["rocket_radius"][1]
        )
        rocket_wall_thickness = self.mass_vars["rocket_wall_thickness"][
            0
        ] * np.random.normal(1, self.mass_vars["rocket_wall_thickness"][1])
        pos_tank_bottom = self.mass_vars["pos_tank_bottom"][0] * np.random.normal(
            1, self.mass_vars["pos_tank_bottom"][1]
        )
        s_l = (
            self.motor_data["s_l"] * rocket_length / self.mass_vars["rocket_length"][0]
        )  # Solid fuel length
        pos_solidfuel_bottom = (
            self.mass_vars["pos_solidfuel_bottom_base"][0]
            * np.random.normal(1, self.mass_vars["pos_solidfuel_bottom_base"][1])
            + s_l
        )  # m - Distance between the nose tip and bottom of the solid fuel grain
        ref_area = self.aero_vars["ref_area"][0] * np.random.normal(
            1, self.aero_vars["ref_area"][1]
        )

        ###Aero
        c_damp_pitch = pitch_damping_coefficient(
            rocket_length,
            rocket_radius,
            fin_number=self.aero_vars["fins"],
            area_per_fin=self.aero_vars["area_per_fin"][0]
            * np.random.normal(1, self.aero_vars["area_per_fin"][1]),
        )
        c_damp_roll = 0
        aero_data = AeroData.from_rasaero(
            self.aero_file,
            ref_area,
            c_damp_pitch,
            c_damp_roll,
            error={
                "CN": np.random.normal(1, self.aero_vars["CN"]),
                "CA": np.random.normal(1, self.aero_vars["CA"]),
                "COP": np.random.normal(1, self.aero_vars["COP"]),
            },
        )

        ###Mass
        mass_model = MassModel()
        mass_model.add_hollowcylinder(
            dry_mass,
            rocket_radius,
            rocket_radius - rocket_wall_thickness,
            rocket_length,
            rocket_length / 2,
        )
        mass_model.add_liquidtank(
            self.motor_data["lmass"] * np.random.normal(1, self.mass_vars["lmass"]),
            self.motor_data["lden"] * np.random.normal(1, self.mass_vars["lden"]),
            self.motor_data["time"],
            rocket_radius,
            pos_tank_bottom,
            self.motor_data["vmass"] * np.random.normal(1, self.mass_vars["vmass"]),
            self.motor_data["vden"] * np.random.normal(1, self.mass_vars["vden"]),
        )
        mass_model.add_solidfuel(
            self.motor_data["smass"] * np.random.normal(1, self.mass_vars["smass"]),
            self.motor_data["time"],
            self.motor_data["sden"] * np.random.normal(1, self.mass_vars["sden"]),
            self.motor_data["s_rout"]
            * rocket_radius
            / self.mass_vars["rocket_radius"][0],
            s_l,
            pos_solidfuel_bottom,
        )

        ###Launchsite
        if (
            self.launch_site_vars["rail_yaw"][0] == 0
            and self.launch_site_vars["rail_pitch"][0] == 0
        ):
            rail_yaw = 2 * np.pi * np.random.rand()
            rail_pitch = np.random.normal(1, self.launch_site_vars["rail_pitch"][1])
        else:
            rail_yaw = self.launch_site_vars["rail_yaw"][0] * np.random.normal(
                1, self.launch_site_vars["rail_yaw"][1]
            )
            rail_pitch = self.launch_site_vars["rail_pitch"][0] * np.random.normal(
                1, self.launch_site_vars["rail_pitch"][1]
            )
        launch_site = LaunchSite(
            rail_length=self.launch_site_vars["rail_length"][0]
            * np.random.normal(1, self.launch_site_vars["rail_length"][1]),
            rail_yaw=rail_yaw,
            rail_pitch=rail_pitch,
            alt=abs(
                self.launch_site_vars["alt"][0]
                * np.random.normal(1, self.launch_site_vars["alt"][1])
            ),  # Not sure this is the correct way to make it >0 and still normally distrobuted, probably clusters just above 0
            longi=self.launch_site_vars["long"][0]
            * np.random.normal(1, self.launch_site_vars["long"][1]),
            lat=self.launch_site_vars["lat"][0]
            * np.random.normal(1, self.launch_site_vars["lat"][1]),
            variable_wind=bool(self.launch_site_vars["variable_wind"]),
            launch_datetime=self.launch_site_vars["launch_datetime"],
            cache_Wind=True
        )

        ###Parachute
        if np.random.binomial(1, self.parachute_vars["failure_rate"]) == 0:
            parachute = Parachute(
                main_s=self.parachute_vars["main_s"][0]
                * np.random.normal(1, self.parachute_vars["main_s"][1]),
                main_c_d=self.parachute_vars["main_c_d"][0]
                * np.random.normal(1, self.parachute_vars["main_c_d"][1]),
                drogue_s=self.parachute_vars["drogue_s"][0]
                * np.random.normal(1, self.parachute_vars["drogue_s"][1]),
                drogue_c_d=self.parachute_vars["drogue_c_d"][0]
                * np.random.normal(1, self.parachute_vars["drogue_c_d"][1]),
                main_alt=self.parachute_vars["main_alt"][0]
                * np.random.normal(1, self.parachute_vars["main_alt"][1]),
                attach_distance=self.parachute_vars["attatch_distance"][0]
                * np.random.normal(1, self.parachute_vars["attatch_distance"][1]),
            )
        else:
            parachute = Parachute(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        env_errors = {k: np.random.normal(1, v) for k, v in self.enviromental.items()}

        thrust_alignment = np.array(
            [
                np.random.normal(1, self.thrust_error["alignment"]),
                np.random.normal(0, self.thrust_error["alignment"]),
                np.random.normal(0, self.thrust_error["alignment"]),
            ]
        )
        thrust_alignment = thrust_alignment / np.linalg.norm(thrust_alignment)

        rocket = Rocket(
            mass_model,
            motor,
            aero_data,
            launch_site,
            h=0.05,
            variable=True,
            alt_poll_interval=1,
            parachute=parachute,
            errors=env_errors,
            thrust_vector=thrust_alignment,
        )

        run_output = rocket.run(debug=debug)
        run_save = pd.DataFrame()
        run_save["time"] = run_output["time"]
        x, y, z = [], [], []
        v_x, v_y, v_z = [], [], []

        for index, pos in enumerate(run_output["pos_i"]):
            pos_l = pos_i2l(pos, rocket.launch_site, run_output["time"][index])
            x.append(pos_l[0])
            y.append(pos_l[1])
            z.append(pos_l[2])

            vel_l = vel_i2l(
                run_output["vel_i"][index],
                rocket.launch_site,
                run_output["time"][index],
            )
            v_x.append(vel_l[0])
            v_y.append(vel_l[1])
            v_z.append(vel_l[2])

        run_save["x"] = x
        run_save["y"] = y
        run_save["z"] = z
        run_save["v_x"] = v_x
        run_save["v_y"] = v_y
        run_save[
            "v_z"
        ] = v_z  # These were'nt saving properly as vectors but really should

        with open("%s/%s.csv" % (save_loc, id), "w+") as f:
            run_save.to_csv(path_or_buf=f)

        self.wind_base = copy.copy(
            launch_site.wind
        )  # incase more data has been downloaded

    def run_model(self, test_mode=False, debug=False, num_cpus=False):
        """Runs the stochastic model

        Parameters
        ----------
        itters : int
            Number of times to run the rocket
        save_loc : string, optional
            Folder to store results, defaults to None (is generated based on date if None)
        Returns
        -------
        string
            save location, if not specified is generated so needs to be returned to be known
        """
        save_loc = "results/%s" % self.name
        if not os.path.exists(save_loc):
            os.makedirs(save_loc)

        if RAY_AVAILABLE == True:
            if num_cpus != False:
                ray.init(num_cpus=num_cpus)
            else:
                ray.init(local_mode=test_mode)

            runs = []
            for run in range(1, self.itterations + 1):
                runs.append(
                    self.run_itteration.remote(self, run, save_loc, debug=debug)
                )
            ray.wait(runs)
        else:
            for run in range(1, self.itterations + 1):
                self.run_itteration(self, run, save_loc)
        return save_loc


def analyse(results_path, itterations, full_results=True, velocity=False):
    """Loads stats model results to put them in a more useful form for use, see stats_analysis_example notebook for example use

    Parameters
    ----------
    results_path : string
        Folder containing results
    itterations : int
        Number of runs used for the model
    full_results : bool, optional
        Return the full x,y,z,t for every run of the model
    velocity : bool, optional
        Return velocity analys, defaults to False - currently not implimented

    Returns
    -------
    if full_results=True
        numpy array
            landing position mean
        numpy array
            landing position covariant matrix
        numpy array
            apogee position mean
        numpy array
            apogee position covariant matrix
        pandas dataframe
            positions of all apogees (columns x,y,z, row for each run)
        pandas dataframe
            positions of all landings (columns x,y,z, row for each run)
        pandas dataframe
            x position throughout flight (row for each run)
        pandas dataframe
            y position throughout flight (row for each run)
        pandas dataframe
            z position throughout flight (row for each run)
        andas dataframe
            time throughout flight (row for each run)
    else
        numpy array
            landing position mean
        numpy array
            landing position covariant matrix
        numpy array
            apogee position mean
        numpy array
            apogee position covariant matrix

    """
    x, y, z, t = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    itts = range(1, itterations + 1)

    for itt in itts:
        tmp = pd.read_csv("%s/%s.csv" % (results_path, itt))

        x = pd.concat([x, pd.DataFrame({itt: tmp["x"]})], ignore_index=True, axis=1)
        y = pd.concat([y, pd.DataFrame({itt: tmp["y"]})], ignore_index=True, axis=1)
        z = pd.concat([z, pd.DataFrame({itt: tmp["z"]})], ignore_index=True, axis=1)
        t = pd.concat([t, pd.DataFrame({itt: tmp["time"]})], ignore_index=True, axis=1)

    apogee = pd.DataFrame()
    apogee["index"] = z.idxmax()
    # I know this isn't the proper pandas way todo this but I can't see how todo it right
    apogee["x"] = [
        x[itt][apogee["index"].tolist()[itt]] for itt in range(0, itterations)
    ]
    apogee["y"] = [
        y[itt][apogee["index"].tolist()[itt]] for itt in range(0, itterations)
    ]
    apogee["alt"] = z.max()
    apogee = apogee.drop("index", axis=1)

    landing = pd.DataFrame()
    landing["x"] = [x[itt].dropna().iloc[-1] for itt in range(0, itterations)]
    landing["y"] = [y[itt].dropna().iloc[-1] for itt in range(0, itterations)]

    landing_mu = np.array([landing["x"].mean(), landing["y"].mean()])
    landing_cov = landing.cov()

    apogee_mu = np.array([apogee["x"].mean(), apogee["y"].mean(), apogee["alt"].mean()])
    apogee_cov = apogee.cov()

    if full_results == True:
        return (
            landing_mu,
            landing_cov,
            apogee_mu,
            apogee_cov,
            apogee,
            landing,
            x,
            y,
            z,
            t,
        )
    else:
        return landing_mu, landing_cov, apogee_mu, apogee_cov
