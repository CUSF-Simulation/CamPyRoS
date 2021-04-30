import unittest
import sys, os, json
from datetime import date, timedelta, datetime

sys.path.append(
    "/".join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))).split("/")[:-1]
    )
)
import campyros as pyro
from campyros import statistical as stats
from campyros import wind
import csv
import time
import numpy as np
import pandas as pd
from datetime import date, timedelta

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

"""

# Setup test case
"""Import motor data to use for the mass model"""
motor_csv = pd.read_csv("campyros/tests/testmotor.csv")
time_array = motor_csv["Time"]
smass_array = motor_csv["Solid Fuel Mass (kg)"]
S_DEN = motor_csv["Solid Fuel Density (kg/m^3)"][0]
S_L = motor_csv["Solid Fuel Length (m)"][0]
S_ROUT = motor_csv["Solid Fuel Outer Diameter (m)"][0]
vmass_array = motor_csv["Vapour Mass (kg)"]
vden_array = motor_csv["Vapour Density (kg/m^3)"]
lmass_array = motor_csv["Liquid Mass (kg)"]
lden_array = motor_csv["Liquid Density (kg/m^3)"]

"""Rocket parameters"""
DRY_MASS = 60  # Rocket dry mass (kg)
ROCKET_L = 6.529  # Rocket length (m)
ROCKET_R = 98.5e-3  # Rocket radius (m)
ROCKET_T = 1e-2  # Rocket wall thickness (m) - used when approximating the rocket airframe as a thin walled cylinder
POS_TANK_BOTTOM = (
    4.456  # Distance between the nose tip and the bottom of the nitrous tank (m)
)
POS_SOLIDFUEL_BOTTOM = (
    4.856 + S_L
)  # Distance between the nose tip and bottom of the solid fuel grain (m)
REF_AREA = 0.0305128422  # Reference area for aerodynamic coefficients (m^2)

"""Set up aerodynamic properties"""
# Get approximate values for the rotational damping coefficients
C_DAMP_PITCH = pyro.pitch_damping_coefficient(
    ROCKET_L, ROCKET_R, fin_number=4, area_per_fin=0.07369928
)
C_DAMP_ROLL = 0

# Import drag coefficients from RASAero II
aero_data = pyro.AeroData.from_rasaero(
    "campyros/tests/testaero.csv", REF_AREA, C_DAMP_PITCH, C_DAMP_ROLL
)
# aero_data.show_plot()   #Show plots of how the program interpreted the data, so you can visually check if it's correct

"""Set up the mass model"""
mass_model = pyro.MassModel()
mass_model.add_hollowcylinder(
    DRY_MASS, ROCKET_R, ROCKET_R - ROCKET_T, ROCKET_L, ROCKET_L / 2
)
mass_model.add_liquidtank(
    lmass_array,
    lden_array,
    time_array,
    ROCKET_R,
    POS_TANK_BOTTOM,
    vmass_array,
    vden_array,
)
mass_model.add_solidfuel(
    smass_array, time_array, S_DEN, S_ROUT, S_L, POS_SOLIDFUEL_BOTTOM
)

"""Create the other objects needed to initialise the Rocket object"""
pulsar = pyro.Motor.from_novus("campyros/tests/testmotor.csv", pos=ROCKET_L)

"""
launch_site = pyro.LaunchSite(rail_length=10, 
                                    rail_yaw=0, 
                                    rail_pitch=0, 
                                    alt=1, 
                                    longi=0.1160127, 
                                    lat=52.2079404, 
                                    variable_wind=True,
                                    forcast_plus_time="016",
                                    run_date="20201216",
                                    fast_wind=False)
"""
launch_site = pyro.LaunchSite(
    rail_length=5,
    rail_yaw=0,
    rail_pitch=0,
    alt=10,
    longi=0.1,
    lat=52.1,
    variable_wind=False,
)  # Use this version if you don't want to use the real wind (e.g. to test something else)

parachute = pyro.Parachute(
    main_s=13.9,
    drogue_s=1.13,
    main_c_d=0.78,
    drogue_c_d=0.78,
    main_alt=500,
    attach_distance=0,
)

"""Create the Rocket object"""
martlet4 = pyro.Rocket(
    mass_model,
    pulsar,
    aero_data,
    launch_site,
    h=0.05,
    variable=True,
    alt_poll_interval=1,
    parachute=parachute,
)

run = martlet4.run(debug=False)

test_output = pyro.from_json("campyros/tests/test.json")
run_time = test_output.time.max()
min_pos = min(
    [(l[0] ** 2 + l[1] ** 2 + l[2] ** 2) ** 0.5 for l in test_output.pos_i.to_list()]
)  # should be about landing site
apogee = max(
    [(l[0] ** 2 + l[1] ** 2 + l[2] ** 2) ** 0.5 for l in test_output.pos_i.to_list()]
)
max_vel = max(
    [(l[0] ** 2 + l[1] ** 2 + l[2] ** 2) ** 0.5 for l in test_output.vel_i.to_list()]
)

parachute_ind = 0
rail_ind = 0
for ind, ev in enumerate(test_output.events):
    if "Parachute deployed" in ev:
        parachute_ind = ind
    if "Cleared rail" in ev:
        rail_ind = ind

parachute_ind_run = 0
rail_ind_run = 0
for ind, ev in enumerate(run.events):
    if "Parachute deployed" in ev:
        parachute_ind_run = ind
    if "Cleared rail" in ev:
        rail_ind_run = ind


class ExampleTest(unittest.TestCase):
    def test_time(self):
        self.assertAlmostEqual(run_time // 5, run.time.max() // 5, places=0)

    def test_apogee(self):
        run_apogee = max(
            [(l[0] ** 2 + l[1] ** 2 + l[2] ** 2) ** 0.5 for l in run.pos_i.to_list()]
        )
        self.assertAlmostEqual(apogee, run_apogee, places=0)

    def test_max_vel(self):
        run_max_vel = max(
            [(l[0] ** 2 + l[1] ** 2 + l[2] ** 2) ** 0.5 for l in run.vel_i.to_list()]
        )
        self.assertAlmostEqual(max_vel, run_max_vel, places=0)

    def test_min_pos(self):
        run_min_pos = min(
            [(l[0] ** 2 + l[1] ** 2 + l[2] ** 2) ** 0.5 for l in run.pos_i.to_list()]
        )
        self.assertAlmostEqual(min_pos / 100, run_min_pos / 100, places=0)

    def test_rail(self):
        self.assertAlmostEqual(
            test_output.time[rail_ind], run.time[rail_ind_run], places=0
        )

    def test_parachute(self):
        self.assertAlmostEqual(
            test_output.time[parachute_ind] // 5,
            run.time[parachute_ind_run] // 5,
            places=0,
        )

    def test_stats(self):
        with open("campyros/tests/test_stats.json", "r") as f:
            stat_dat = json.load(f)
        stat_dat["launch_site"]["launch_datetime"] = (
            date.today() - timedelta(days=2)
        ).strftime("%Y%m%d %H:%M")
        with open("campyros/tests/test_stats.json", "w") as f:
            json.dump(stat_dat, f)
        stats_model = stats.StatisticalModel("campyros/tests/test_stats.json")
        ran = stats_model.run_model(test_mode=True, num_cpus=1)
        print(ran)
        if ran == "results/stats_testcase":
            ran = True
        else:
            ran = False
        self.assertEqual(
            True,
            ran,
            msg="Statistical model run failed, no further information automatically available",
        )

    def test_wind(self):
        wind1 = wind.Wind((date.today() - timedelta(days=1)).strftime("%Y%m%d %H:%M"))
        wind1_val = wind1.get_wind(239, 923, 292728238)
        self.assertIsInstance(wind1_val, type(np.array([0.0, 0.0])))
        self.assertEqual(3, len(wind1_val))

        wind2 = wind.Wind(variable=False)
        wind2_val = wind2.get_wind(-239, -923, -292728238)
        self.assertEqual(0, wind2_val[0])
        self.assertEqual(0, wind2_val[1])


if __name__ == "__main__":
    unittest.main()
