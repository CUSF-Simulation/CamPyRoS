import trajectory, trajectory.post, trajectory.aero, csv
import numpy as np
from martlet4 import martlet4

trajectory_data = trajectory.from_json("output.json")

tangent_ogive = trajectory.post.TangentOgive(xprime = 73.7e-2, yprime = (19.7e-2)/2)

'''
#Using a fixed wall temperature:
analysis = trajectory.post.HeatTransfer(tangent_ogive, trajectory_data, martlet4)       
'''

#Using a variable wall temperature:
analysis = trajectory.post.HeatTransfer(tangent_ogive, trajectory_data, martlet4,
                                        fixed_wall_temperature = False,
                                        starting_temperature = None, 
                                        nosecone_mass = 1, 
                                        specific_heat_capacity = 900, 
                                        turbulent_transition_Rex = 7.5e6)       

'''                 
#If you want to run a single step
analysis.step(print_style="metric")
'''

#To run multiple steps:
#analysis.run(iterations = 300, starting_index = 0, print_style="minimal")
#analysis.to_json("martlet4_heating_variable_Tw.json")

#To import from the .JSON
analysis.from_json("martlet4_heating_variable_Tw.json")

analysis.plot_station(station_number = 9, imax = 300)
#analysis.plot_heat_transfer(automatic_rescaling=True)
#analysis.plot_fluid_properties(automatic_rescaling=True)