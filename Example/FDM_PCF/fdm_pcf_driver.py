###############################################################################
# Perform required imports
# ~~~~~~~~~~~~~~~~~~~~~~~~
# downloading, importing, geometry file


import os
import numpy as np
import matplotlib.pyplot as plt
import ansys.fluent.core as pyFluent
from ansys.fluent.core import SurfaceDataType, SurfaceFieldDataRequest
from ansys.fluent.visualization import Contour, GraphicsWindow, PlaneSurface
from ansys.fluent.core.solver import VelocityInlet
from colorama import Fore, Style
from utils import project_to_plane, plot_velocity_contour
from other_utils import setup_logger, get_colors
from colorama import Fore, Style
from datetime import datetime


###############################################################################
# Set up logging and  colorama
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
exp_dir = os.path.join('experiments', "FDM_PCF")
os.makedirs(exp_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
log_file = os.path.join(exp_dir, f"solver_{timestamp}.log")
setup_logger(log_file)

color = get_colors()

###############################################################################
# Launch Fluent
# ~~~~~~~~~~~~~
# two processor, print fluent version
# Redirect all fluent mesg into a specified file

solver_session = pyFluent.launch_fluent(
    precision="double",
    processor_count=2,
    mode="solver",
)

cwd = os.getcwd()
print(solver_session.get_fluent_version())

###############################################################################
# Import mesh and perform mesh check
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import mesh and perform mesh check
# save the current dir

mesh_file=r"E:\Ansys_simulation\3Dprinted\FDM-PCF.msh"
save_path=os.getcwd()
solver_session.settings.file.read_case(file_name=mesh_file)
solver_session.settings.mesh.check()

###############################################################################
# General Moudle
# ~~~~~~~~~~~~~~
# Set transient time
#

solver_general = solver_session.settings.setup.general
solver_general.solver.time = "unsteady-2nd-order"
solver_general.solver.time.print_state()
solver_general.solver.time.allowed_values()


###############################################################################
# Setup model for CFD analysis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Select "sst k-omega" model
#

solver_model = solver_session.settings.setup.models.viscous
solver_model.model = "k-omega"
solver_model.k_omega_model = "sst"

###############################################################################
# Create material
# ~~~~~~~~~~~~~~~
# Create a material named "Air"
#

print(color["R"] + "--------------- Matrials -------------------------" + color["RESET"])
solver_materials = solver_session.settings.setup.materials
solver_materials.database.copy_by_name(type="fluid", name="air")
air_dict = solver_materials.fluid["air"].get_state()
air_dict["density"]["value"] = 1.225
air_dict["viscosity"]["value"] = 1.7894e-05
solver_materials.fluid["air"].set_state(air_dict)

###############################################################################
# Set up cell zone conditions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set up the cell zone conditions for the fluid zone
# i.e. Define the fluid zone material property
# Set "material" to "air"
#

# Take the default
#solver_session.setup.cell_zone_conditions.fluid[""].general.matiral = ("air")


###############################################################################
# Set up boundary conditions for CFD analysis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# for the "inlet", "outlet", "wall"
#
#

# inlet, Setting: Value
# Velocity Specification Method: Magnitude, Normal to Boundary
# Velocity Magnitude: 1.5[m/s]
# Turbulent module:
#    Specification Method: Intensity and Viscosity Ratio
#    Turbluent Intensity: 5 [%]
#    Turbulent Viscosity Rati [10]
inlet = solver_session.settings.setup.boundary_conditions.velocity_inlet["inlet"]
inlet.momentum.velocity_magnitude.value = 1.5
inlet.turbulence.turbulence_specification = "Intensity and Viscosity Ratio"
inlet.turbulence.turbulent_intensity = 0.05
inlet.turbulence.turbulent_viscosity_ratio = 10


# outlet, Setting: Value
# Turbulent module:
#    Specification Method: Intensity and Viscosity Ratio
#    Backflow Turbluent Intensity: 5 [%]
#    Backflow Turbulent Viscosity Ratio: [10]

outlet = solver_session.settings.setup.boundary_conditions.pressure_outlet["outlet"]
outlet.turbulence.turbulence_specification = "Intensity and Viscosity Ratio"
outlet.turbulence.turbulent_intensity = 0.05
outlet.turbulence.turbulent_viscosity_ratio = 10

###############################################################################
# Solution module: Set Method for CFD analysis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BUG Maybe do not need set methods Explicitly in code
# The system will take one automatically
#

#solver_methods = solver_session.settings.solution.methods()
#solver_methods.flow_scheme = "SIMPLE"

###############################################################################
# Solution module: Initialize flow field
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize the flow field using hybrid initialization.
#

print(color["R"] + "---------------Initialization moudel-------------------------" + color["RESET"])
solver_session.settings.solution.initialization.hybrid_initialize()

###############################################################################
# Solution module: Set run Caculation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# SOlve for 150 iterations

solver_solution = solver_session.settings.solution
solver_solution.run_calculation.iterate(iter_count=100)

#######################################################################################
# Save the case file
# ~~~~~~~~~~~~~~~~~~
#

solver_session.settings.file.write(file_type="case-data", file_name="FDM-PCF.cas.h5")

###############################################################################
# Field_data Module
# ~~~~~~~~~~~~~~~~~
# Take a flat, perpendicular outlet plane
#

# Create an istance of the FieldData class
# the normal_data is a area vector
field_data = solver_session.fields.field_data

face_data_request = SurfaceFieldDataRequest(
        surfaces=["outlet"],
        data_types=[SurfaceDataType.FacesNormal,
                    SurfaceDataType.FacesCentroid,
                   ]
        )
all_data = field_data.get_field_data(face_data_request)["outlet"]

normal_data = all_data.face_normals
normal_mean = normal_data.mean(axis=0)
normal_unit = normal_mean / np.linalg.norm(normal_mean)
print(normal_data.shape)

# Get centroid data
centroid_data = all_data.face_centroids
centroid_mean = centroid_data.mean(axis=0)
print(centroid_data.shape)

###############################################################################
# Get Solution Info
# ~~~~~~~~~~~~~~~~~
#

# Get Solution Variable Info
solution_variable_info = solver_session.fields.solution_variable_info
zones_info = solution_variable_info.get_zones_info() 
print("Domains:", zones_info.domains)      # e.g. ['mixture'] 
print("Zones:", zones_info.zones)          # e.g. ['inlet','wall','outlet',...] 
domain_name = "mixture"                    # change to domains in your case
zone_names = ["outlet"]                     # change to zones in your case 

vars_info = solution_variable_info.get_variables_info(zone_names=zone_names, domain_name=domain_name)
print("Available SVAR names:", vars_info.solution_variables)  # e.g. ['SV_U', 'SV_V', 'SV_P', 'SV_W']
svu_info = vars_info['SV_U'] 
svv_info = vars_info['SV_V'] 
svw_info = vars_info['SV_W'] 
svcentroid_info = vars_info['SV_CENTROID'] 
print(svu_info.name, svu_info.dimension, svu_info.field_type) # SV_U 1 <class 'numpy.float64'> 

# Get Solution Variable Data 
# The centroid data in SV is the same as field data
solution_variable_data = solver_session.fields.solution_variable_data  
sv_u = solution_variable_data.get_data(variable_name="SV_U", zone_names=zone_names, domain_name=domain_name)['outlet'] 
sv_v = solution_variable_data.get_data(variable_name="SV_V", zone_names=zone_names, domain_name=domain_name)['outlet']
sv_w = solution_variable_data.get_data(variable_name="SV_W", zone_names=zone_names, domain_name=domain_name)['outlet']
outlet_position = solution_variable_data.get_data(variable_name="SV_CENTROID", zone_names=zone_names, domain_name=domain_name)['outlet']
outlet_position = np.reshape(outlet_position, (-1, 3))

outlet_vel = np.stack((sv_u, sv_v, sv_w), axis=1)
outlet_vel_mag = np.linalg.norm(outlet_vel, axis=-1)
print(outlet_vel.shape)
print(outlet_vel[1][:])

print(outlet_position.shape)
print(outlet_position[1][:])


###############################################################################
# Result module: Configure graphics picture export
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

# Create plane surface using normal point and centroid point
# Do not know how to use
outlet_plane = PlaneSurface.create_from_point_and_normal(
        solver = solver_session,
        point=centroid_mean,
        normal=normal_unit
        )


solver_results = solver_session.settings.results
graphics = solver_results.graphics

# Define graph resolution by hand
if graphics.picture.use_window_resolution.is_active():
    graphics.picture.use_window_resolution = False
graphics.picture.x_resolution = 1920
graphics.picture.y_resolution = 1440

###############################################################################
# Post-Processing Workflow
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Graphics module
# Create a contour of velocity magnitude, show and save
#

solver_results = solver_session.results

graphics = solver_results.graphics
graphics.contour["velocity_outlet"] = {
        "field": "velocity-magnitude",
        "surfaces_list": ["outlet"],
        "node_values": True,
        }
velocity_outlet = solver_results.graphics.contour["velocity_outlet"]
velocity_outlet.range_options = {
                    "auto_range": True
                }

velocity_outlet.print_state()
velocity_outlet.display()

graphics.views.restore_view(view_name="front")
graphics.views.auto_scale()
graphics.picture.save_picture(file_name="outlet_surf_velocity_magnitude.png")


###############################################################################
# Post-Processing Workflow
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Draw a outlet velocity profile by plt
#

save_dir = os.path.join(cwd, "figure")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "vel_mag.png")
coords2d, axis1, axis2, origin = project_to_plane(outlet_position, normal_unit, centroid_mean)

# save and load
np.savetxt(os.path.join(cwd, "pts.txt"), coords2d, fmt="%.6e", delimiter=" ")
np.savetxt(os.path.join(cwd, "vel_mag.txt"), outlet_vel_mag, fmt="%.6e")
pts = np.loadtxt(os.path.join(cwd, "pts.txt"))
vel_mag = np.loadtxt(os.path.join(cwd, "vel_mag.txt"))


# Easy mode
#fig, ax = plot_velocity_contour(coords2d, outlet_vel_mag)

# Hard mode
np.multiply(coords2d, 1000, out=coords2d)
fig, ax = plot_velocity_contour(
        points=coords2d, vel=outlet_vel_mag, 
        cmap="viridis", fill_nan_method="nearest",   
        grid_res=200, smooth_sigma=0.5, 
        figsize=(8,4), levels=50)

if os.path.exists(save_path):
    os.remove(save_path)

fig.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig)

###############################################################################
# Close Fluent
# ~~~~~~~~~~~~
# Close Fluent.
#

solver_session.exit()







