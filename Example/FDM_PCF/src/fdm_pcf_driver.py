"""
    FDM_PCF wind  tunnel design
    by BojunZhang
"""
###############################################################################
# Perform required imports
# ~~~~~~~~~~~~~~~~~~~~~~~~
# downloading, importing, geometry file
import os
from datetime import datetime

import ansys.fluent.core as pyfluent
import matplotlib.pyplot as plt
import numpy as np
from ansys.fluent.core import SurfaceDataType, SurfaceFieldDataRequest
from ansys.fluent.core.solver import VelocityInlet
from ansys.fluent.visualization import Contour, GraphicsWindow, PlaneSurface
from colorama import Fore, Style
from utils import get_colors, plot_velocity_contour, project_to_plane

color = get_colors()

###############################################################################
# Launch Fluent
# ~~~~~~~~~~~~~
# two processor, print fluent version
# Redirect all fluent mesg into a specified file

solver_session = pyfluent.launch_fluent(
    precision="single",
    processor_count=4,
    mode="solver",
)

version_tag = "v5"
mesh_tag = "fine4"
cwd = os.getcwd()
print(solver_session.get_fluent_version())

###############################################################################
# Import mesh and perform mesh check
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import mesh and perform mesh check
# save the current dir

mesh_dir = os.path.join(cwd, "mesh")
mesh_path = os.path.join(mesh_dir, f"FDM_PCF_{version_tag}_{mesh_tag}.msh")
solver_session.settings.file.read_case(file_name=mesh_path)
solver_session.settings.mesh.check()

###############################################################################
# General Module
# ~~~~~~~~~~~~~~
# Set transient time
#

solver_general = solver_session.settings.setup.general
solver_general.solver.time = "steady"


###############################################################################
# Setup model for CFD analysis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Select "k-omega-sst" model
#

solver_model = solver_session.settings.setup.models

solver_model.energy.enabled = True
solver_model.viscous.model = "k-omega"
solver_model.viscous.k_omega_model = "sst"

###############################################################################
# Create material
# ~~~~~~~~~~~~~~~
# Create a material named "Air"
#

print(
    color["R"] + "--------------- Materials -------------------------" + color["RESET"]
)
solver_materials = solver_session.settings.setup.materials
solver_materials.database.copy_by_name(type="fluid", name="air")
air_dict = solver_materials.fluid["air"].get_state()
air_dict["density"]["value"] = 1.225
air_dict["viscosity"]["value"] = 1.7894e-05
solver_materials.fluid["air"].set_state(air_dict)

###############################################################################
# Set up boundary conditions for CFD analysis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# for the "inlet", "outlet", "wall
#


inlet = solver_session.settings.setup.boundary_conditions.velocity_inlet["inlet"]
inlet.momentum.velocity_magnitude.value = 1.5
inlet.momentum.initial_gauge_pressure = 0
inlet.turbulence.turbulence_specification = "Intensity and Viscosity Ratio"
inlet.turbulence.turbulent_intensity = 0.05
inlet.turbulence.turbulent_viscosity_ratio = 10

outlet = solver_session.settings.setup.boundary_conditions.pressure_outlet["outlet"]
outlet.momentum.gauge_pressure = 0
outlet.turbulence.turbulence_specification = "Intensity and Viscosity Ratio"
outlet.turbulence.turbulent_intensity = 0.05
outlet.turbulence.turbulent_viscosity_ratio = 10

###############################################################################
# Check convergence criteria
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#

residuals_options = solver_session.settings.solution.monitor.residual
residuals_options.equations["continuity"].absolute_criteria = 0.0001
residuals_options.equations["continuity"].monitor = True  # Enable continuity residuals
residuals_options.equations["x-velocity"].absolute_criteria = 0.0001
residuals_options.equations["y-velocity"].absolute_criteria = 0.0001
residuals_options.equations["z-velocity"].absolute_criteria = 0.0001

###############################################################################
# Solution module: Initialize flow field
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize the flow field using hybrid initialization.
#

print(
    color["R"]
    + "---------------Initialization module-------------------------"
    + color["RESET"]
)
solver_session.settings.solution.initialization.hybrid_initialize()


###############################################################################
# Solution module: Set method
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

solver_solution = solver_session.settings.solution
solver_solution.methods.p_v_coupling.flow_scheme='SIMPLE'

#######################################################################################
# File moudle 
# ~~~~~~~~~~~
# Auto save
#

solver_file = solver_session.settings.file
solver_file.auto_save.data_frequency.set_state(100)
solver_file.auto_save.case_frequency.set_state('if-case-is-modified')
solver_file.auto_save.retain_most_recent_files.set_state(True)
solver_file.auto_save.max_files.set_state(1)

data_dir = os.path.join(cwd, "data")
os.makedirs(data_dir, exist_ok=True)
dat_path = os.path.join(data_dir, f"FDM-PCF_{version_tag}_{mesh_tag}")
solver_file.auto_save.root_name.set_state(dat_path)

###############################################################################
# Solution module: Set run Caculation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Solve for 150 iterations
#

solver_solution.run_calculation.iterate(iter_count=5000)
case_path = os.path.join(data_dir, f"FDM-PCF_{version_tag}_{mesh_tag}.cas.h5")
solver_session.settings.file.write_case(file_name=case_path)
solver_solution.run_calculation.calculate()

dat_path = os.path.join(data_dir, f"FDM-PCF_{version_tag}_{mesh_tag}.dat.h5")
solver_session.settings.file.write_data(file_name=dat_path)

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
    data_types=[
        SurfaceDataType.FacesNormal,
        SurfaceDataType.FacesCentroid,
        SurfaceDataType.Vertices,
    ],
)
all_data = field_data.get_field_data(face_data_request)["outlet"]

# Get normal data
normal_data = all_data.face_normals
normal_mean = normal_data.mean(axis=0)
normal_unit = normal_mean / np.linalg.norm(normal_mean)
print(normal_data.shape)

# Get centroid data
centroid_data = all_data.face_centroids
centroid_mean = centroid_data.mean(axis=0)
print(centroid_data.shape)

# Get vertex data
vertex_data = all_data.vertices
print(vertex_data.shape)

###############################################################################
# Get Solution Info
# ~~~~~~~~~~~~~~~~~
#

# Get Solution Variable Info
solution_variable_info = solver_session.fields.solution_variable_info
zones_info = solution_variable_info.get_zones_info()
print("Domains:", zones_info.domains)  # e.g. ['mixture']
print("Zones:", zones_info.zones)  # e.g. ['inlet','wall','outlet',...]
domain_name = "mixture"  # change to domains in your case
zone_names = ["outlet"]  # change to zones in your case


# Outlet solution
zone_names = ["outlet"]
solution_variable_data = solver_session.fields.solution_variable_data
sv_u = solution_variable_data.get_data(
    variable_name="SV_U", zone_names=zone_names, domain_name=domain_name
)["outlet"]
sv_v = solution_variable_data.get_data(
    variable_name="SV_V", zone_names=zone_names, domain_name=domain_name
)["outlet"]
sv_w = solution_variable_data.get_data(
    variable_name="SV_W", zone_names=zone_names, domain_name=domain_name
)["outlet"]
outlet_vel = np.stack((sv_u, sv_v, sv_w), axis=1)
outlet_vel_mag = np.linalg.norm(outlet_vel, axis=-1)
outlet_vel_m = outlet_vel_mag.mean()

zone_names = ["outlet"]
outlet_position = solution_variable_data.get_data(
    variable_name="SV_CENTROID", zone_names=zone_names, domain_name=domain_name
)["outlet"]
outlet_position = np.reshape(outlet_position, (-1, 3))
print("outlet_position:")
print(outlet_position.shape)
########################################################################l#######
# Result module: Configure graphics picture export
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

# Create plane surface using normal point and centroid point
# Do not know how to use
outlet_plane = PlaneSurface.create_from_point_and_normal(
    solver=solver_session, point=centroid_mean, normal=normal_unit
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
velocity_outlet.range_options = {"auto_range": True}

velocity_outlet.print_state()
velocity_outlet.display()

graphics.views.restore_view(view_name="front")
graphics.views.auto_scale()
figure_dir = os.path.join(cwd, "figure")
figure_path = os.path.join(
    figure_dir, f"outlet_surf_velocity_magnitude_{version_tag}_{mesh_tag}.png"
)
graphics.picture.save_picture(file_name=figure_path)

###############################################################################
# Reduce the backflow
# ~~~~~~~~~~~~~~~~~~~
#

# Compute the standard deviation of the velocity at the outlet
sigma_v = np.std(outlet_vel_mag)
C_v = np.std(outlet_vel_mag) / np.mean(outlet_vel_mag) * 100
print(f"Standard deviation sigma_v: {sigma_v}%")
print(f"Degree of velocity non-uniformity C_v: {C_v}%")
print(f"mean value: {np.mean(outlet_vel_mag)}")


result_path = os.path.join(data_dir, f"FDM_PCF_{version_tag}_{mesh_tag}.txt")

with open(result_path, "w", encoding="utf-8") as f:
    f.write(f"Standard deviation sigma_v: {sigma_v}%\n")
    f.write(f"Degree of velocity non‑uniformity C_v: {C_v}%\n")
    f.write(f"Mean value: {np.mean(outlet_vel_mag)}\n")


###############################################################################
# Post-Processing Workflow
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Draw a outlet velocity profile by plt
#

figure_dir = os.path.join(cwd, "figure")
os.makedirs(figure_dir, exist_ok=True)
figure_path = os.path.join(figure_dir, f"vel_mag_{version_tag}_{mesh_tag}.png")
coords2d, axis1, axis2, origin = project_to_plane(
    outlet_position, normal_unit, centroid_mean
)

# save and load
np.savetxt(
    os.path.join(data_dir, f"pts_{version_tag}_{mesh_tag}.txt"),
    coords2d,
    fmt="%.6e",
    delimiter=" ",
)
np.savetxt(
    os.path.join(data_dir, f"vel_mag_{version_tag}_{mesh_tag}.txt"), outlet_vel_mag, fmt="%.6e"
)

pts = np.loadtxt(os.path.join(data_dir, f"pts_{version_tag}_{mesh_tag}.txt"))
vel_mag = np.loadtxt(os.path.join(data_dir, f"vel_mag_{version_tag}_{mesh_tag}.txt"))


# Easy mode
# fig, ax = plot_velocity_contour(coords2d, outlet_vel_mag)

# Hard mode
np.multiply(coords2d, 1000, out=coords2d)
fig, ax = plot_velocity_contour(
    points=coords2d,
    vel=outlet_vel_mag,
    cmap="viridis",
    fill_nan_method="nearest",
    grid_res=200,
    smooth_sigma=0.5,
    figsize=(8, 4),
    levels=50,
)

if os.path.exists(figure_path):
    os.remove(figure_path)

fig.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

###############################################################################
# Close Fluent
# ~~~~~~~~~~~~
# Close Fluent.
#

solver_session.exit()
