"""
    Heat Chamber heat transfer
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

###############################################################################
# Launch Fluent
# ~~~~~~~~~~~~~
# two processor, print fluent version
# Redirect all fluent mesg into a specified file

solver_session = pyfluent.launch_fluent(
    precision="single",
    processor_count=14,
    mode="solver",
)

version_tag = "v3"
mesh_tag = "coarse"
cwd = os.getcwd()
print(solver_session.get_fluent_version())

###############################################################################
# Import mesh and perform mesh check
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import mesh and perform mesh check
# save the current dir

mesh_dir = os.path.join(cwd, "mesh")
mesh_path = os.path.join(mesh_dir, f"HeatChamber_{version_tag}_{mesh_tag}.msh")
solver_session.settings.file.read_case(file_name=mesh_path)
solver_session.settings.mesh.check()

###############################################################################
# General Module
# ~~~~~~~~~~~~~~
# Launch natural convection
# unit (m/s^2)
#

solver_general = solver_session.settings.setup.general
solver_general.solver.time = "steady"
solver_general.operating_conditions.gravity.enable = True
solver_general.operating_conditions.gravity.components = [0, 0 , -9.81]
print("-"*20 + "General setting Done" + "-"*20)


###############################################################################
# Setup model for CFD analysis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Select "k-omega-sst" model
#

solver_model = solver_session.settings.setup.models

solver_model.energy.enabled = True
solver_model.viscous.model = "k-omega"
solver_model.viscous.k_omega_model = "sst"

print("-"*20 + "Models setting Done" + "-"*20)

###############################################################################
# Create material
# ~~~~~~~~~~~~~~~
# Create a material named "Air" for fluid 
# Create a material named "ABS" for solid
# ----- Unit -----
# All numerical values below are in Fluent's default SI units (kg, m, s, K).
# Density (rho): kg/m^3
# Specific Heat (Cp): J/(kg*K)
# Thermal Conductivity (k): W/(m*K)
# 

solver_materials = solver_session.settings.setup.materials
solver_materials.database.copy_by_name(type="fluid", name="air")
air = solver_materials.fluid["air"]
air.density.option = "incompressible-ideal-gas"
air.viscosity.option = "sutherland"
air.thermal_conductivity.option = "kinetic-theory"

ABS = solver_materials.solid.create("ABS")
ABS.chemical_formula = ""
ABS.density.value = 1040
ABS.specific_heat.value = 871
ABS.thermal_conductivity.value = 0.17

al = solver_materials.solid.create("al")
al.chemical_formula = ""
al.density.value = 2719
al.specific_heat.value = 871
al.thermal_conductivity.value = 202.4

brass = solver_materials.solid.create("brass")
brass.chemical_formula = ""
brass.density.value = 8500
brass.specific_heat.value = 380
brass.thermal_conductivity.value = 109
print("-"*20 + "Materials setting Done" + "-"*20)

###############################################################################
# Set up cell zone conditions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set up the cell zone conditions for the fluid zone and solid zone
# Material Assignment Key:
# - air: Fluid domain (Heat Chamber air, set up as Incompressible Ideal Gas)
# - ABS: 3D Printed Specimen
# - al: Aluminum (Used for Heat Bed)   260C, 533.15 K
# - brass: Brass (Used for Nozzle)     110C, 383.15 K
#

solver_cell_zone_conditions = solver_session.setup.cell_zone_conditions
fluid_zone = solver_cell_zone_conditions.fluid['heat_chamber_v2-heat_chamber_fluid___']
fluid_zone.general.material = "air"

solid_zone_ABS = solver_cell_zone_conditions.solid["heat_chamber_v2-heat_chamber_solid_abs"]
solid_zone_ABS.general.material = "al"
solid_zone_nozzle = solver_cell_zone_conditions.solid["heat_chamber_v2-heat_chamber_solid_nozzle"]
solid_zone_nozzle.general.material = "brass"
solid_zone_nozzle.fixed_values.enable = True
nozzle_T = 260.0 + 273.15
solid_zone_nozzle.fixed_values.variables["Temperature"].option = "value"
solid_zone_nozzle.fixed_values.variables["Temperature"].value = nozzle_T

solid_zone_heat_bed = solver_cell_zone_conditions.solid["heat_chamber_v2-heat_chamber_solid_heat_bed"]
solid_zone_heat_bed.general.material = "brass"
solid_zone_heat_bed.fixed_values.enable = True
heat_bed_T = 110.0 + 273.15
solid_zone_heat_bed.fixed_values.variables["Temperature"].option = "value"
solid_zone_heat_bed.fixed_values.variables["Temperature"].value = heat_bed_T

# Split heat_bed and ABS interface into two solids
solver_session.tui.mesh.modify_zones.slit_interior_between_diff_solids(
    'yes',
    'al',
    'brass'
)
print("-"*20 + "Cell Zones Conditions setting Done" + "-"*20)



###############################################################################
# Set up boundary conditions for CFD analysis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# "inlet"
# Force convection with velocity 3m/s and Temperature 363.15K
# "wall"
# adiabatic for wall_outer
# coupled for interface

inlet = solver_session.settings.setup.boundary_conditions.velocity_inlet["inlet"]
inlet.momentum.velocity_magnitude.value = 3
inlet.momentum.initial_gauge_pressure = 0
inlet.turbulence.turbulence_specification = "Intensity and Viscosity Ratio"
inlet.turbulence.turbulent_intensity = 0.05
inlet.turbulence.turbulent_viscosity_ratio = 10
inlet.thermal.temperature.option = 'value'
inlet.thermal.temperature.value = 90 + 273.15

outlet = solver_session.settings.setup.boundary_conditions.pressure_outlet["outlet"]
outlet.momentum.gauge_pressure = 0
outlet.turbulence.turbulence_specification = "Intensity and Viscosity Ratio"
outlet.turbulence.turbulent_intensity = 0.05
outlet.turbulence.turbulent_viscosity_ratio = 10
outlet.thermal.backflow_total_temperature.option = 'value'
outlet.thermal.backflow_total_temperature.value = 90 + 273.15

wall_outer = solver_session.settings.setup.boundary_conditions.wall["wall_outer"]
wall_outer.thermal.thermal_condition = "Heat Flux"
wall_outer.thermal.heat_flux.option = "value"
wall_outer.thermal.heat_flux.value = 0

wall_bcs = solver_session.settings.setup.boundary_conditions.wall
wall_names = wall_bcs.get_object_names()
excluded_wall = "wall_outer"
for wall_name in wall_names:
    if wall_name != excluded_wall:
        wall = wall_bcs[wall_name]
        wall.thermal.thermal_condition = "Coupled"
#        print(f"Set wall '{wall_name}' to Thermal Condition: Coupled")

#    else:
#        print(f"Skipped '{wall_name}' (External Wall - remains set to Heat Flux = 0)")

print("-"*20 + "Boudary Conditions setting Done" + "-"*20)


###############################################################################
# Check convergence criteria
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
#

residuals_options = solver_session.settings.solution.monitor.residual
residuals_options.equations["continuity"].absolute_criteria = 1e-4
residuals_options.equations["x-velocity"].absolute_criteria = 1e-4
residuals_options.equations["y-velocity"].absolute_criteria = 1e-4
residuals_options.equations["z-velocity"].absolute_criteria = 1e-4
residuals_options.equations["energy"].absolute_criteria = 1e-05

###############################################################################
# Solution module: Initialize flow field
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Initialize the flow field using hybrid initialization.
#

solver_session.settings.solution.initialization.hybrid_initialize()
print("-"*20 + "Solution.Initialization setting Done" + "-"*20)



###############################################################################
# Solution module: Set method
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

solver_solution = solver_session.settings.solution
solver_solution.methods.p_v_coupling.flow_scheme='SIMPLE'

print("-"*20 + "Solution.methods SIMPLE setting Done" + "-"*20)

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
dat_path = os.path.join(data_dir, f"HeatChamber_{version_tag}_{mesh_tag}")
solver_file.auto_save.root_name.set_state(dat_path)

###############################################################################
# Solution module: Set run Caculation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Solve for 150 iterations
#

solver_solution.run_calculation.iterate(iter_count=5000)
case_path = os.path.join(data_dir, f"HeatChamber_{version_tag}_{mesh_tag}.cas.h5")
solver_session.settings.file.write_case(file_name=case_path)
solver_solution.run_calculation.calculate()

dat_path = os.path.join(data_dir, f"HeatChamber_{version_tag}_{mesh_tag}.dat.h5")
solver_session.settings.file.write_data(file_name=dat_path)

print("-"*20 + "Solution Done" + "-"*20)
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
