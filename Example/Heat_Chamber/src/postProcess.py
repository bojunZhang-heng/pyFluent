###########################
# Perform required imports
# downloading, importing, geometry file
#

import os
import numpy as np
import matplotlib.pyplot as plt
import ansys.fluent.core as pyFluent
from ansys.fluent.core import SurfaceDataType, SurfaceFieldDataRequest
from ansys.fluent.visualization import Contour, GraphicsWindow, PlaneSurface
from ansys.fluent.core.solver import VelocityInlet
from colorama import Fore, Style
from utils import project_to_plane, plot_velocity_contour


###############################################################################
# Launch Fluent
# ~~~~~~~~~~~~~
# two processor, print fluent version
# Redirect all fluent mesg into a specified file

solver_session = pyFluent.launch_fluent(
    precision="double",
    processor_count=4,
    mode="solver",
)

print(solver_session.get_fluent_version())
cwd = os.getcwd()

###############################################################################
# File module
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import dat.h5 and cas.h5 file
# save the current dir
#

version_tag = "v5"
mesh_tag = "fine4"
solver_file = solver_session.settings.file

data_dir = os.path.join(cwd, "data")
os.makedirs(data_dir, exist_ok=True)
case_path = os.path.join(data_dir, f"FDM-PCF_{version_tag}_{mesh_tag}.cas.h5")
data_path = os.path.join(data_dir, f"FDM-PCF_{version_tag}_{mesh_tag}.dat.h5")
solver_file.read_case(file_name=case_path)
solver_file.read_data(file_name=data_path)


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
    figure_dir, f"outlet_surf_velocity_magnitude_{version_tag}.png"
)
graphics.picture.save_picture(file_name=figure_path)


###############################################################################
# Post-Processing Workflow
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Draw a outlet velocity profile by plt
#

figure_dir = os.path.join(cwd, "tmp")
os.makedirs(figure_dir, exist_ok=True)
figure_path = os.path.join(figure_dir, f"vel_mag_{version_tag}.png")
coords2d, axis1, axis2, origin = project_to_plane(
    outlet_position, normal_unit, centroid_mean
)

# save and load
np.savetxt(
    os.path.join(data_dir, f"pts_{version_tag}.txt"),
    coords2d,
    fmt="%.6e",
    delimiter=" ",
)
np.savetxt(
    os.path.join(data_dir, f"vel_mag_{version_tag}.txt"), outlet_vel_mag, fmt="%.6e"
)

pts = np.loadtxt(os.path.join(data_dir, f"pts_{version_tag}.txt"))
vel_mag = np.loadtxt(os.path.join(data_dir, f"vel_mag_{version_tag}.txt"))


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
    cbar_fontsize=12,
    cbar_tick_fontsize=11,
)

if os.path.exists(figure_path):
    os.remove(figure_path)

fig.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

###############################################################################
# Reduce the backflow
# ~~~~~~~~~~~~~~~~~~~
# Compute the standard deviation of the velocity at the outlet
#

C_v = np.std(outlet_vel_mag) / np.mean(outlet_vel_mag) * 100
print(f"Degree of velocity non-uniformity C_v: {C_v}%")
###############################################################################
# Close Fluent
# ~~~~~~~~~~~~
# Close Fluent.
#

solver_session.exit()
