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

solver_file = solver_session.settings.file

cas_File = r"E:\Ansys_simulation\pyFluent\Example\mixing_elbow_settings_api\FDM-PCF.cas.h5"
dat_File = r"E:\Ansys_simulation\pyFluent\Example\mixing_elbow_settings_api\FDM-PCF.dat.h5"
solver_file.read_case(file_name=cas_File)
solver_file.read_data(file_name=dat_File)

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

# Create plane surface using normal point and centroid point
outlet_plane = PlaneSurface.create_from_point_and_normal(
        solver = solver_session,
        point=centroid_mean,
        normal=normal_unit
        )

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
#outletvel_mag = np.sqrt(u**2 + v**2 + w**2)
print(outlet_vel.shape)
print(outlet_vel[1][:])

print(outlet_position.shape)
print(outlet_position[1][:])

###############################################################################
# Post-Processing Workflow
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Draw a outlet velocity profile by plt
#

save_dir = os.path.join(cwd, "figure")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "vel_mag.png")
coords2d, axis1, axis2, origin = project_to_plane(outlet_position, normal_unit, centroid_mean)

# Easy mode
#fig, ax = plot_velocity_contour(coords2d, outlet_vel_mag)

# Hard mode
fig, ax = plot_velocity_contour(
        points=coords2d, vel=outlet_vel_mag, 
        cmap="viridis", fill_nan_method="nearest",   
        grid_res=200, smooth_sigma=0.5, 
        figsize=(4,12), levels=50)
        

fig.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig)


###############################################################################
# Post-Processing Workflow
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Graphics module
# Create a contour of velocity magnitude, show and save
#

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
# Close Fluent
# ~~~~~~~~~~~~
# Close Fluent.
#

#solver_session.exit()




