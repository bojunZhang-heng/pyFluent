###########################
# Perform required imports
# downloading, importing, geometry file
#

import os
import ansys.fluent.core as pyFluent
from ansys.fluent.core import SurfaceDataType, SurfaceFieldDataRequest
from ansys.fluent.visualization import Contour, GraphicsWindow
from ansys.fluent.core.solver import VelocityInlet
from utils import setup_logger, get_colors
from colorama import Fore, Style

color = get_colors()

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

#outlet_surf = solver_session.settings.setup.boundary_conditions.pressure_outlet["outlet"].surface
field_data = solver_session.fields.field_data

normal_request = SurfaceFieldDataRequest(
        surfaces=["outlet"],
        data_types=[SurfaceDataType.FacesNormal],
        )
normal_data = field_data.get_field_data(normal_request)

print(color["R"] + "--------------- Surface Data  -------------------------" + color["RESET"])
print(normal_data["outlet"].face_normals.shape)
print(normal_data["outlet"].face_normals[5])


###############################################################################
# Post-Processing Workflow
# ~~~~~~~~~~~~~~~~~~~~~~~~
# Graphics module
# Create a contour of velocity magnitude, show and save
#

solver_results = solver_session.settings.results



# filed: denotes the variable type, such as velocity
# surfaces_list: select the surface which is need to display
# node_values: True is smoother; False is cell denote, blocker
# display(): Necessary,display the picture in the graph
# range_options: both of them false means take the colorbar according to
#   the local min and max value
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

solver_session.exit()




