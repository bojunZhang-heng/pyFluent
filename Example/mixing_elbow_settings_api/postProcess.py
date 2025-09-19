###########################
# Perform required imports
# downloading, importing, geometry file
#

import os
import ansys.fluent.core as pyFluent
from ansys.fluent.visualization import Contour, GraphicsWindow

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
# Post-Processinf Workflow
# ~~~~~~~~~~~~~~~~~~~~~~~~
#

# Create, get and set, so Fluent knows which faces to use
solver_results = solver_session.settings.results
solver_results.surfaces.zone_surface.create(name="inlet_surf")
inlet_surf = solver_results.surfaces.zone_surface["inlet_surf"]
inlet_surf.zone_name = "inlet"

# Create a contour of velocity magnitude, show and save
#contour_inlet = Contour(solver=solver_session, field="velocity-magnitude", surfaces=["inlet_surf"])
#disp1 = GraphicsWindow()
#disp1.add_graphics(contour_inlet)
#disp1.show()

graphics = solver_results.graphics
graphics.picture.save_picture(file_name="inlet_surf_velocity_magnitude.png")

###############################################################################
# Close Fluent
# ~~~~~~~~~~~~
# Close Fluent.
#

solver_session.exit()
















