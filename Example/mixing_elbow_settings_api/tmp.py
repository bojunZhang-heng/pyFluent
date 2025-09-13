###########################
# Perform required imports
# downloading, importing, geometry file

import os
import logging
import ansys.fluent.core as pyFluent
from utils import setup_logger, get_colors
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

solver_session.settings.setup.materials.database.copy_by_name(type="fluid", name="air")

###############################################################################
# Set up cell zone conditions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set up the cell zone conditions for the fluid zone
# Set "material" to "air"
#

# BUG I have not set fluid zone yet!
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
inlet.momentum.velocity_magnitude.value = 0.4
inlet.turbulence.turbulent_intensity = 0.05
inlet.turbulence.turbulent_viscosity_ratio = 10


# outlet, Setting: Value
# Turbulent module:
#    Specification Method: Intensity and Viscosity Ratio
#    Backflow Turbluent Intensity: 5 [%]
#    Backflow Turbulent Viscosity Ratio: [10]

outlet = solver_session.settings.setup.boundary_conditions.pressure_outlet["outlet"]
outlet.turbulence.turbulent_intensity = 0.05
outlet.turbulence.turbulent_viscosity_ratio = 10

###############################################################################
# Set Method for CFD analysis
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# BUG Maybe do not need set methods Explicitly in code
# The system will take one automatically
#

#solver_methods = solver_session.settings.solution.methods()
#solver_methods.flow_scheme = "SIMPLE"

###############################################################################
# Initialize flow field
# ~~~~~~~~~~~~~~~~~~~~~
# Initialize the flow field using hybrid initialization.
#

print(color["R"] + "---------------Initialization moudel-------------------------" + color["RESET"])
solver_session.settings.solution.initialization.hybrid_initialize()

###############################################################################
# Close Fluent
# ~~~~~~~~~~~~
# Close Fluent.
#

solver_session.exit()







