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

solver_session.get_fluent_version()

###############################################################################
# Import mesh and perform mesh check
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Import mesh and perform mesh check
# save the current dir

mesh_file=r"E:\Ansys_simulation\3Dprinted\FDM-PCF.msh"
save_path=os.getcwd()
solver_session.file.read_case(file_name=mesh_file)
solver_session.mesh.check()

###############################################################################
# Create material
# ~~~~~~~~~~~~~~~
# Create a material named "Air"
#

solver_session.setup.materials.database.copy_by_name(type="fluid", name="air")

###############################################################################
# Set up cell zone conditions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Set up the cell zone conditions for the fluid zone
# Set "material" to "air"
#

# BUG I have not set fluid zone yet!
#solver_session.setup.cell_zone_conditions.fluid[""].general.matiral = ("air")


###############################################################################
# Set up boundary conditions for the "inlet", "outlet", "wall"
# for CFD analysis
#

inlet = solver_session.setup.boundary_conditions.velocity_inlet["inlet"]
inlet.momentum.velocity_magnitude.value = 0.4










