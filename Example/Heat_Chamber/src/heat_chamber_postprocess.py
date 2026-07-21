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


