import os
import platform

import ansys.fluent.core as pyfluent
from ansys.fluent.core import examples
from ansys.fluent.visualization import Contour, GraphicsWindow

#######################################################################################
# Launch Fluent session with meshing mode
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

meshing_session = pyfluent.launch_fluent(
    mode="meshing", 
    precision="double", 
    processor_count=2,
    cleanup_on_exit=True
)
print(meshing_session.get_fluent_version())

#######################################################################################
# Initialize the Meshing Workflow
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Watertight Geometry
# Import geometry
#

geometry_filename = r"E:\Ansys_simulation\3Dprinted\FDM-PartCooling_files\dp0\FFF\DM\FFF.scdocx"

workflow = meshing_session.workflow
workflow.InitializeWorkflow(WorkflowType="Watertight Geometry")
workflow.TaskObject["Import Geometry"].Arguments = dict(FileName=geometry_filename)
workflow.TaskObject["Import Geometry"].Execute()

#######################################################################################
# Add Local Sizing
# ~~~~~~~~~~~~~~~~~~~~~
# Face sizing and BOI

# Face sizing
add_local_sizing = workflow.TaskObject["Add Local Sizing"]
add_local_sizing.Arguments.set_state(
    {
        "AddChild": "yes",
        "BOIControlName": "facesize_inlet",
        "BOIFaceLabelList": ["inlet"],
        "BOIGrowthRate": 1.15,
        "BOISize": 6,
    }
)
add_local_sizing.AddChildAndUpdate()

add_local_sizing.Arguments.set_state(
    {
        "AddChild": "yes",
        "BOIControlName": "facesize_outlet",
        "BOIFaceLabelList": ["outlet"],
        "BOIGrowthRate": 1.15,
        "BOISize": 4,
    }
)
add_local_sizing.AddChildAndUpdate()

add_local_sizing.Arguments.set_state(
    {
        "AddChild": "yes",
        "BOIControlName": "facesize_wall",
        "BOIFaceLabelList": ["wall"],
        "BOIGrowthRate": 1.15,
        "BOISize": 8,
    }
)
add_local_sizing.AddChildAndUpdate()

# BOI
add_local_sizing.Arguments.set_state(
    {
        "AddChild": "yes",
        "BOIControlName": "big_one",
        "BOIExecution": "Body Of Influence",
        "BOIFaceLabelList": ["big_one"],
        "BOISize": 10,
    }
)

add_local_sizing.Arguments.set_state(
    {
        "AddChild": "yes",
        "BOIControlName": "small_one",
        "BOIExecution": "Body Of Influence",
        "BOIFaceLabelList": ["small_one"],
        "BOISize": 5,
    }
)
add_local_sizing.AddChildAndUpdate()
add_local_sizing.Execute()




