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
version_tag = "v4"
cwd = os.getcwd()
print(meshing_session.get_fluent_version())

#######################################################################################
# Initialize the Meshing Workflow
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Watertight Geometry
# Import geometry
#

geometry_dir = os.path.join(cwd, "geometry")
geometry_path = os.path.join(geometry_dir, f"FDM_PCF_{version_tag}.scdocx")

workflow = meshing_session.workflow
workflow.InitializeWorkflow(WorkflowType="Watertight Geometry")
workflow.TaskObject["Import Geometry"].Arguments = dict(FileName=geometry_path)
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
        "BOISize": 0.1,
    }
)
add_local_sizing.AddChildAndUpdate()

add_local_sizing.Arguments.set_state(
    {
        "AddChild": "yes",
        "BOIControlName": "facesize_outlet",
        "BOIFaceLabelList": ["outlet"],
        "BOIGrowthRate": 1.15,
        "BOISize": 0.1,
    }
)
add_local_sizing.AddChildAndUpdate()

add_local_sizing.Arguments.set_state(
    {
        "AddChild": "yes",
        "BOIControlName": "facesize_wall",
        "BOIFaceLabelList": ["wall"],
        "BOIGrowthRate": 1.15,
        "BOISize": 6,
    }
)
add_local_sizing.AddChildAndUpdate()

# BOI
add_local_sizing.Arguments.set_state(
    {
        "AddChild": "yes",
        "BOIControlName": "BOI_coarse",
        "BOIExecution": "Body Of Influence",
        "BOIFaceLabelList": ["zone---coarse"],
        "BOISize": 10,
    }
)
add_local_sizing.AddChildAndUpdate()

add_local_sizing.Arguments.set_state(
    {
        "AddChild": "yes",
        "BOIControlName": "BOI_inlet",
        "BOIExecution": "Body Of Influence",
        "BOIFaceLabelList": ["zone---inlet"],
        "BOISize": 5,
    }
)
add_local_sizing.AddChildAndUpdate()

add_local_sizing.Arguments.set_state(
    {
        "AddChild": "yes",
        "BOIControlName": "BOI_outlet",
        "BOIExecution": "Body Of Influence",
        "BOIFaceLabelList": ["zone---outlet"],
        "BOISize": 5,
    }
)
add_local_sizing.AddChildAndUpdate()

add_local_sizing.Arguments.set_state(
    {
        "AddChild": "yes",
        "BOIControlName": "BOI_throat",
        "BOIExecution": "Body Of Influence",
        "BOIFaceLabelList": ["zone---throat"],
        "BOISize": 5,
    }
)
add_local_sizing.AddChildAndUpdate()

add_local_sizing.Execute()


#######################################################################################
# Generate the Surface Mesh
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# 

generate_surface_mesh = workflow.TaskObject["Generate the Surface Mesh"]
generate_surface_mesh.Arguments = dict(
    {
        "CFDSurfaceMeshControls": {
            "CurvatureNormalAngle": 12,
            "GrowthRate": 1.15,
            "MaxSize": 50,
            "MinSize": 1,
            "SizeFunctions": "Curvature",
        }
    }
)

generate_surface_mesh.Execute()
generate_surface_mesh.InsertNextTask(CommandName="ImproveSurfaceMesh")
improve_surface_mesh = workflow.TaskObject["Improve Surface Mesh"]
improve_surface_mesh.Arguments.update_dict({"FaceQualityLimit": 0.4})
improve_surface_mesh.Execute()

#######################################################################################
# Describe Geometry
# ~~~~~~~~~~~~~~~~~
# 

workflow.TaskObject["Describe Geometry"].Arguments = dict(
    CappingRequired="Yes",
    SetupType="The geometry consists of only fluid regions with no voids",
)
workflow.TaskObject["Describe Geometry"].Execute()

#######################################################################################
# Update Boundarires
# ~~~~~~~~~~~~~~~~~~
# 

workflow.TaskObject["Update Boundaries"].Execute()

#######################################################################################
# Update Regions
# ~~~~~~~~~~~~~~~~~~~~~~~~~
# 

workflow.TaskObject["Update Regions"].Execute()

