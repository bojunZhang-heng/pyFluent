import os
import ansys.fluent.core as pyfluent

solver_session = pyfluent.launch_fluent(
    precision="double",
    processor_count=2,
    mode="solver",
)

print(solver_session.get_fluent_version())
