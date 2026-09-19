import os
import flopy
import depletion_constraints as dc

HERE = os.path.dirname(os.path.abspath(__file__))

sim = flopy.mf6.MFSimulation.load(sim_ws=".", exe_name=os.path.join(HERE, "mf6"))
success, buff = sim.run_simulation()
if not success:
    raise RuntimeError("MF6 run failed")

dc.apply(".")
