#
#  ISC License
#
#  Copyright (c) 2016, Autonomous Vehicle Systems Lab, University of Colorado at Boulder
#
#  Permission to use, copy, modify, and/or distribute this software for any
#  purpose with or without fee is hereby granted, provided that the above
#  copyright notice and this permission notice appear in all copies.
#
#  THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
#  WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
#  MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
#  ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
#  ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
#  OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#
r"""
Overview
--------

This script is base BSK Sim script used for the MC examples.

"""

# Get current file path
import inspect
import os
import sys

import time
import numpy as np
from Basilisk.utilities import orbitalMotion, macros, vizSupport, simHelpers

# Part C instrumentation: per-run perf_counter stamps (construction, execution,
# retention/data-pull) appended as one log line per MC run. Path is overridable
# via env var so separate campaigns (baseline vs combined-ablation) can log to
# separate files without editing this module.
import os
_MC_PERRUN_LOG = os.environ.get(
    "MC_PERRUN_LOG", "/home/wise/code/bsk-eval/logs/4x-20-mc_per_run.log"
)


def _append_perrun_log(line):
    with open(_MC_PERRUN_LOG, "a") as f:
        f.write(line + "\n")
        f.flush()

filename = inspect.getframeinfo(inspect.currentframe()).filename
path = os.path.dirname(os.path.abspath(filename))

# Import master classes: simulation base class and scenario base class
# NOTE: this is an edited copy living in bsk-eval/scripts/, not in its original
# BskSim/scenarios/ directory, so the normal path-relative-to-self imports are
# replaced with absolute paths into the original BskSim tree.
_BSKSIM_DIR = "/home/wise/code/bsk-eval/basilisk-src/examples/BskSim"
sys.path.append(_BSKSIM_DIR)
sys.path.append(_BSKSIM_DIR + '/models')
sys.path.append(_BSKSIM_DIR + '/plotting')
from BSK_masters import BSKSim, BSKScenario
import BSK_Dynamics, BSK_Fsw
import BSK_Plotting as BSK_plt


# Create your own scenario child class
class scenario_AttFeedback(BSKSim, BSKScenario):
    def __init__(self):
        # Part C instrumentation: this __init__ body is the "scenario
        # construction function" the MC Controller calls (setSimulationFunction).
        _t0 = time.perf_counter()

        super(scenario_AttFeedback, self).__init__()
        self.name = 'scenarioBskSimAttFeedbackMC'

        # declare additional class variables
        self.msgRecList = {}
        self.sNavTransName = "sNavTransMsg"
        self.attGuidName = "attGuidMsg"

        self.set_DynModel(BSK_Dynamics)
        self.set_FswModel(BSK_Fsw)

        self.configure_initial_conditions()
        self.log_outputs()

        # if this scenario is to interface with the BSK Viz, uncomment the following line
        DynModels = self.get_DynModel()
        vizSupport.enableUnityVisualization(self, DynModels.taskName, DynModels.scObject
                                            # , saveFile=__file__
                                            , rwEffectorList=DynModels.rwStateEffector
                                            )

        self._construction_seconds = time.perf_counter() - _t0

    def configure_initial_conditions(self):
        # Configure Dynamics initial conditions
        oe = orbitalMotion.ClassicElements()
        oe.a = 10000000.0  # meters
        oe.e = 0.01
        oe.i = 33.3 * macros.D2R
        oe.Omega = 48.2 * macros.D2R
        oe.omega = 347.8 * macros.D2R
        oe.f = 85.3 * macros.D2R

        DynModels = self.get_DynModel()
        mu = DynModels.gravFactory.gravBodies['earth'].mu
        rN, vN = orbitalMotion.elem2rv(mu, oe)
        orbitalMotion.rv2elem(mu, rN, vN)
        DynModels.scObject.hub.r_CN_NInit = rN  # m   - r_CN_N
        DynModels.scObject.hub.v_CN_NInit = vN  # m/s - v_CN_N
        DynModels.scObject.hub.sigma_BNInit = [[0.1], [0.2], [-0.3]]  # sigma_BN_B
        DynModels.scObject.hub.omega_BN_BInit = [[0.001], [-0.01], [0.03]]  # rad/s - omega_BN_B

    def log_outputs(self):
        FswModel = self.get_FswModel()
        DynModel = self.get_DynModel()
        samplingTime = FswModel.processTasksTimeStep

        self.msgRecList[self.attGuidName] = FswModel.attGuidMsg.recorder(samplingTime)
        self.AddModelToTask(DynModel.taskName, self.msgRecList[self.attGuidName])

        self.msgRecList[self.sNavTransName] = DynModel.simpleNavObject.transOutMsg.recorder(samplingTime)
        self.AddModelToTask(DynModel.taskName, self.msgRecList[self.sNavTransName])

        return

    def pull_outputs(self, showPlots):
        # FSW process outputs, remove first data point as it is before FSW is called
        attErrRec = self.msgRecList[self.attGuidName]

        sigma_BR = np.delete(attErrRec.sigma_BR, 0, 0)
        omega_BR_B = np.delete(attErrRec.omega_BR_B, 0, 0)

        # Plot results
        BSK_plt.clear_all_plots()
        timeData = np.delete(attErrRec.times(), 0, 0) * macros.NANO2MIN
        BSK_plt.plot_attitude_error(timeData, sigma_BR)
        BSK_plt.plot_rate_error(timeData, omega_BR_B)
        figureList = {}
        if showPlots:
            BSK_plt.show_all_plots()
        else:
            fileName = os.path.basename(os.path.splitext(__file__)[0])
            figureNames = ["attitudeErrorNorm", "rateError"]
            figureList = BSK_plt.save_all_plots(fileName, figureNames)

        return figureList


def runScenario(scenario):
    """method to initialize and execute the scenario"""
    # Part C instrumentation: this is the "execution function" the MC
    # Controller calls (setExecutionFunction). Stamped phases: execution
    # (Initialize+ExecuteSimulation) and retention/data-pull (a replica of the
    # read-only RetentionPolicy.getDataForRetention step the Controller runs
    # right after this function returns, over the same msgRecList entries, so
    # we get a real measurement of that cost without modifying Controller.py).
    _t0 = time.perf_counter()

    # Raised to one full orbital period (same orbit elements/mu as step 1: a=1.0e7 m,
    # mu=earth.mu ~3.986004360e14 m^3/s^2 -> T ~= 9952.0 s ~= 165.87 min, rounded up
    # to 166 min).
    simulationTime = macros.min2nano(166.)

    scenario.InitializeSimulation()
    scenario.modeRequest = 'inertial3D'
    scenario.ConfigureStopTime(simulationTime)
    scenario.ExecuteSimulation()

    _t1 = time.perf_counter()

    attRec = scenario.msgRecList[scenario.attGuidName]
    navRec = scenario.msgRecList[scenario.sNavTransName]
    _ = simHelpers.addTimeColumn(attRec.times(), attRec.sigma_BR)
    _ = simHelpers.addTimeColumn(attRec.times(), attRec.omega_BR_B)
    _ = simHelpers.addTimeColumn(navRec.times(), navRec.r_BN_N)

    _t2 = time.perf_counter()

    _append_perrun_log(
        f"MC_RUN pid={os.getpid()} construction_s={scenario._construction_seconds:.6f} "
        f"execution_s={_t1 - _t0:.6f} retention_s={_t2 - _t1:.6f}"
    )
    return


def run(showPlots):
    """
        The scenarios can be run with the followings setups parameters:

        Args:
            showPlots (bool): Determines if the script should display plots

    """
    scenario = scenario_AttFeedback()
    runScenario(scenario)
    scenario.pull_outputs(showPlots)

    return

if __name__ == "__main__":
    run(True)
