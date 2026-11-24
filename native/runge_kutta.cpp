// An explicit Runge-Kutta integrator for Basilisk's DynamicObject that allocates nothing per step.
//
// It does the floating-point operations of Basilisk's svIntegratorRungeKutta<N>, in the same order
// for every state element, so a run is bit-identical to one with Basilisk's integrator: each stage
// sets the states to x0, sets each derivative to the first nonzero a_ij * k_j with each later nonzero
// term added in j order, and propagates through DynParamManager::propagateStateVector, which adds
// derivative * h to each state. It holds each state's x0, k_j and increment in matrices sized once.
// ADR 0004 rev 1 records why WiseDynamics compiles it.
//
// It reads Basilisk's classes directly, so it is built against the headers of the exact Basilisk
// build it links, whose source revision wd_basilisk_revision gives; wd_abi gives Basilisk's own
// description of the build facts this library was compiled with, for Python to compare with the ones
// Basilisk records.

#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "architecture/utilities/bskAbiDescriptor.h"
#include "simulation/dynamics/_GeneralModuleFiles/dynamicObject.h"
#include "simulation/dynamics/_GeneralModuleFiles/stateData.h"
#include "simulation/dynamics/_GeneralModuleFiles/stateVecIntegrator.h"

namespace wise_dynamics {

class RungeKutta : public StateVecIntegrator {
  public:
    RungeKutta(DynamicObject* dynamics, std::size_t stageCount, const double* aRowMajor, const double* weights,
               const double* nodes)
        : StateVecIntegrator(dynamics),
          stages(stageCount),
          a(aRowMajor, aRowMajor + stageCount * stageCount),
          b(weights, weights + stageCount),
          c(nodes, nodes + stageCount)
    {
    }

    // The states are registered when the simulation initialises, before the first step, so the
    // slots are bound once, at the first step.
    void integrate(double currentTime, double timeStep) override
    {
        if (slots.empty()) bind();
        for (auto& slot : slots) slot.x0 = slot.data->getStateReference();
        for (std::size_t stage = 0; stage < stages; stage++) {
            propagate(timeStep, &a[stage * stages], stage);
            for (auto* dynamics : dynPtrs) dynamics->equationsOfMotion(currentTime + c[stage] * timeStep, timeStep);
            for (auto& slot : slots) slot.k[stage] = slot.data->getStateDerivReference();
        }
        propagate(timeStep, b.data(), stages);
    }

  private:
    struct Slot {
        StateData* data;
        Eigen::MatrixXd x0;
        std::vector<Eigen::MatrixXd> k;
        Eigen::MatrixXd increment;
    };

    void bind()
    {
        for (auto* dynamics : dynPtrs) {
            for (auto& entry : dynamics->dynManager.stateContainer.stateMap) {
                StateData* data = entry.second.get();
                slots.push_back({data, data->getStateReference(),
                                 std::vector<Eigen::MatrixXd>(stages, data->getStateDerivReference()),
                                 data->getStateDerivReference()});
            }
        }
    }

    // Sets every state to x0 + h * sum_j coefficients[j] * k_j over j < maxStage, as
    // svIntegratorRungeKutta::propagateStateWithKVectors does, and to x0 when every coefficient is 0.
    void propagate(double timeStep, const double* coefficients, std::size_t maxStage)
    {
        bool any = false;
        for (std::size_t j = 0; j < maxStage; j++) any = any || coefficients[j] != 0;
        for (auto& slot : slots) {
            slot.data->setState(slot.x0);
            if (!any) continue;
            bool first = true;
            for (std::size_t j = 0; j < maxStage; j++) {
                if (coefficients[j] == 0) continue;
                if (first) slot.increment.noalias() = slot.k[j] * coefficients[j];
                else slot.increment.noalias() += slot.k[j] * coefficients[j];
                first = false;
            }
            slot.data->setDerivative(slot.increment);
        }
        if (!any) return;
        for (auto* dynamics : dynPtrs) dynamics->dynManager.propagateStateVector(timeStep);
    }

    const std::size_t stages;
    const std::vector<double> a, b, c;
    std::vector<Slot> slots;
};

// Basilisk's own description of the build facts this library is compiled with, as JSON keyed by the
// dotted keys of Basilisk.getBuildInfo()["abi"], each the fact Basilisk records there.
std::string abiJson()
{
    const bsk::abi::CxxAbiInfo info = bsk::abi::createCxxAbiInfo();
    std::ostringstream json;
    json << "{"
         << "\"target.architecture\": \"" << info.targetArchitecture << "\", "
         << "\"target.instructionSet\": \"" << info.instructionSet << "\", "
         << "\"cxx.standardLibrary.useCxx11Abi\": " << info.libstdcxxUseCxx11Abi << ", "
         << "\"cxx.layoutCanaries.stdStringSize\": " << info.stdStringSize << ", "
         << "\"cxx.layoutCanaries.stdVectorIntSize\": " << info.stdVectorIntSize << ", "
         << "\"cxx.layoutCanaries.sysModelSize\": " << info.sysModelSize << ", "
         << "\"cxx.layoutCanaries.sysModelAlignment\": " << info.sysModelAlignment << ", "
         << "\"dependencies.eigen.version\": \"" << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "."
         << EIGEN_MINOR_VERSION << "\", "
         << "\"dependencies.eigen.vectorization\": \"" << info.eigenVectorization << "\", "
         << "\"dependencies.eigen.maxAlignBytes\": " << info.eigenMaxAlignBytes << ", "
         << "\"dependencies.eigen.maxStaticAlignBytes\": " << info.eigenMaxStaticAlignBytes << ", "
         << "\"dependencies.eigen.vectorXdSize\": " << info.eigenVectorXdSize << ", "
         << "\"dependencies.eigen.matrix3dSize\": " << info.eigenMatrix3dSize << "}";
    return json.str();
}

}  // namespace wise_dynamics

extern "C" {

// Gives `dynamics` an integrator of the `stages`-stage tableau (a row-major stages x stages, b and c
// of stages each), which `dynamics` then owns, and returns it. Returns nullptr when the integrator
// cannot be built, or when `dynamics` refuses and deletes it, as one synced to another's integration does.
__attribute__((visibility("default"))) void* wd_install_runge_kutta(
    void* dynamics, std::size_t stages, const double* a, const double* b, const double* c) noexcept
{
    try {
        auto* owner = static_cast<DynamicObject*>(dynamics);
        auto* integrator = new wise_dynamics::RungeKutta(owner, stages, a, b, c);
        owner->setIntegrator(integrator);
        return owner->integrator == integrator ? integrator : nullptr;
    }
    catch (...) {
        return nullptr;
    }
}

// Whether `dynamics` integrates with this library's integrator.
__attribute__((visibility("default"))) bool wd_integrates_here(void* dynamics) noexcept
{
    return dynamic_cast<wise_dynamics::RungeKutta*>(static_cast<DynamicObject*>(dynamics)->integrator) != nullptr;
}

// The source revision of the Basilisk this library was built against.
__attribute__((visibility("default"))) const char* wd_basilisk_revision() noexcept { return WD_BASILISK_REVISION; }

// Basilisk's description of this library's build facts, as `abiJson` gives it; the string lives as long as
// the library.
__attribute__((visibility("default"))) const char* wd_abi() noexcept
{
    static const std::string json = wise_dynamics::abiJson();
    return json.c_str();
}
}
