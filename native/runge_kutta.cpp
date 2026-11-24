// An explicit Runge-Kutta integrator for Basilisk's DynamicObject that allocates nothing per step.
//
// It does the floating-point operations of Basilisk's svIntegratorRungeKutta<N>, in the same order
// for every state element, so a run is bit-identical to one with Basilisk's integrator: each stage
// sets the states to x0, sets each derivative to the first nonzero a_ij * k_j with each later nonzero
// term added in j order, and propagates through DynParamManager::propagateStateVector, which adds
// derivative * h to each state. Where Basilisk rebuilds string-keyed maps of heap matrices at every
// stage, this holds each state's x0, k_j and increment in matrices sized once.
//
// It reads Basilisk's classes directly, so it is built against the headers of the exact Basilisk
// build it links, whose source revision wd_basilisk_revision gives, and reports with wd_abi_* the
// compile-time facts their layout depends on, for Python to compare with the ones Basilisk records.

#include <cstddef>
#include <vector>

#include <Eigen/Dense>

#include "simulation/dynamics/_GeneralModuleFiles/dynamicObject.h"
#include "simulation/dynamics/_GeneralModuleFiles/stateData.h"
#include "simulation/dynamics/_GeneralModuleFiles/stateVecIntegrator.h"

namespace wise_dynamics {

class RungeKutta : public StateVecIntegrator {
  public:
    RungeKutta(DynamicObject* dynamics, std::size_t stages, const double* a, const double* b, const double* c)
        : StateVecIntegrator(dynamics), stages(stages), a(a, a + stages * stages), b(b, b + stages), c(c, c + stages)
    {
    }

    void integrate(double currentTime, double timeStep) override
    {
        if (!bound()) bind();
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

    // Whether the slots are the states of dynPtrs, in their order.
    bool bound() const
    {
        auto slot = slots.begin();
        for (const auto* dynamics : dynPtrs) {
            for (const auto& entry : dynamics->dynManager.stateContainer.stateMap) {
                if (slot == slots.end() || slot->data != entry.second.get()) return false;
                ++slot;
            }
        }
        return slot == slots.end();
    }

    void bind()
    {
        slots.clear();
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

}  // namespace wise_dynamics

#define WD_STRINGIZE(x) #x
#define WD_VERSION(x, y, z) WD_STRINGIZE(x) "." WD_STRINGIZE(y) "." WD_STRINGIZE(z)

extern "C" {

// Gives `dynamics` an integrator of the `stages`-stage tableau (a row-major stages x stages, b and c
// of stages each) and returns it, or nullptr when it cannot be built. `dynamics` owns it, unless it
// refuses it, as one whose integration is synced to another's does: it then deletes it, and the
// pointer returned dangles, which Python only compares with the spacecraft's integrator.
__attribute__((visibility("default"))) void* wd_install_runge_kutta(
    void* dynamics, std::size_t stages, const double* a, const double* b, const double* c) noexcept
{
    try {
        auto* owner = static_cast<DynamicObject*>(dynamics);
        auto* integrator = new wise_dynamics::RungeKutta(owner, stages, a, b, c);
        owner->setIntegrator(integrator);
        return integrator;
    }
    catch (...) {
        return nullptr;
    }
}

// The source revision of the Basilisk this library was built against.
__attribute__((visibility("default"))) const char* wd_basilisk_revision() noexcept { return WD_BASILISK_REVISION; }

// The compile-time facts the layout of Basilisk's classes depends on, as Basilisk's getBuildInfo()
// "abi" names them.
__attribute__((visibility("default"))) const char* wd_abi_architecture() noexcept
{
#if defined(__x86_64__)
    return "x86_64";
#else
    return "unknown";
#endif
}

__attribute__((visibility("default"))) int wd_abi_use_cxx11_abi() noexcept { return _GLIBCXX_USE_CXX11_ABI; }

__attribute__((visibility("default"))) int wd_abi_eigen_max_align_bytes() noexcept { return EIGEN_MAX_ALIGN_BYTES; }

__attribute__((visibility("default"))) const char* wd_abi_eigen_version() noexcept
{
    return WD_VERSION(EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION);
}
}
