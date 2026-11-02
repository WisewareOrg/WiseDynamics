"""The plugin loads into every pytest session of an environment with `wise-dynamics`
installed, so it loads without importing Basilisk."""


def test_a_session_using_no_sim_fixture_imports_no_basilisk(pytester):
    pytester.makepyfile(
        """
        import sys

        def test_modules():
            assert "wise_dynamics.pytest_plugin" in sys.modules
            assert [name for name in sys.modules if name.startswith("Basilisk")] == []
        """
    )
    pytester.runpytest_subprocess().assert_outcomes(passed=1)
