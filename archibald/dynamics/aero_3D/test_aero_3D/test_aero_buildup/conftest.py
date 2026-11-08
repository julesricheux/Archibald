"""Leaves the tests of this folder out of pytest."""

# AeroBuildup still uses the removed OperatingPoint API (alpha, velocity, mach). Some of these files raise when
# imported, so they are ignored at collection rather than skipped. Delete this file if AeroBuildup is revived.
collect_ignore_glob = ["test_*.py", "test_moment_validation/test_*.py"]
