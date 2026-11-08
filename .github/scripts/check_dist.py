"""Fail if a built wheel/sdist contains anything other than the package itself."""
import glob
import sys
import tarfile
import zipfile

pkg = sys.argv[1]  # import name, e.g. "archibald"
SDIST_ALLOWED = {"pyproject.toml", "PKG-INFO", "README.md", "LICENSE", "MANIFEST.in", "setup.cfg"}
bad = []

wheels = glob.glob("dist/*.whl")
sdists = glob.glob("dist/*.tar.gz")
if not wheels or not sdists:
    sys.exit("Expected one wheel and one sdist in dist/")

for whl in wheels:
    for name in zipfile.ZipFile(whl).namelist():
        top = name.split("/")[0]
        if top != pkg and not top.endswith(".dist-info"):
            bad.append(f"{whl}: {name}")

for sd in sdists:
    for name in tarfile.open(sd).getnames():
        parts = name.split("/")[1:]  # drop the "<name>-<version>/" root
        if not parts or not parts[0]:
            continue
        top = parts[0]
        if top == pkg or top in SDIST_ALLOWED or top.endswith(".egg-info"):
            continue
        bad.append(f"{sd}: {name}")

if bad:
    print("Unexpected files in distribution:")
    print("\n".join(f"  {b}" for b in bad[:50]))
    sys.exit(1)
print(f"OK: distributions only contain '{pkg}' and packaging metadata")
