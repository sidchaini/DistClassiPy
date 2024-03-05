import os
import re


def get_version_from_pyproject():
    pyproject_path = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
    with open(pyproject_path, "r") as f:
        pyproject_content = f.read()
    version_match = re.search(r'^version\s*=\s*"(.*?)"', pyproject_content, re.M)
    if version_match:
        return version_match.group(1)
    else:
        raise RuntimeError("Version not found in pyproject.toml")


__version__ = get_version_from_pyproject()
