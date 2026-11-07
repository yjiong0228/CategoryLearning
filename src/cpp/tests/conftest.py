"""Filesystem aliases for read-only data boundary tests on each platform."""
import os
import shutil
import subprocess

import pytest


@pytest.fixture
def directory_alias():
    def create(link, target):
        if os.name != "nt":
            link.symlink_to(target, target_is_directory=True)
            return
        shell = shutil.which("powershell") or shutil.which("pwsh")
        if not shell:
            pytest.skip("PowerShell is needed to create a Windows junction")
        quote = lambda value: "'" + str(value).replace("'", "''") + "'"
        command = f"$ErrorActionPreference='Stop'; New-Item -ItemType Junction -Path {quote(link)} -Target {quote(target)} | Out-Null"
        result = subprocess.run([shell, "-NoProfile", "-NonInteractive", "-Command", command],
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
    return create
