import shutil
import subprocess
import sys

print("Python:", sys.version)

for cmd in ["git", "ffmpeg", "docker"]:
    path = shutil.which(cmd)
    print(f"{cmd}: {path or 'NOT FOUND'}")
    if path:
        try:
            out = subprocess.run([cmd, "--version"], text=True, capture_output=True, timeout=10)
            print(out.stdout.splitlines()[0] if out.stdout else out.stderr.splitlines()[0])
        except Exception as exc:
            print(f"{cmd} version check failed: {exc}")

print("CHECK_ENV_DONE")