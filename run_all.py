"""
run_all.py  (ROOT orchestrator)
===============================
ONE command for the whole team. For each role folder, it finds that folder's
run_all_*.py script and runs it FROM INSIDE the folder, so each role resolves
its own weights/, configs/, logs/ with the plain relative paths it was written
with. Results from all four roles print back to back.

Usage:
  python run_all.py

Layout this expects (folder names exactly as on disk):
  repo/
    run_all.py                         <- THIS FILE
    Role A/run_all_role_A.py
    Role B/run_all_role_B.py
    Role C/run_all_role_C.py
    Joint/run_all_multi_offline.py

Each role folder is expected to hold exactly one run_all_*.py. If a folder has
none, it is reported and skipped instead of crashing the whole run.
"""

import os
import glob
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# (display title, folder name exactly as it appears on disk)
ROLES = [
    ("Role A - Value-based (DQN / Double / Dueling)", "Role A"),
    ("Role B - Policy-based (REINFORCE / A2C / DDPG)", "Role B"),
    ("Role C - Planning (Dyna-Q)",                    "Role C"),
    ("Joint - Offline RL + Multi-agent RL",           "Joint"),
]


def find_script(folder_path):
    candidates = sorted(glob.glob(os.path.join(folder_path, "run_all_*.py")))
    if candidates:
        return candidates[0]
    plain = os.path.join(folder_path, "run_all.py")
    return plain if os.path.isfile(plain) else None


def run_role(title, folder):
    print("\n" + "=" * 74)
    print(title)
    print("=" * 74)
    folder_path = os.path.join(ROOT, folder)
    if not os.path.isdir(folder_path):
        print(f"[skip] folder not found: {folder}/")
        return
    script = find_script(folder_path)
    if script is None:
        print(f"[skip] no run_all_*.py found in {folder}/")
        return
    script_name = os.path.basename(script)
    print(f"(running {folder}/{script_name})")
    try:
        subprocess.run([sys.executable, script_name], cwd=folder_path, check=False)
    except Exception as e:
        print(f"[error in {folder}] {e}")


def main():
    print("Running the full team results pipeline (4 roles).")
    for title, folder in ROLES:
        run_role(title, folder)
    print("\n" + "=" * 74)
    print("ALL ROLES DONE.")
    print("=" * 74)


if __name__ == "__main__":
    main()
