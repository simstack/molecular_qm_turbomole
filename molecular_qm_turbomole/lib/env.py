import os
from pathlib import Path


def prepend_tm_env(run_script: str) -> str:
    """Prepend a POSIX-shell prelude that loads a Turbomole environment script."""
    here = Path(__file__).resolve().parent
    default_env = str((here / "scripts" / "turbomole_int_nano.sh").resolve())
    home_env = os.path.expandvars("$HOME/simstack/turbomole_int_nano.sh")

    prelude = f"""set -e
if [ "${{TM_DEBUG:-0}}" = "1" ]; then
  set -x
fi

TM_ENV_SCRIPT="${{TURBOMOLE_ENV_SCRIPT:-}}"
if [ -z "$TM_ENV_SCRIPT" ]; then
  if [ -f "{default_env}" ]; then
    TM_ENV_SCRIPT="{default_env}"
  elif [ -f "{home_env}" ]; then
    TM_ENV_SCRIPT="{home_env}"
  fi
fi

if [ -n "$TM_ENV_SCRIPT" ] && [ -f "$TM_ENV_SCRIPT" ]; then
  echo "[TM ENV] sourcing $TM_ENV_SCRIPT"
  . "$TM_ENV_SCRIPT"
fi

set -e
set +u 2>/dev/null || true

if ! command -v define >/dev/null 2>&1; then
  if ! command -v module >/dev/null 2>&1; then
    if [ -f /etc/profile.d/modules.sh ]; then
      . /etc/profile.d/modules.sh
    elif [ -f /usr/share/Modules/init/bash ]; then
      . /usr/share/Modules/init/bash
    elif [ -f /usr/share/module/init/bash ]; then
      . /usr/share/module/init/bash
    fi
  fi
  if command -v module >/dev/null 2>&1; then
    module load turbomole/7.6 || true
  fi
fi

command -v define >/dev/null 2>&1 || {{
  echo "[TM ENV ERROR] define not found in PATH. Set TURBOMOLE_ENV_SCRIPT or fix modules/env script."
  exit 127
}}
"""
    return prelude + "\n" + run_script


def build_define_script() -> str:
    lines = [
        'echo "[TM] workdir: $(pwd)"',
        'echo "[TM] define: $(command -v define)"',
        "define < define.inp > define.out 2>&1",
        'if grep -qi "define ended abnormally" define.out; then',
        '  echo "[TM ERROR] define output indicates abnormal termination."',
        "  tail -n 120 define.out || true",
        "  exit 2",
        "fi",
        "if [ ! -f control ]; then",
        '  echo "[TM ERROR] define finished but no control file was created."',
        "  tail -n 120 define.out || true",
        "  exit 2",
        "fi",
        'echo "[TM] define completed successfully"',
    ]
    return "\n".join(lines)


def build_ground_state_script(*, optimization: bool, use_ri: bool, gradients: bool) -> str:
    scf_program = "ridft" if use_ri else "dscf"
    gradient_program = "rdgrad" if use_ri else "grad"
    jobex_command = "jobex -ri" if use_ri else "jobex"
    lines = [
        'echo "[TM] workdir: $(pwd)"',
        f'echo "[TM] SCF engine: {scf_program}"',
    ]
    if optimization:
        lines.append(
            f"{jobex_command} > jobex.out 2>&1 || {{ "
            'echo "[TM ERROR] jobex failed."; '
            "tail -n 200 jobex.out || true; "
            "exit 21; "
            "}"
        )
    else:
        lines.append(
            f"{scf_program} > {scf_program}.out 2>&1 || {{ "
            f'echo "[TM ERROR] {scf_program} failed."; '
            f"tail -n 200 {scf_program}.out || true; "
            "exit 22; "
            "}"
        )
        if gradients:
            lines.append(
                f"{gradient_program} > {gradient_program}.out 2>&1 || {{ "
                f'echo "[TM ERROR] {gradient_program} failed."; '
                f"tail -n 200 {gradient_program}.out || true; "
                "exit 23; "
                "}"
            )
    lines.append('echo "[TM] produced files:"')
    lines.append("ls -la")
    return "\n".join(lines)
