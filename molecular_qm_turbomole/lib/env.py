def _aoforce_script_lines() -> list[str]:
    return [
        "aoforce > aoforce.out 2>&1 || { "
        'echo "[TM ERROR] aoforce failed."; '
        "tail -n 200 aoforce.out || true; "
        "exit 24; "
        "}",
        "test -f vibspectrum || { "
        'echo "[TM ERROR] vibspectrum not produced."; '
        "ls -la; "
        "tail -n 80 aoforce.out || true; "
        "exit 25; "
        "}",
    ]


def build_frequency_script() -> str:
    lines = [
        'echo "[TM] workdir: $(pwd)"',
        'echo "[TM] aoforce: $(command -v aoforce)"',
        *_aoforce_script_lines(),
        'echo "[TM] produced files:"',
        "ls -la",
    ]
    return "\n".join(lines)


def build_hyperpolarizability_script() -> str:
    lines = [
        'echo "[TM] workdir: $(pwd)"',
        'echo "[TM] escf: $(command -v escf)"',
        "escf > escf.out 2>&1 || { "
        'echo "[TM ERROR] escf failed for hyperpol."; '
        "tail -n 200 escf.out || true; "
        "exit 3; "
        "}",
        "test -f hyperpols || { "
        'echo "[TM ERROR] hyperpols file not produced."; '
        "ls -la; "
        "tail -n 200 escf.out || true; "
        "exit 4; "
        "}",
        'echo "[TM] produced files:"',
        "ls -la",
    ]
    return "\n".join(lines)
