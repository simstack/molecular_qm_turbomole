from pathlib import Path
from typing import Sequence

# Groups managed by define / turbomole2 — custom control_groups may not override these.
PROTECTED_CONTROL_GROUPS = frozenset(
    {
        "$end",
        "$coord",
        "$atoms",
        "$basis",
        "$auxbasis",
        "$jbas",
        "$jkbas",
        "$cbas",
        "$dft",
        "$scfconv",
        "$scfiterlimit",
        "$title",
    }
)


def render_dft_data_group(functional_name: str, gridsize: str) -> list[str]:
    return [
        "$dft",
        f"functional {functional_name}",
        f"gridsize {gridsize}",
    ]


def render_scfconv_data_group(scfconv: int) -> list[str]:
    return [f"$scfconv {int(scfconv)}"]


def control_group_name(first_line: str) -> str:
    token = first_line.strip().split()[0]
    return token.casefold()


def parse_control_group(raw: str, *, index: int | None = None) -> list[str]:
    """
    Parse one user-supplied TURBOMOLE control data group.

    Each entry must be a single `$name …` group (optional continuation lines).
    Raises ValueError if the text is not a valid control group.
    """
    label = f"control_groups[{index}]" if index is not None else "control_groups entry"
    if not isinstance(raw, str):
        raise ValueError(f"{label} must be a string.")

    lines = [line.rstrip() for line in raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        raise ValueError(f"{label} is empty.")

    compact = [line for line in lines if line.strip()]
    if not compact:
        raise ValueError(f"{label} is empty.")

    first = compact[0].strip()
    if first.startswith("%"):
        raise ValueError(
            f"{label} looks like an ORCA-style '%…' block; "
            "TURBOMOLE control groups must start with '$' (e.g. '$freeze')."
        )
    if not first.startswith("$"):
        raise ValueError(
            f"{label} must start with a TURBOMOLE data-group marker '$…', got: {first!r}."
        )

    name = control_group_name(first)
    if name == "$end":
        raise ValueError(f"{label} must not use the reserved '$end' marker.")
    if name in PROTECTED_CONTROL_GROUPS:
        raise ValueError(
            f"{label} uses protected group '{compact[0].strip().split()[0]}'; "
            "which is managed by turbomole2/define and cannot be overridden via control_groups."
        )

    for line in compact[1:]:
        stripped = line.strip()
        if stripped.startswith("$"):
            raise ValueError(
                f"{label} contains more than one data group ('{stripped.split()[0]}'). "
                "Provide each '$…' group as a separate control_groups entry."
            )
        if stripped.startswith("%"):
            raise ValueError(
                f"{label} contains ORCA-style '%…' content; use TURBOMOLE '$…' groups only."
            )

    return compact


def parse_control_groups(raw_groups: Sequence[str]) -> list[list[str]]:
    """Parse and validate all user control_groups; return lists for replace_control_data_groups."""
    parsed: list[list[str]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_groups):
        group = parse_control_group(raw, index=index)
        name = control_group_name(group[0])
        if name in seen:
            raise ValueError(
                f"Duplicate control group '{group[0].strip().split()[0]}' in control_groups."
            )
        seen.add(name)
        parsed.append(group)
    return parsed


def replace_control_data_groups(control_text: str, data_groups: Sequence[Sequence[str]]) -> str:
    groups = [tuple(line.rstrip() for line in group if str(line).strip()) for group in data_groups]
    if not groups:
        return control_text

    group_names = []
    for group in groups:
        first_line = group[0].strip()
        if not first_line.startswith("$"):
            raise ValueError(f"Control data groups must start with '$': {first_line}")
        group_names.append(first_line.split()[0].casefold())

    lines = control_text.splitlines()
    filtered: list[str] = []
    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip()
        normalized = stripped.split()[0].casefold() if stripped.startswith("$") else ""
        if normalized in group_names:
            idx += 1
            while idx < len(lines) and not lines[idx].strip().startswith("$"):
                idx += 1
            continue
        filtered.append(lines[idx].rstrip())
        idx += 1

    while filtered and not filtered[-1].strip():
        filtered.pop()
    if not filtered or filtered[-1].strip().casefold() != "$end":
        raise ValueError("TURBOMOLE control file is missing the final $end marker.")
    filtered.pop()

    for group in groups:
        if filtered and filtered[-1].strip():
            filtered.append("")
        filtered.extend(group)

    filtered.append("$end")
    return "\n".join(filtered) + "\n"


def patch_control_file(path: str | Path, data_groups: Sequence[Sequence[str]]) -> None:
    control_path = Path(path)
    control_text = control_path.read_text(encoding="utf-8")
    updated = replace_control_data_groups(control_text, data_groups)
    control_path.write_text(updated, encoding="utf-8")


def append_control_groups(path: str | Path, raw_groups: Sequence[str]) -> list[list[str]]:
    """Validate user control_groups and merge them into an existing control file."""
    parsed = parse_control_groups(raw_groups)
    if parsed:
        patch_control_file(path, parsed)
    return parsed
