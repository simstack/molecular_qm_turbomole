import re
from pathlib import Path

import numpy as np

from molecular_qm_turbomole.lib.control_utils import (
    patch_control_file,
    render_hyperpolarizability_data_group,
)
from molecular_qm_turbomole.models.turbomole_input import (
    HyperpolarizabilityModeEnum,
    TurbomoleQMInput2,
)
from simstack.models.simple_table import SimpleTable

# Same conversion used by the historical turbomole SimpleTable / hyper_main parser.
AU_TO_1E30_ESU = 8.6393e-3
DIPOLE_NORM_EPS = 1e-12
ALIGNMENT_TOL = 1e-8
_DIPOLE_CANDIDATES = ("control", "ridft.out", "dscf.out", "job.last", "escf.out")

_NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
_PAIR_HEADER_RE = re.compile(
    r"\b(?P<number>\d+)(?:st|nd|rd|th)\s+pair\s+of\s+frequencies\b",
    re.IGNORECASE,
)
_FREQ_NM_RE = re.compile(
    rf"Frequencies\s*/\s*nm:\s+(?P<left>{_NUMBER_RE})\s+(?P<right>{_NUMBER_RE})",
    re.IGNORECASE,
)
_COMPONENT_VALUE_RE = re.compile(
    rf"\b(?P<component>[xyz]{{3}})\b\s*(?P<value>{_NUMBER_RE})",
    re.IGNORECASE,
)
_CONTROL_DIPOLE_RE = re.compile(
    rf"\bx\s+(?P<x>{_NUMBER_RE})\s+y\s+(?P<y>{_NUMBER_RE})\s+z\s+(?P<z>{_NUMBER_RE})",
    re.IGNORECASE,
)


def hyperpolarizability_requested(qm_input: TurbomoleQMInput2) -> bool:
    """True for static or dynamic β. Enum NONE is a non-empty string, so never bool() it."""
    return qm_input.hyperpolarizability != HyperpolarizabilityModeEnum.NONE


def hyperpolarizability_wavelength_nm(qm_input: TurbomoleQMInput2) -> float:
    """Optical λ for dynamic β. Static/none return 0 so control gets no frequency lines."""
    if qm_input.hyperpolarizability == HyperpolarizabilityModeEnum.STATIC:
        return 0.0
    if qm_input.hyperpolarizability == HyperpolarizabilityModeEnum.DYNAMIC:
        return float(qm_input.hyperpol_frequency_nm or 0.0)
    return 0.0


def validate_hyperpolarizability_request(qm_input: TurbomoleQMInput2) -> None:
    if not hyperpolarizability_requested(qm_input):
        return
    if int(qm_input.states or 0) > 0:
        raise ValueError(
            "Hyperpolarizability for excited states (states > 0) is not supported. "
            "Set states=0 or hyperpolarizability=none."
        )
    if qm_input.hyperpolarizability == HyperpolarizabilityModeEnum.DYNAMIC:
        wavelength_nm = hyperpolarizability_wavelength_nm(qm_input)
        if wavelength_nm <= 0.0:
            raise ValueError(
                "Dynamic hyperpolarizability requires a positive hyperpol_frequency_nm."
            )


def apply_hyperpolarizability_control(
    qm_input: TurbomoleQMInput2,
    path: str | Path = "control",
) -> list[str]:
    wavelength_nm = hyperpolarizability_wavelength_nm(qm_input)
    group = render_hyperpolarizability_data_group(wavelength_nm)
    patch_control_file(path, [group])
    control_text = Path(path).read_text(encoding="utf-8")
    verify_hyperpolarizability_control(control_text, wavelength_nm)
    return group


def verify_hyperpolarizability_control(control_text: str, wavelength_nm: float) -> None:
    if "$scfinstab hyperpol nm" not in control_text.casefold():
        raise RuntimeError(
            "Prepared TURBOMOLE hyperpolarizability control is missing "
            "'$scfinstab hyperpol nm'."
        )
    wavelength_nm = float(wavelength_nm or 0.0)
    if wavelength_nm <= 0.0:
        return
    values = _number_tokens(control_text)
    tolerance = max(1.0e-8, abs(wavelength_nm) * 1.0e-8)
    if not any(abs(value - wavelength_nm) <= tolerance for value in values):
        raise RuntimeError(
            "Dynamic TURBOMOLE hyperpolarizability requested but the prepared "
            f"'$scfinstab hyperpol nm' block does not contain lambda_nm={wavelength_nm:.10g}."
        )


def parse_hyperpols_frequency_pairs(
    hyperpols_path: Path = Path("hyperpols"),
) -> list[tuple[float, float]]:
    if not hyperpols_path.exists():
        return []
    pairs: list[tuple[float, float]] = []
    for line in hyperpols_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _FREQ_NM_RE.search(line)
        if match:
            pairs.append(
                (_parse_float_token(match.group("left")), _parse_float_token(match.group("right")))
            )
    return pairs


def verify_dynamic_hyperpols_output(
    qm_input: TurbomoleQMInput2,
    hyperpols_path: Path = Path("hyperpols"),
) -> list[tuple[float, float]]:
    if not hyperpolarizability_requested(qm_input):
        return []
    wavelength_nm = hyperpolarizability_wavelength_nm(qm_input)
    if wavelength_nm <= 0.0:
        return parse_hyperpols_frequency_pairs(hyperpols_path)

    pairs = parse_hyperpols_frequency_pairs(hyperpols_path)
    if not pairs:
        raise RuntimeError(
            "Dynamic TURBOMOLE hyperpolarizability requested "
            f"(lambda_nm={wavelength_nm:.10g}) but no frequency pairs could be parsed "
            f"from '{hyperpols_path}'."
        )
    if not any(abs(left) > 1.0e-12 or abs(right) > 1.0e-12 for left, right in pairs):
        formatted = ", ".join(f"({left:.6g}, {right:.6g})" for left, right in pairs)
        raise RuntimeError(
            "Dynamic TURBOMOLE hyperpolarizability requested "
            f"(lambda_nm={wavelength_nm:.10g}) but '{hyperpols_path}' contains only "
            f"zero-frequency static pairs: {formatted}."
        )
    return pairs


def parse_hyperpolarizability_table(
    hyperpols_path: Path = Path("hyperpols"),
    *,
    workdir: Path = Path("."),
) -> SimpleTable | None:
    """
    Historical turbomole SimpleTable: one row per frequency pair with
    z-dipole-aligned beta_zzz in 10^-30 esu. Does not write extra processed files.
    """
    hyperpols_path = Path(hyperpols_path)
    if not hyperpols_path.exists():
        return None

    try:
        tensors = _parse_beta_tensors(hyperpols_path)
    except ValueError:
        return None
    if not tensors:
        return None

    try:
        rotation = _rotation_matrix_from_workdir(Path(workdir))
    except ValueError:
        rotation = None
    table = SimpleTable(name="Hyperpolarizability")
    table.add_column("pair", "int")
    table.add_column("beta_zzz_1e30_esu", "float")

    for pair_number, beta_au in tensors:
        aligned = rotate_beta_tensor(beta_au, rotation) if rotation is not None else beta_au
        table.add_row(
            {
                "pair": pair_number,
                "beta_zzz_1e30_esu": float(aligned[2, 2, 2] * AU_TO_1E30_ESU),
            }
        )

    if not table.row:
        return None
    return table


def rotate_beta_tensor(beta_tensor: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
    return np.einsum(
        "ai,bj,ck,ijk->abc",
        rotation_matrix,
        rotation_matrix,
        rotation_matrix,
        beta_tensor,
    )


def _parse_beta_tensors(hyperpols_path: Path) -> list[tuple[int, np.ndarray]]:
    lines = hyperpols_path.read_text(encoding="utf-8", errors="replace").splitlines()
    headers: list[tuple[int, int]] = []
    for index, line in enumerate(lines):
        match = _PAIR_HEADER_RE.search(line)
        if match:
            headers.append((index, int(match.group("number"))))
    if not headers:
        raise ValueError(f"Could not find frequency-pair blocks in '{hyperpols_path}'.")

    tensors: list[tuple[int, np.ndarray]] = []
    for header_position, (start_index, pair_number) in enumerate(headers):
        next_start = (
            headers[header_position + 1][0] if header_position + 1 < len(headers) else len(lines)
        )
        tensors.append(
            (pair_number, _parse_beta_components(lines[start_index + 1 : next_start], pair_number))
        )
    return tensors


def _parse_beta_components(lines: list[str], pair_number: int) -> np.ndarray:
    beta = np.zeros((3, 3, 3), dtype=float)
    seen: set[str] = set()
    axis_index = {"x": 0, "y": 1, "z": 2}
    for line in lines:
        for match in _COMPONENT_VALUE_RE.finditer(line):
            component = match.group("component").lower()
            indices = tuple(axis_index[axis] for axis in component)
            beta[indices] = _parse_float_token(match.group("value"))
            seen.add(component)
    if len(seen) != 27:
        raise ValueError(
            f"Frequency pair {pair_number} contains {len(seen)} beta components; expected 27."
        )
    return beta


def _rotation_matrix_from_workdir(workdir: Path) -> np.ndarray | None:
    dipole = _parse_dipole_from_workdir(workdir)
    if dipole is None:
        return None
    return _rotation_matrix_to_align_with_positive_z(dipole)


def _parse_dipole_from_workdir(workdir: Path) -> np.ndarray | None:
    for name in _DIPOLE_CANDIDATES:
        path = workdir / name
        if not path.is_file():
            continue
        try:
            return _parse_dipole_vector(path)
        except ValueError:
            continue
    return None


def _parse_dipole_vector(path: Path) -> np.ndarray:
    content = path.read_text(encoding="utf-8", errors="replace")
    if path.name == "control" or "$dipole" in content.casefold():
        for line in content.splitlines():
            match = _CONTROL_DIPOLE_RE.search(line)
            if match:
                return np.array(
                    [
                        _parse_float_token(match.group("x")),
                        _parse_float_token(match.group("y")),
                        _parse_float_token(match.group("z")),
                    ],
                    dtype=float,
                )

    lines = content.splitlines()
    for line_index, line in enumerate(lines):
        if "dipole moment" not in line.lower():
            continue
        components: dict[str, float] = {}
        for block_line in lines[line_index + 1 : line_index + 20]:
            stripped = block_line.strip()
            if not stripped:
                if components:
                    break
                continue
            parts = stripped.split()
            if parts and parts[0].lower() in {"x", "y", "z"}:
                numbers = [_parse_float_token(token) for token in re.findall(_NUMBER_RE, stripped)]
                if numbers:
                    components[parts[0].lower()] = numbers[-1]
            if all(axis in components for axis in ("x", "y", "z")):
                return np.array(
                    [components["x"], components["y"], components["z"]],
                    dtype=float,
                )

    raise ValueError(f"Could not parse a dipole vector from '{path}'.")


def _rotation_matrix_to_align_with_positive_z(dipole_vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(dipole_vector))
    if norm < DIPOLE_NORM_EPS:
        raise ValueError("The permanent dipole norm is near zero; cannot define a z-dipole frame.")

    source = dipole_vector / norm
    target = np.array([0.0, 0.0, 1.0], dtype=float)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if np.isclose(dot, 1.0, atol=ALIGNMENT_TOL):
        return np.eye(3)
    if np.isclose(dot, -1.0, atol=ALIGNMENT_TOL):
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            dtype=float,
        )
    cross = np.cross(source, target)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=float,
    )
    sin_squared = float(np.dot(cross, cross))
    return np.eye(3) + skew + skew @ skew * ((1.0 - dot) / sin_squared)


def _parse_float_token(token: str) -> float:
    return float(token.replace("D", "E").replace("d", "e"))


def _number_tokens(text: str) -> list[float]:
    values: list[float] = []
    for token in re.findall(_NUMBER_RE, text):
        try:
            values.append(_parse_float_token(token))
        except ValueError:
            continue
    return values
