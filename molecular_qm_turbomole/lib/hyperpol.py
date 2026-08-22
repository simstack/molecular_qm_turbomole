import re
from pathlib import Path

from molecular_qm_turbomole.lib.control_utils import (
    patch_control_file,
    render_hyperpolarizability_data_group,
)
from molecular_qm_turbomole.models.turbomole_input import (
    HyperpolarizabilityModeEnum,
    TurbomoleQMInput2,
)
from simstack.models.simple_table import SimpleTable

_NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
_PAIR_HEADER_RE = re.compile(
    r"\b(?P<number>\d+)(?:st|nd|rd|th)\s+pair\s+of\s+frequencies\b",
    re.IGNORECASE,
)
_FREQ_NM_RE = re.compile(
    rf"Frequencies\s*/\s*nm:\s+(?P<left>{_NUMBER_RE})\s+(?P<right>{_NUMBER_RE})",
    re.IGNORECASE,
)
_ZZZ_RE = re.compile(rf"\bzzz\s+(?P<value>{_NUMBER_RE})", re.IGNORECASE)


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
) -> SimpleTable | None:
    if not hyperpols_path.exists():
        return None

    table = SimpleTable(name="Hyperpolarizability")
    table.add_column("pair", "int")
    table.add_column("frequency_nm_1", "float")
    table.add_column("frequency_nm_2", "float")
    table.add_column("beta_zzz_au", "float")

    lines = hyperpols_path.read_text(encoding="utf-8", errors="replace").splitlines()
    headers: list[tuple[int, int]] = []
    for index, line in enumerate(lines):
        match = _PAIR_HEADER_RE.search(line)
        if match:
            headers.append((index, int(match.group("number"))))
    if not headers:
        return None

    for header_position, (start_index, pair_number) in enumerate(headers):
        next_start = (
            headers[header_position + 1][0] if header_position + 1 < len(headers) else len(lines)
        )
        block = "\n".join(lines[start_index:next_start])
        freq_match = _FREQ_NM_RE.search(block)
        zzz_match = _ZZZ_RE.search(block)
        if zzz_match is None:
            continue
        freq_1 = _parse_float_token(freq_match.group("left")) if freq_match else 0.0
        freq_2 = _parse_float_token(freq_match.group("right")) if freq_match else 0.0
        table.add_row(
            {
                "pair": pair_number,
                "frequency_nm_1": freq_1,
                "frequency_nm_2": freq_2,
                "beta_zzz_au": _parse_float_token(zzz_match.group("value")),
            }
        )

    if not table.row:
        return None
    return table


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
