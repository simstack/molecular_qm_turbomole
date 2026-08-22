from typing import Any
import logging

TURBOMOLE_HYPERPOL_DEFAULT_EDELT = 0.005
HYPERPOL_MODE_UI_FIELD = "hyperpolarizability_mode"
HYPERPOL_FREQUENCY_UI_FIELD = "hyperpol_frequency_nm_ui"
HYPERPOL_EDELT_UI_FIELD = "edelt_ui"
HYPERPOL_UI_SETTINGS_FIELD = "hyperpolarizability_settings"
HYPERPOL_ENABLED_FIELD = "hyperpolarizability"
HYPERPOL_FREQUENCY_FIELD = "hyperpol_frequency_nm"
HYPERPOL_EDELT_FIELD = "edelt"
HYPERPOL_MODE_OFF = "off"
HYPERPOL_MODE_STATIC = "static"
HYPERPOL_MODE_DYNAMIC = "dynamic"
HYPERPOL_MODE_VALUES = (
    HYPERPOL_MODE_OFF,
    HYPERPOL_MODE_STATIC,
    HYPERPOL_MODE_DYNAMIC,
)
HYPERPOL_CANONICAL_FIELDS = (
    HYPERPOL_ENABLED_FIELD,
    HYPERPOL_FREQUENCY_FIELD,
    HYPERPOL_EDELT_FIELD,
    HYPERPOL_UI_SETTINGS_FIELD,
)


def hyperpolarizability_mode_schema() -> dict[str, Any]:
    return {
        "type": "string",
        "enum": list(HYPERPOL_MODE_VALUES),
        "default": HYPERPOL_MODE_OFF,
        "title": "Hyperpolarizability",
        "description": "Choose off, static/DC, or dynamic hyperpolarizability.",
    }


def hyperpolarizability_frequency_schema() -> dict[str, Any]:
    return {
        "type": "number",
        "minimum": 0.0,
        "default": 0.0,
        "description": "Wavelength in nm for dynamic hyperpolarizability.",
        "title": "Hyperpol Frequency Nm",
    }


def hyperpolarizability_mode_ui_schema() -> dict[str, Any]:
    return {"ui:widget": "select"}


def hyperpolarizability_frequency_ui_schema() -> dict[str, Any]:
    return {"ui:condition": {HYPERPOL_MODE_UI_FIELD: HYPERPOL_MODE_DYNAMIC}}


def add_hyperpolarizability_mode_validation_rules(
    schema: dict[str, Any],
) -> dict[str, Any]:
    """
    Keep hyperpolarizability fields stable in the visible form.

    A dependencies/oneOf branch controlled by hyperpolarizability_mode makes
    some JSON-schema form renderers replace the active schema subtree as the
    user changes values. That causes accordion/subform scroll jumps. The
    frequency field is always declared at the root and hidden through
    ui:condition; this helper keeps only the dynamic-mode validation rule.
    """
    dependencies = schema.get("dependencies")
    if isinstance(dependencies, dict):
        dependencies.pop(HYPERPOL_MODE_UI_FIELD, None)

    schema.setdefault("allOf", []).append(
        {
            "if": {
                "properties": {
                    HYPERPOL_MODE_UI_FIELD: {"const": HYPERPOL_MODE_DYNAMIC}
                },
                "required": [HYPERPOL_MODE_UI_FIELD],
            },
            "then": {
                "properties": {
                    HYPERPOL_FREQUENCY_UI_FIELD: hyperpolarizability_frequency_schema()
                },
                "required": [HYPERPOL_FREQUENCY_UI_FIELD],
            },
        }
    )
    return schema


def hyperpolarizability_settings_schema(
    *,
    include_edelt: bool = True,
    require_frequency: bool = False,
) -> dict[str, Any]:
    properties: dict[str, Any] = {
        HYPERPOL_FREQUENCY_UI_FIELD: hyperpolarizability_frequency_schema()
    }
    if include_edelt:
        properties[HYPERPOL_EDELT_UI_FIELD] = {
            "type": "number",
            "exclusiveMinimum": 0.0,
            "description": (
                "Electric-field step for hyperpolarizability numerical derivatives. "
                f"Leave empty to use the TURBOMOLE default ({TURBOMOLE_HYPERPOL_DEFAULT_EDELT})."
            ),
            "title": "edelt",
        }

    schema: dict[str, Any] = {
        "type": "object",
        "title": "",
        "properties": properties,
    }
    if require_frequency:
        schema["required"] = [HYPERPOL_FREQUENCY_UI_FIELD]
    return schema


def _has_payload_value(value: Any) -> bool:
    return value not in {None, ""}


def _values_equivalent(left: Any, right: Any) -> bool:
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return left == right


def _normalize_mode_value(value: Any) -> str | None:
    if not _has_payload_value(value):
        return None
    raw_value = getattr(value, "value", value)
    normalized = str(raw_value).strip().lower()
    aliases = {
        "false": HYPERPOL_MODE_OFF,
        "none": HYPERPOL_MODE_OFF,
        "no": HYPERPOL_MODE_OFF,
        "0": HYPERPOL_MODE_OFF,
        HYPERPOL_MODE_OFF: HYPERPOL_MODE_OFF,
        "true": HYPERPOL_MODE_STATIC,
        "yes": HYPERPOL_MODE_STATIC,
        "on": HYPERPOL_MODE_STATIC,
        "1": HYPERPOL_MODE_STATIC,
        "dc": HYPERPOL_MODE_STATIC,
        HYPERPOL_MODE_STATIC: HYPERPOL_MODE_STATIC,
        HYPERPOL_MODE_DYNAMIC: HYPERPOL_MODE_DYNAMIC,
    }
    return aliases.get(normalized, normalized)


def _coerce_bool(value: Any) -> bool:
    raw_value = getattr(value, "value", value)
    if isinstance(raw_value, str):
        return raw_value.strip().lower() not in {"", "0", "false", "none", "no", "off"}
    return bool(raw_value)


def _infer_mode_from_canonical(data: dict[str, Any]) -> str | None:
    if not _has_payload_value(data.get(HYPERPOL_ENABLED_FIELD)):
        return None
    if not _coerce_bool(data.get(HYPERPOL_ENABLED_FIELD)):
        return HYPERPOL_MODE_OFF
    frequency = data.get(HYPERPOL_FREQUENCY_FIELD, 0.0)
    try:
        return HYPERPOL_MODE_DYNAMIC if float(frequency or 0.0) > 0 else HYPERPOL_MODE_STATIC
    except (TypeError, ValueError):
        return HYPERPOL_MODE_DYNAMIC


def _apply_hyperpolarizability_mode(data: dict[str, Any], mode_value: Any) -> None:
    mode = _normalize_mode_value(mode_value) or _infer_mode_from_canonical(data)
    if mode is None:
        return
    if mode not in HYPERPOL_MODE_VALUES:
        raise ValueError(
            f"{HYPERPOL_MODE_UI_FIELD} must be one of {', '.join(HYPERPOL_MODE_VALUES)}."
        )

    data[HYPERPOL_MODE_UI_FIELD] = mode
    if mode == HYPERPOL_MODE_OFF:
        data[HYPERPOL_ENABLED_FIELD] = False
        data[HYPERPOL_FREQUENCY_FIELD] = 0.0
    elif mode == HYPERPOL_MODE_STATIC:
        data[HYPERPOL_ENABLED_FIELD] = True
        data[HYPERPOL_FREQUENCY_FIELD] = 0.0
    else:
        data[HYPERPOL_ENABLED_FIELD] = True
        data.setdefault(HYPERPOL_FREQUENCY_FIELD, 0.0)


def _apply_hyperpolarizability_override(
    data: dict[str, Any],
    *,
    canonical_field: str,
    ui_field: str,
    settings: Any,
    logger: logging.Logger | None,
    context: str,
) -> None:
    current_value = data.get(canonical_field)
    candidates: list[tuple[Any, str]] = []

    if isinstance(settings, dict):
        nested_ui_value = settings.get(ui_field)
        if _has_payload_value(nested_ui_value):
            candidates.append(
                (nested_ui_value, f"{HYPERPOL_UI_SETTINGS_FIELD}.{ui_field}")
            )
        nested_canonical_value = settings.get(canonical_field)
        if _has_payload_value(nested_canonical_value):
            candidates.append(
                (nested_canonical_value, f"{HYPERPOL_UI_SETTINGS_FIELD}.{canonical_field}")
            )

    top_level_ui_value = data.pop(ui_field, None)
    if _has_payload_value(top_level_ui_value):
        candidates.append((top_level_ui_value, ui_field))

    if not candidates:
        return

    chosen_value, chosen_source = candidates[0]
    conflicting_sources: list[str] = []

    if _has_payload_value(current_value) and not _values_equivalent(current_value, chosen_value):
        conflicting_sources.append(f"{canonical_field}={current_value!r}")

    for candidate_value, candidate_source in candidates[1:]:
        if not _values_equivalent(candidate_value, chosen_value):
            conflicting_sources.append(f"{candidate_source}={candidate_value!r}")

    if conflicting_sources and logger is not None:
        logger.warning(
            "%s received conflicting %s values; preferring %r from %s over %s",
            context,
            canonical_field,
            chosen_value,
            chosen_source,
            ", ".join(conflicting_sources),
        )

    data[canonical_field] = chosen_value


def normalize_hyperpolarizability_payload(
    data: Any,
    *,
    logger: logging.Logger | None = None,
    context: str = "hyperpolarizability payload",
) -> Any:
    if not isinstance(data, dict):
        return data

    settings = data.pop(HYPERPOL_UI_SETTINGS_FIELD, None)
    mode_value = data.get(HYPERPOL_MODE_UI_FIELD)
    if not _has_payload_value(mode_value) and isinstance(settings, dict):
        mode_value = settings.get(HYPERPOL_MODE_UI_FIELD)
    _apply_hyperpolarizability_override(
        data,
        canonical_field=HYPERPOL_FREQUENCY_FIELD,
        ui_field=HYPERPOL_FREQUENCY_UI_FIELD,
        settings=settings,
        logger=logger,
        context=context,
    )
    _apply_hyperpolarizability_override(
        data,
        canonical_field=HYPERPOL_EDELT_FIELD,
        ui_field=HYPERPOL_EDELT_UI_FIELD,
        settings=settings,
        logger=logger,
        context=context,
    )
    _apply_hyperpolarizability_mode(data, mode_value)
    return data


def strip_hidden_hyperpolarizability_fields_from_schema(schema: dict[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for field_name in HYPERPOL_CANONICAL_FIELDS:
            properties.pop(field_name, None)

    required = schema.get("required")
    if isinstance(required, list):
        schema["required"] = [
            field_name for field_name in required if field_name not in HYPERPOL_CANONICAL_FIELDS
        ]
    return schema


def strip_hidden_hyperpolarizability_fields_from_ui_schema(ui_schema: dict[str, Any]) -> dict[str, Any]:
    for field_name in HYPERPOL_CANONICAL_FIELDS:
        ui_schema.pop(field_name, None)
    return ui_schema
