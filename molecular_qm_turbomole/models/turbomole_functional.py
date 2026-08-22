from enum import Enum
from typing import Any, Dict, List

from odmantic import EmbeddedModel, Field
from pydantic import model_validator

from simstack.models import simstack_model
from simstack.util.cleaned_json_schema import cleaned_json_schema
from simstack.util.generate_ui_schema import generate_ui_schema


class TurbomoleFunctionalEnum(str, Enum):
    """Named TURBOMOLE `$dft functional` keywords (define / LibXC shortcuts).

    Values are the strings written to the control file. Aliases such as ``B3LYP``
    are accepted by :meth:`coerce`.
    """

    # LDA
    S_VWN = "s-vwn"
    S_VWN_GAUSSIAN = "s-vwn_Gaussian"
    PWLDA = "pwlda"

    # GGA
    B_LYP = "b-lyp"
    B_VWN = "b-vwn"
    B_P = "b-p"
    PBE = "pbe"
    REVPBE = "revpbe"
    SOGGA11 = "sogga11"
    KT3 = "kt3"
    B97_D = "b97-d"
    B97_3C = "b97-3c"

    # meta-GGA
    TPSS = "tpss"
    SCAN = "scan"
    SCAN_LIBXC = "scan-libxc"
    R2SCAN = "r2scan"
    R2SCAN_3C = "r2scan-3c"
    R4SCAN = "r4scan"
    R_PP_SCAN = "r++scan"
    TASK = "task"
    REVTPSS = "revtpss"
    PKZB = "pkzb"
    TAO_MO = "tao-mo"
    M06_L = "m06-l"
    M11_L = "m11-l"
    MN12_L = "mn12-l"
    MN15_L = "mn15-l"

    # global hybrids
    BH_LYP = "bh-lyp"
    B3_LYP = "b3-lyp"
    B3_LYP_GAUSSIAN = "b3-lyp_Gaussian"
    PBE0 = "pbe0"
    TPSSH = "tpssh"
    REVTPSSH = "revtpssh"
    R2SCANH = "r2scanh"
    R2SCAN0 = "r2scan0"
    R2SCAN50 = "r2scan50"
    SCAN0 = "scan0"
    BMK = "bmk"
    B97M_V = "b97m-v"
    SOGGA11_X = "sogga-11x"
    M05 = "m05"
    M05_2X = "m05-2x"
    M06 = "m06"
    M06_2X = "m06-2x"
    MN12 = "mn12"
    MN15 = "mn15"
    PW6B95 = "pw6b95"
    PBEH_3C = "pbeh-3c"

    # range-separated hybrids
    CAM_B3LYP = "cam-b3lyp"
    TUNED_CAM_B3LYP = "tuned-cam-b3lyp"
    HSE06 = "hse06"
    HSE06_LIBXC = "hse06-libxc"
    M11 = "m11"
    REVM11 = "revM11"
    MN12_SX = "mn12-sx"
    WB97 = "wb97"
    WB97X = "wb97x"
    WB97X_V = "wb97x-v"
    WB97M_V = "wb97m-v"
    WB97X_D = "wb97x-d"
    LR_WPBE = "lr-wpbe"
    LRC_WPBEH = "lrc-wpbeh"
    CAM_QTP_00 = "cam-qtp-00"
    CAM_QTP_01 = "cam-qtp-01"
    CAM_QTP_02 = "cam-qtp-02"

    # double hybrids (energy; geometry via numerical gradient)
    B2_PLYP = "b2-plyp"
    B2GP_PLYP = "b2gp-plyp"
    XYG3 = "xyg3"

    # local hybrids
    LH07T_SVWN = "lh07t-svwn"
    LH07S_SVWN = "lh07s-svwn"
    LH12CT_SSIRPW92 = "lh12ct-ssirpw92"
    LH12CT_SSIFPW92 = "lh12ct-ssifpw92"
    LH14T_CALPBE = "lh14t-calpbe"
    LH20T = "lh20t"
    PSTS = "psts"
    MPSTS = "mpsts"
    MPSTS_NOA2 = "mpsts-noa2"
    LHJ14 = "lhj14"
    LHJ_HF = "lhj-hf"
    LHJ_HFCAL = "lhj-hfcal"
    TMHF = "tmhf"
    TMHF_3P = "tmhf-3p"

    # orbital-dependent
    LHF = "lhf"
    OEP = "oep"

    @classmethod
    def coerce(cls, raw: Any) -> "TurbomoleFunctionalEnum":
        """Map a stored / UI / legacy name onto a TURBOMOLE functional keyword."""
        if isinstance(raw, cls):
            return raw
        if raw is None or raw == "":
            return cls.B3_LYP
        if isinstance(raw, dict):
            raw = raw.get("functional", raw)
        nested = getattr(raw, "functional", raw)
        if nested is not raw:
            raw = nested
        raw = getattr(raw, "value", raw)
        text = str(raw).strip()
        if not text:
            return cls.B3_LYP
        lowered = text.lower()
        for item in cls:
            if item.value.lower() == lowered:
                return item
        mapped = _FUNCTIONAL_ALIASES.get(_alias_key(text))
        if mapped is not None:
            return cls(mapped)
        member = text.replace("-", "_").replace(" ", "_").replace("+", "P").upper()
        if member in cls.__members__:
            return cls[member]
        raise ValueError(f"Unsupported TURBOMOLE functional: {raw}")


def _alias_key(text: str) -> str:
    return text.strip().upper().replace("_", "-").replace("Ω", "W").replace("ω", "W")


# Common chemistry names and molecular_qm_models.FunctionalEnum values -> TURBOMOLE keywords.
_FUNCTIONAL_ALIASES: Dict[str, str] = {
    "S-VWN": "s-vwn",
    "SVWN": "s-vwn",
    "PWLDA": "pwlda",
    "BLYP": "b-lyp",
    "B-LYP": "b-lyp",
    "BVWN": "b-vwn",
    "B-VWN": "b-vwn",
    "BP86": "b-p",
    "B-P": "b-p",
    "BP": "b-p",
    "PBE": "pbe",
    "REVPBE": "revpbe",
    "TPSS": "tpss",
    "SCAN": "scan",
    "R2SCAN": "r2scan",
    "R2SCAN-3C": "r2scan-3c",
    "BHLYP": "bh-lyp",
    "BH-LYP": "bh-lyp",
    "BHANDHLYP": "bh-lyp",
    "B3LYP": "b3-lyp",
    "B3-LYP": "b3-lyp",
    "PBE0": "pbe0",
    "TPSSH": "tpssh",
    "M06": "m06",
    "M06-2X": "m06-2x",
    "M062X": "m06-2x",
    "M06-L": "m06-l",
    "M06L": "m06-l",
    "CAM-B3LYP": "cam-b3lyp",
    "CAMB3LYP": "cam-b3lyp",
    "HSE06": "hse06",
    "WB97": "wb97",
    "WB97X": "wb97x",
    "WB97X-D": "wb97x-d",
    "WB97XD": "wb97x-d",
    "WB97X-V": "wb97x-v",
    "WB97M-V": "wb97m-v",
    "B2PLYP": "b2-plyp",
    "B2-PLYP": "b2-plyp",
    "B97D": "b97-d",
    "B97-D": "b97-d",
    "B97-3C": "b97-3c",
    "PBEH-3C": "pbeh-3c",
}


TURBOMOLE_FUNCTIONAL_VALUES: List[str] = [item.value for item in TurbomoleFunctionalEnum]
TURBOMOLE_DEFAULT_FUNCTIONAL = TurbomoleFunctionalEnum.B3_LYP


def as_turbomole_functional_doc(raw: Any) -> Dict[str, Any]:
    functional = TurbomoleFunctionalEnum.coerce(raw)
    return {
        "field_name": "TurbomoleFunctional",
        "functional": functional.value,
    }


@simstack_model
class TurbomoleFunctional(EmbeddedModel):
    """TURBOMOLE density functional. Dispersion stays a sibling on the QM input."""

    field_name: str = "TurbomoleFunctional"
    functional: TurbomoleFunctionalEnum = Field(
        TURBOMOLE_DEFAULT_FUNCTIONAL,
        json_schema_extra={
            "enum": TURBOMOLE_FUNCTIONAL_VALUES,
            "description": "TURBOMOLE density functional ($dft functional keyword)",
            "title": "Functional",
        },
    )

    @model_validator(mode="before")
    @classmethod
    def ensure_fieldname(cls, data):
        if not isinstance(data, dict):
            if isinstance(data, (TurbomoleFunctionalEnum, str)):
                return {
                    "field_name": cls.__name__,
                    "functional": TurbomoleFunctionalEnum.coerce(data),
                }
            return data
        data.pop("id", None)
        data.pop("_id", None)
        if "field_name" not in data:
            data["field_name"] = cls.__name__
        if "functional" in data:
            data["functional"] = TurbomoleFunctionalEnum.coerce(data["functional"])
        data.pop("dispersion_correction", None)
        return data

    def keyword(self) -> str:
        value = self.functional
        if isinstance(value, TurbomoleFunctionalEnum):
            return value.value
        return TurbomoleFunctionalEnum.coerce(value).value

    @classmethod
    def json_schema(cls, recursive=True):
        schema = cleaned_json_schema(cls)
        schema["title"] = cls.__name__
        schema["description"] = "TURBOMOLE density functional"
        properties = schema.setdefault("properties", {})
        properties["functional"] = {
            "type": "string",
            "enum": list(TURBOMOLE_FUNCTIONAL_VALUES),
            "default": TURBOMOLE_DEFAULT_FUNCTIONAL.value,
            "title": "Functional",
            "description": "TURBOMOLE density functional ($dft functional keyword)",
        }
        return schema

    @classmethod
    def ui_schema(cls):
        ui_schema = generate_ui_schema(cls)
        ui_schema["field_name"] = {"ui:widget": "hidden"}
        ui_schema.setdefault("functional", {})["ui:widget"] = "select"
        return ui_schema
