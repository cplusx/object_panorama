from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


CONDITION_FLAG_NAMES = ("has_model_rgb", "has_edge_depth", "has_model_normal")
LEGACY_CONDITION_TYPE_TO_LABEL = {
    0: "rgb_plus_normal",
    1: "rgb_plus_edge_depth",
    2: "normal_plus_edge_depth",
}
LEGACY_CONDITION_TYPE_TO_FLAGS = {
    0: {"has_model_rgb": 1, "has_edge_depth": 0, "has_model_normal": 1},
    1: {"has_model_rgb": 1, "has_edge_depth": 1, "has_model_normal": 0},
    2: {"has_model_rgb": 0, "has_edge_depth": 1, "has_model_normal": 1},
}
FLAGS_TO_LEGACY_CONDITION_TYPE = {
    tuple(flags[name] for name in CONDITION_FLAG_NAMES): legacy_type_id
    for legacy_type_id, flags in LEGACY_CONDITION_TYPE_TO_FLAGS.items()
}


def condition_variant_specs() -> tuple[dict[str, Any], ...]:
    return tuple(condition_metadata_from_legacy_type(legacy_type_id) for legacy_type_id in sorted(LEGACY_CONDITION_TYPE_TO_FLAGS))


def condition_metadata_from_legacy_type(legacy_type_id: int) -> dict[str, Any]:
    legacy_type_id = int(legacy_type_id)
    if legacy_type_id not in LEGACY_CONDITION_TYPE_TO_FLAGS:
        raise ValueError(
            f"Unsupported legacy condition_type_id={legacy_type_id}; expected one of {sorted(LEGACY_CONDITION_TYPE_TO_FLAGS)}"
        )
    flags = dict(LEGACY_CONDITION_TYPE_TO_FLAGS[legacy_type_id])
    return {
        "flags": flags,
        "label": LEGACY_CONDITION_TYPE_TO_LABEL[legacy_type_id],
        "legacy_condition_type_id": legacy_type_id,
    }


def condition_metadata_from_flags(flags_value: Any) -> dict[str, Any]:
    flags = normalize_condition_flags(flags_value)
    flags_tuple = tuple(flags[name] for name in CONDITION_FLAG_NAMES)
    legacy_type_id = FLAGS_TO_LEGACY_CONDITION_TYPE.get(flags_tuple)
    if legacy_type_id is None:
        raise ValueError(
            "Unsupported condition flag combination "
            f"{flags}; expected one of {[LEGACY_CONDITION_TYPE_TO_FLAGS[type_id] for type_id in sorted(LEGACY_CONDITION_TYPE_TO_FLAGS)]}"
        )
    return {
        "flags": flags,
        "label": LEGACY_CONDITION_TYPE_TO_LABEL[legacy_type_id],
        "legacy_condition_type_id": legacy_type_id,
    }


def condition_metadata_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    explicit_flags = _extract_explicit_flags(record)
    legacy_type_id = record.get("condition_type_id")

    if explicit_flags is None and legacy_type_id is None:
        raise ValueError(
            "Record must include either explicit condition flags "
            f"({', '.join(CONDITION_FLAG_NAMES)}) or condition_type_id"
        )

    if explicit_flags is None:
        return condition_metadata_from_legacy_type(int(legacy_type_id))

    metadata = condition_metadata_from_flags(explicit_flags)
    if legacy_type_id is not None and int(legacy_type_id) != metadata["legacy_condition_type_id"]:
        raise ValueError(
            "condition_type_id does not match explicit condition flags: "
            f"condition_type_id={legacy_type_id}, flags={metadata['flags']}"
        )
    return metadata


def normalize_condition_flags(flags_value: Any) -> dict[str, int]:
    if isinstance(flags_value, Mapping):
        missing = [name for name in CONDITION_FLAG_NAMES if name not in flags_value]
        if missing:
            raise ValueError(f"Condition flag mapping is missing keys: {missing}")
        values = {name: _normalize_flag_value(flags_value[name], name) for name in CONDITION_FLAG_NAMES}
        extra_keys = [key for key in flags_value.keys() if key not in CONDITION_FLAG_NAMES]
        if extra_keys:
            raise ValueError(f"Condition flag mapping has unexpected keys: {extra_keys}")
        return values

    if isinstance(flags_value, Sequence) and not isinstance(flags_value, (str, bytes, bytearray)):
        if len(flags_value) != len(CONDITION_FLAG_NAMES):
            raise ValueError(
                f"condition_flags sequence must have length {len(CONDITION_FLAG_NAMES)}, got {len(flags_value)}"
            )
        return {
            name: _normalize_flag_value(value, name)
            for name, value in zip(CONDITION_FLAG_NAMES, flags_value)
        }

    raise ValueError(
        "condition flags must be provided as either a mapping with named fields or a length-3 sequence"
    )


def _extract_explicit_flags(record: Mapping[str, Any]) -> dict[str, int] | None:
    named_flags_present = [name in record for name in CONDITION_FLAG_NAMES]
    if any(named_flags_present):
        if not all(named_flags_present):
            missing = [name for name in CONDITION_FLAG_NAMES if name not in record]
            raise ValueError(f"Explicit condition flags are incomplete; missing keys: {missing}")
        return normalize_condition_flags({name: record[name] for name in CONDITION_FLAG_NAMES})

    if "condition_flags" in record:
        return normalize_condition_flags(record["condition_flags"])

    return None


def _normalize_flag_value(value: Any, name: str) -> int:
    value = int(value)
    if value not in {0, 1}:
        raise ValueError(f"Condition flag '{name}' must be 0 or 1, got {value}")
    return valuefrom __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


CONDITION_FLAG_NAMES = ("has_model_rgb", "has_edge_depth", "has_model_normal")
LEGACY_CONDITION_TYPE_TO_LABEL = {
    0: "rgb_plus_normal",
    1: "rgb_plus_edge_depth",
    2: "normal_plus_edge_depth",
}
LEGACY_CONDITION_TYPE_TO_FLAGS = {
    0: {"has_model_rgb": 1, "has_edge_depth": 0, "has_model_normal": 1},
    1: {"has_model_rgb": 1, "has_edge_depth": 1, "has_model_normal": 0},
    2: {"has_model_rgb": 0, "has_edge_depth": 1, "has_model_normal": 1},
}
FLAGS_TO_LEGACY_CONDITION_TYPE = {
    tuple(flags[name] for name in CONDITION_FLAG_NAMES): legacy_type_id
    for legacy_type_id, flags in LEGACY_CONDITION_TYPE_TO_FLAGS.items()
}


def condition_variant_specs() -> tuple[dict[str, Any], ...]:
    return tuple(condition_metadata_from_legacy_type(legacy_type_id) for legacy_type_id in sorted(LEGACY_CONDITION_TYPE_TO_FLAGS))


def condition_metadata_from_legacy_type(legacy_type_id: int) -> dict[str, Any]:
    legacy_type_id = int(legacy_type_id)
    if legacy_type_id not in LEGACY_CONDITION_TYPE_TO_FLAGS:
        raise ValueError(
            f"Unsupported legacy condition_type_id={legacy_type_id}; expected one of {sorted(LEGACY_CONDITION_TYPE_TO_FLAGS)}"
        )
    flags = dict(LEGACY_CONDITION_TYPE_TO_FLAGS[legacy_type_id])
    return {
        "flags": flags,
        "label": LEGACY_CONDITION_TYPE_TO_LABEL[legacy_type_id],
        "legacy_condition_type_id": legacy_type_id,
    }


def condition_metadata_from_flags(flags_value: Any) -> dict[str, Any]:
    flags = normalize_condition_flags(flags_value)
    flags_tuple = tuple(flags[name] for name in CONDITION_FLAG_NAMES)
    legacy_type_id = FLAGS_TO_LEGACY_CONDITION_TYPE.get(flags_tuple)
    if legacy_type_id is None:
        raise ValueError(
            "Unsupported condition flag combination "
            f"{flags}; expected one of {[LEGACY_CONDITION_TYPE_TO_FLAGS[type_id] for type_id in sorted(LEGACY_CONDITION_TYPE_TO_FLAGS)]}"
        )
    return {
        "flags": flags,
        "label": LEGACY_CONDITION_TYPE_TO_LABEL[legacy_type_id],
        "legacy_condition_type_id": legacy_type_id,
    }


def condition_metadata_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    explicit_flags = _extract_explicit_flags(record)
    legacy_type_id = record.get("condition_type_id")

    if explicit_flags is None and legacy_type_id is None:
        raise ValueError(
            "Record must include either explicit condition flags "
            f"({', '.join(CONDITION_FLAG_NAMES)}) or condition_type_id"
        )

    if explicit_flags is None:
        return condition_metadata_from_legacy_type(int(legacy_type_id))

    metadata = condition_metadata_from_flags(explicit_flags)
    if legacy_type_id is not None and int(legacy_type_id) != metadata["legacy_condition_type_id"]:
        raise ValueError(
            "condition_type_id does not match explicit condition flags: "
            f"condition_type_id={legacy_type_id}, flags={metadata['flags']}"
        )
    return metadata


def normalize_condition_flags(flags_value: Any) -> dict[str, int]:
    if isinstance(flags_value, Mapping):
        missing = [name for name in CONDITION_FLAG_NAMES if name not in flags_value]
        if missing:
            raise ValueError(f"Condition flag mapping is missing keys: {missing}")
        values = {name: _normalize_flag_value(flags_value[name], name) for name in CONDITION_FLAG_NAMES}
        extra_keys = [key for key in flags_value.keys() if key not in CONDITION_FLAG_NAMES]
        if extra_keys:
            raise ValueError(f"Condition flag mapping has unexpected keys: {extra_keys}")
        return values

    if isinstance(flags_value, Sequence) and not isinstance(flags_value, (str, bytes, bytearray)):
        if len(flags_value) != len(CONDITION_FLAG_NAMES):
            raise ValueError(
                f"condition_flags sequence must have length {len(CONDITION_FLAG_NAMES)}, got {len(flags_value)}"
            )
        return {
            name: _normalize_flag_value(value, name)
            for name, value in zip(CONDITION_FLAG_NAMES, flags_value)
        }

    raise ValueError(
        "condition flags must be provided as either a mapping with named fields or a length-3 sequence"
    )


def _extract_explicit_flags(record: Mapping[str, Any]) -> dict[str, int] | None:
    named_flags_present = [name in record for name in CONDITION_FLAG_NAMES]
    if any(named_flags_present):
        if not all(named_flags_present):
            missing = [name for name in CONDITION_FLAG_NAMES if name not in record]
            raise ValueError(f"Explicit condition flags are incomplete; missing keys: {missing}")
        return normalize_condition_flags({name: record[name] for name in CONDITION_FLAG_NAMES})

    if "condition_flags" in record:
        return normalize_condition_flags(record["condition_flags"])

    return None


def _normalize_flag_value(value: Any, name: str) -> int:
    value = int(value)
    if value not in {0, 1}:
        raise ValueError(f"Condition flag '{name}' must be 0 or 1, got {value}")
    return value