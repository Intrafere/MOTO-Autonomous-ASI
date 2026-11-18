"""Schema-aware recovery of one mislabeled key in model-authored JSON objects.

Pure and deterministic: no I/O and no model calls. A role declares the ordered
``ObjectSchema`` its prompt asks for; ``recover_key_labels`` then relabels at most
one unknown key per object as the one missing schema key, and only when every
guard below holds. Otherwise the object is returned unchanged and the role's
normal contract validation reports the problem.

Guards (deny by default):
- Only ``KeySpec(repairable=True)`` free-text string fields may be a relabel
  target, and an unknown key whose name is protected is never a source.
- After removing allowed extension keys and declared aliases, the object has
  exactly one unknown key ``u``; exactly one schema key ``m`` is missing for it
  (a missing required key, or else the one optional key whose slot ``u`` fills
  with both of that key's schema neighbours present); the object has no
  duplicate keys; every other key appears in schema order; ``u`` sits exactly in
  ``m``'s slot; and at least ``MIN_ANCHORS`` other schema keys match exactly.
- The best name match for ``u`` over all schema keys (present and missing) is
  ``m`` by normalized edit similarity, with a margin of more than ``MIN_MARGIN``
  over the second best; ``m`` qualifies by similarity >= ``MIN_SIMILARITY`` or by
  a shared token of length >= 4 that occurs in no other schema key.
- ``u``'s value is a non-empty string.

A repair only moves the model's own value to the expected key. It never creates
content, defaults, or decisions. Lists of objects are recovered per item, each
judged on its own. Dynamic-key objects are never schemas.
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Optional

from backend.shared.model_output_contract import OutputRepair

MIN_ANCHORS = 3
MIN_SIMILARITY = 0.6
MIN_MARGIN = 0.1
MIN_SHARED_TOKEN_LENGTH = 4

VALUE_TYPES = frozenset({"str", "bool", "int", "number", "list", "object", "list_object", "any"})

# Identity fences, routes, discriminators, exact-edit anchors, and decision
# fields. These may never be declared repairable and never act as a relabel
# source, whatever a role's schema says.
NEVER_REPAIRABLE_KEYS = frozenset({
    "action",
    "answer_format",
    "attempt",
    "certainty_level",
    "critique_needed",
    "decision",
    "find",
    "has_provable_theorems",
    "id",
    "idea_index",
    "is_proof_amenable",
    "needs_construction",
    "needs_edit",
    "needs_more_time",
    "needs_theorem_work",
    "needs_update",
    "new_string",
    "old_string",
    "operation",
    "outline_complete",
    "paper_id",
    "placement_preference",
    "proof_id",
    "proposal_kind",
    "replace",
    "retry_open_target_id",
    "section_complete",
    "should_remove",
    "step_id",
    "submission_number",
    "submission_type",
    "theorem_id",
    "theorem_origin",
    "topic_id",
    "topic_number",
    "unit",
    "validated",
})


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")


@dataclass(frozen=True)
class KeySpec:
    """One key of a role's ordered JSON response schema."""

    name: str
    required: bool = True
    value_type: str = "str"
    repairable: bool = False
    aliases: tuple[str, ...] = ()
    item_schema: Optional["ObjectSchema"] = None

    def __post_init__(self) -> None:
        if self.value_type not in VALUE_TYPES:
            raise ValueError(f"Unknown KeySpec value_type {self.value_type!r} for {self.name!r}")
        if self.repairable and self.value_type != "str":
            raise ValueError(f"Only free-text string keys may be repairable, not {self.name!r}")
        if self.repairable and (
            _normalize(self.name) in NEVER_REPAIRABLE_KEYS
            or any(_normalize(alias) in NEVER_REPAIRABLE_KEYS for alias in self.aliases)
        ):
            raise ValueError(f"Key {self.name!r} is protected and can never be repairable")
        if self.item_schema is not None and self.value_type not in ("object", "list_object"):
            raise ValueError(f"item_schema needs value_type object or list_object on {self.name!r}")


@dataclass(frozen=True)
class ObjectSchema:
    """An ordered object schema; ``item_id_key`` names the item's identity field."""

    name: str
    keys: tuple[KeySpec, ...]
    allow_extra: frozenset[str] = frozenset()
    item_id_key: str = ""

    def __post_init__(self) -> None:
        names = [spec.name for spec in self.keys]
        aliases = [alias for spec in self.keys for alias in spec.aliases]
        if len(set(names)) != len(names):
            raise ValueError(f"Schema {self.name!r} declares a key twice")
        if set(aliases) & set(names) or len(set(aliases)) != len(aliases):
            raise ValueError(f"Schema {self.name!r} aliases collide with keys or each other")
        if set(self.allow_extra) & (set(names) | set(aliases)):
            raise ValueError(f"Schema {self.name!r} allow_extra overlaps declared keys")


def _edit_similarity(left: str, right: str) -> float:
    a, b = _normalize(left), _normalize(right)
    if not a and not b:
        return 1.0
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (char_a != char_b),
            ))
        previous = current
    return 1.0 - previous[-1] / max(len(a), len(b))


def _tokens(name: str) -> set[str]:
    return {token for token in _normalize(name).split("_") if token}


def _has_unique_shared_token(unknown: str, target: str, schema: ObjectSchema) -> bool:
    for token in _tokens(unknown) & _tokens(target):
        if len(token) < MIN_SHARED_TOKEN_LENGTH:
            continue
        holders = sum(1 for spec in schema.keys if token in _tokens(spec.name))
        if holders == 1:
            return True
    return False


def _relabel_one(obj: dict, schema: ObjectSchema) -> Optional[tuple[str, KeySpec, float]]:
    """Return ``(unknown_key, target_spec, similarity)`` when every guard holds."""
    if getattr(obj, "duplicate_keys", ()):
        return None
    names = [spec.name for spec in schema.keys]
    index_of = {name: position for position, name in enumerate(names)}
    alias_owner = {alias: spec.name for spec in schema.keys for alias in spec.aliases}

    sequence = [
        key for key in obj
        if key not in schema.allow_extra and key not in alias_owner
    ]
    unknown = [key for key in sequence if key not in index_of]
    if len(unknown) != 1:
        return None
    source = unknown[0]
    if _normalize(source) in NEVER_REPAIRABLE_KEYS:
        return None

    satisfied = set(sequence) | {alias_owner[key] for key in obj if key in alias_owner}
    missing = [spec for spec in schema.keys if spec.name not in satisfied]
    missing_required = [spec for spec in missing if spec.required]

    position = sequence.index(source)
    others = [index_of[key] for key in sequence if key != source]
    if any(later <= earlier for earlier, later in zip(others, others[1:])):
        return None
    before = [index_of[key] for key in sequence[:position]]
    after = [index_of[key] for key in sequence[position + 1:]]
    if len(others) < MIN_ANCHORS:
        return None

    def fills_slot(spec: KeySpec) -> bool:
        slot = index_of[spec.name]
        return all(index < slot for index in before) and all(index > slot for index in after)

    if len(missing_required) > 1:
        return None
    if missing_required:
        target = missing_required[0]
    else:
        candidates = []
        for spec in missing:
            slot = index_of[spec.name]
            if slot == 0 or slot == len(names) - 1:
                continue
            if names[slot - 1] in satisfied and names[slot + 1] in satisfied and fills_slot(spec):
                candidates.append(spec)
        if len(candidates) != 1:
            return None
        target = candidates[0]

    if not target.repairable or not fills_slot(target):
        return None
    # Repairable keys are free-text strings (KeySpec enforces it): the moved
    # value must be the model's own non-empty text, never a filled-in default.
    value = obj[source]
    if not isinstance(value, str) or not value.strip():
        return None

    scores = sorted(
        ((_edit_similarity(source, spec.name), spec.name) for spec in schema.keys),
        reverse=True,
    )
    best_score, best_name = scores[0]
    if best_name != target.name:
        return None
    if len(scores) > 1 and scores[1][0] >= best_score - MIN_MARGIN:
        return None
    if best_score < MIN_SIMILARITY and not _has_unique_shared_token(source, target.name, schema):
        return None
    return source, target, best_score


def _relabelled_copy(obj: dict, source: str, target: str) -> dict:
    result = copy.copy(obj)
    items = list(obj.items())
    result.clear()
    for key, value in items:
        result[target if key == source else key] = value
    return result


def _item_id(obj: dict, schema: ObjectSchema) -> str:
    if not schema.item_id_key:
        return ""
    value = obj.get(schema.item_id_key)
    if value in (None, ""):
        for spec in schema.keys:
            if spec.name == schema.item_id_key:
                value = next((obj.get(alias) for alias in spec.aliases if obj.get(alias)), "")
    return str(value) if isinstance(value, (str, int)) else ""


def _recover(obj: Any, schema: ObjectSchema, path: str) -> tuple[Any, list[OutputRepair]]:
    if not isinstance(obj, dict):
        return obj, []
    repairs: list[OutputRepair] = []
    result = obj
    found = _relabel_one(obj, schema)
    if found is not None:
        source, target, similarity = found
        result = _relabelled_copy(obj, source, target.name)
        repairs.append(OutputRepair(
            kind="key_label",
            item_path=path,
            item_id=_item_id(result, schema),
            from_key=source,
            to_key=target.name,
            similarity=round(similarity, 3),
        ))

    for spec in schema.keys:
        if spec.item_schema is None or spec.name not in result:
            continue
        value = result[spec.name]
        child_path = f"{path}.{spec.name}" if path else spec.name
        if spec.value_type == "list_object" and isinstance(value, list):
            new_items = []
            changed = False
            for index, item in enumerate(value):
                new_item, item_repairs = _recover(item, spec.item_schema, f"{child_path}[{index}]")
                changed = changed or new_item is not item
                new_items.append(new_item)
                repairs.extend(item_repairs)
            new_value = new_items if changed else value
        elif spec.value_type == "object" and isinstance(value, dict):
            new_value, child_repairs = _recover(value, spec.item_schema, child_path)
            repairs.extend(child_repairs)
        else:
            continue
        if new_value is not value:
            if result is obj:
                result = copy.copy(obj)
            result[spec.name] = new_value
    return result, repairs


def recover_key_labels(obj: Any, schema: ObjectSchema) -> tuple[Any, list[OutputRepair]]:
    """Relabel at most one mislabeled key per object; see the module guards.

    The input is never mutated. When nothing is repaired the very same object is
    returned. Duplicate keys are detected only on objects parsed by
    ``json_parser.parse_json`` (which records them); recovery refuses those.
    """
    return _recover(obj, schema, "")
