# ════════════════════════════════════════════════════════
#  DataBridge AI — Replayable Source Feature Derivations
#  Stage 20.5: signed formula + string/regex replay for prediction/monitoring
# ════════════════════════════════════════════════════════
from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from core.formula import MAX_FORMULA_LENGTH, referenced_formula_columns, safe_eval_formula


DERIVATION_RECIPE_VERSION = 1
DERIVATION_STEP_VERSION = 1
DERIVATION_KIND_FORMULA = "formula"
DERIVATION_KIND_REGEX_EXTRACT = "regex_extract"
DERIVATION_KIND_VALUE_MAP = "value_map"
SUPPORTED_DERIVATION_KINDS = {
    DERIVATION_KIND_FORMULA,
    DERIVATION_KIND_REGEX_EXTRACT,
    DERIVATION_KIND_VALUE_MAP,
}
MAX_DERIVATION_STEPS = 128
MAX_DERIVED_COLUMN_NAME = 200
MAX_REGEX_PATTERN_LENGTH = 240
MAX_REGEX_CAPTURE_GROUP = 20
MAX_TEXT_SOURCE_LENGTH = 20_000
MAX_VALUE_MAP_ENTRIES = 500
MAX_VALUE_MAP_KEY_LENGTH = 240
MAX_VALUE_MAP_VALUE_LENGTH = 2_000
VALUE_MAP_UNMATCHED_STRATEGIES = {"preserve", "constant", "null"}


class FeatureDerivationError(ValueError):
    """Raised when a replayable feature derivation contract is invalid."""


def _clean_column_name(value: Any, *, field: str = "column") -> str:
    name = str(value or "").strip()
    if not name:
        raise FeatureDerivationError(f"Derived {field} name is required.")
    if len(name) > MAX_DERIVED_COLUMN_NAME:
        raise FeatureDerivationError(
            f"Derived {field} name is too long (maximum {MAX_DERIVED_COLUMN_NAME} characters)."
        )
    return name




def _clean_source_column(df: pd.DataFrame, value: Any) -> str:
    source = _clean_column_name(value, field="source")
    if source not in df.columns:
        raise FeatureDerivationError(f"Source column '{source}' does not exist.")
    return source


def _validate_text_source(series: pd.Series, source: str) -> pd.Series:
    text = series.astype("string")
    lengths = text.str.len().dropna()
    if not lengths.empty and int(lengths.max()) > MAX_TEXT_SOURCE_LENGTH:
        raise FeatureDerivationError(
            f"Source column '{source}' contains text longer than the {MAX_TEXT_SOURCE_LENGTH:,}-character safety limit."
        )
    return text


def _validate_regex_pattern(pattern: str) -> re.Pattern[str]:
    clean = str(pattern or "")
    if not clean:
        raise FeatureDerivationError("Regex pattern is required.")
    if len(clean) > MAX_REGEX_PATTERN_LENGTH:
        raise FeatureDerivationError(
            f"Regex pattern is too long (maximum {MAX_REGEX_PATTERN_LENGTH} characters)."
        )
    # Python's stdlib regex engine has no execution timeout. Block constructs most
    # associated with catastrophic backtracking or context-sensitive evaluation.
    blocked_fragments = ("(?=", "(?!", "(?<=", "(?<!", "(?(" )
    if any(fragment in clean for fragment in blocked_fragments):
        raise FeatureDerivationError("Lookarounds and conditional regex constructs are not allowed in replayable features.")
    if re.search(r"\\[1-9]", clean) or "(?P=" in clean:
        raise FeatureDerivationError("Regex backreferences are not allowed in replayable features.")
    # Reject a quantified group that itself contains an unbounded quantifier, and
    # repeated wildcard quantifiers. Titanic-style capture expressions remain valid.
    if re.search(r"\([^)]*[+*][^)]*\)\s*(?:[+*]|\{)", clean):
        raise FeatureDerivationError("Nested regex quantifiers are not allowed in replayable features.")
    if re.search(r"\.\*[^^$|)]*\.\*|\.\+[^^$|)]*\.\+", clean):
        raise FeatureDerivationError("Repeated wildcard quantifiers are not allowed in replayable features.")
    try:
        compiled = re.compile(clean)
    except re.error as exc:
        raise FeatureDerivationError(f"Invalid regex pattern: {exc}") from exc
    if compiled.groups < 1:
        raise FeatureDerivationError("Regex Extract requires at least one capture group in parentheses.")
    if compiled.groups > MAX_REGEX_CAPTURE_GROUP:
        raise FeatureDerivationError(
            f"Regex pattern has too many capture groups (maximum {MAX_REGEX_CAPTURE_GROUP})."
        )
    return compiled


def _regex_extract_series(
    df: pd.DataFrame,
    *,
    source_column: str,
    pattern: str,
    capture_group: int,
    ignore_case: bool,
    no_match_value: Any = None,
) -> pd.Series:
    source = _clean_source_column(df, source_column)
    text = _validate_text_source(df[source], source)
    compiled_base = _validate_regex_pattern(pattern)
    group = int(capture_group)
    if group < 1 or group > compiled_base.groups:
        raise FeatureDerivationError(
            f"Capture group {group} does not exist; pattern defines {compiled_base.groups} group(s)."
        )
    flags = re.IGNORECASE if bool(ignore_case) else 0
    compiled = re.compile(compiled_base.pattern, flags=flags)
    extracted = text.str.extract(compiled, expand=True).iloc[:, group - 1].astype("string")
    if no_match_value is not None:
        extracted = extracted.fillna(str(no_match_value))
    return extracted


def build_regex_extract_derivation_step(
    df: pd.DataFrame,
    output_column: str,
    source_column: str,
    pattern: str,
    *,
    capture_group: int = 1,
    ignore_case: bool = False,
    no_match_value: Any = None,
    require_new_column: bool = True,
) -> dict[str, Any]:
    """Create a deterministic regex-extraction feature without mutating its source column."""
    if not isinstance(df, pd.DataFrame):
        raise FeatureDerivationError("A DataFrame is required to define a derived feature.")
    output = _clean_column_name(output_column, field="output")
    if require_new_column and output in df.columns:
        raise FeatureDerivationError(
            f"Column '{output}' already exists. Replayable derived features must use a new column name."
        )
    source = _clean_source_column(df, source_column)
    clean_pattern = str(pattern or "")
    group = int(capture_group)
    result = _regex_extract_series(
        df,
        source_column=source,
        pattern=clean_pattern,
        capture_group=group,
        ignore_case=bool(ignore_case),
        no_match_value=no_match_value,
    )
    if len(result) != len(df) or not result.index.equals(df.index):
        raise FeatureDerivationError("Regex extraction did not preserve the dataset row/index contract.")
    return {
        "step_version": DERIVATION_STEP_VERSION,
        "kind": DERIVATION_KIND_REGEX_EXTRACT,
        "output_column": output,
        "inputs": [source],
        "source_column": source,
        "pattern": clean_pattern,
        "capture_group": group,
        "ignore_case": bool(ignore_case),
        "no_match_value": None if no_match_value is None else str(no_match_value),
    }


def _normalise_value_mapping(mapping: Mapping[Any, Any]) -> dict[str, str]:
    if not isinstance(mapping, Mapping) or not mapping:
        raise FeatureDerivationError("Value Mapping requires at least one source → replacement entry.")
    if len(mapping) > MAX_VALUE_MAP_ENTRIES:
        raise FeatureDerivationError(
            f"Value Mapping exceeds the {MAX_VALUE_MAP_ENTRIES}-entry safety limit."
        )
    clean: dict[str, str] = {}
    for raw_key, raw_value in mapping.items():
        key = str(raw_key).strip()
        value = str(raw_value).strip()
        if not key:
            raise FeatureDerivationError("Value Mapping contains an empty source value.")
        if len(key) > MAX_VALUE_MAP_KEY_LENGTH:
            raise FeatureDerivationError("A Value Mapping source value is too long.")
        if len(value) > MAX_VALUE_MAP_VALUE_LENGTH:
            raise FeatureDerivationError("A Value Mapping replacement value is too long.")
        if key in clean:
            raise FeatureDerivationError(f"Value Mapping defines '{key}' more than once.")
        clean[key] = value
    return dict(sorted(clean.items(), key=lambda item: item[0]))


def _value_map_series(
    df: pd.DataFrame,
    *,
    source_column: str,
    mapping: Mapping[Any, Any],
    unmatched_strategy: str,
    unmatched_value: Any = None,
) -> pd.Series:
    source = _clean_source_column(df, source_column)
    text = _validate_text_source(df[source], source)
    clean_mapping = _normalise_value_mapping(mapping)
    strategy = str(unmatched_strategy or "preserve").strip().lower()
    if strategy not in VALUE_MAP_UNMATCHED_STRATEGIES:
        raise FeatureDerivationError(
            "Unmatched Value Mapping strategy must be preserve, constant, or null."
        )
    result = text.map(clean_mapping)
    matched = text.isin(list(clean_mapping))
    if strategy == "preserve":
        result = result.where(matched | text.isna(), text)
    elif strategy == "constant":
        if unmatched_value is None:
            raise FeatureDerivationError("A constant unmatched value is required.")
        result = result.where(matched | text.isna(), str(unmatched_value))
    else:  # null
        result = result.where(matched, pd.NA)
    result = result.where(text.notna(), pd.NA)
    return result.astype("string")


def build_value_map_derivation_step(
    df: pd.DataFrame,
    output_column: str,
    source_column: str,
    mapping: Mapping[Any, Any],
    *,
    unmatched_strategy: str = "preserve",
    unmatched_value: Any = None,
    require_new_column: bool = True,
) -> dict[str, Any]:
    """Create a deterministic exact-value normalization feature."""
    if not isinstance(df, pd.DataFrame):
        raise FeatureDerivationError("A DataFrame is required to define a derived feature.")
    output = _clean_column_name(output_column, field="output")
    if require_new_column and output in df.columns:
        raise FeatureDerivationError(
            f"Column '{output}' already exists. Replayable derived features must use a new column name."
        )
    source = _clean_source_column(df, source_column)
    clean_mapping = _normalise_value_mapping(mapping)
    strategy = str(unmatched_strategy or "preserve").strip().lower()
    result = _value_map_series(
        df,
        source_column=source,
        mapping=clean_mapping,
        unmatched_strategy=strategy,
        unmatched_value=unmatched_value,
    )
    if len(result) != len(df) or not result.index.equals(df.index):
        raise FeatureDerivationError("Value Mapping did not preserve the dataset row/index contract.")
    return {
        "step_version": DERIVATION_STEP_VERSION,
        "kind": DERIVATION_KIND_VALUE_MAP,
        "output_column": output,
        "inputs": [source],
        "source_column": source,
        "mapping": clean_mapping,
        "unmatched_strategy": strategy,
        "unmatched_value": None if unmatched_value is None else str(unmatched_value),
    }


def build_formula_derivation_step(
    df: pd.DataFrame,
    output_column: str,
    formula: str,
    *,
    require_new_column: bool = True,
) -> dict[str, Any]:
    """Build one deterministic formula step after validating it on the active schema."""
    if not isinstance(df, pd.DataFrame):
        raise FeatureDerivationError("A DataFrame is required to define a derived feature.")
    output = _clean_column_name(output_column, field="output")
    if require_new_column and output in df.columns:
        raise FeatureDerivationError(
            f"Column '{output}' already exists. Replayable derived features must use a new column name."
        )
    clean_formula = str(formula or "").strip()
    if not clean_formula:
        raise FeatureDerivationError("Formula is required.")
    if len(clean_formula) > MAX_FORMULA_LENGTH:
        raise FeatureDerivationError(
            f"Formula is too long. Maximum length is {MAX_FORMULA_LENGTH} characters."
        )
    try:
        inputs = referenced_formula_columns(df, clean_formula)
        # Evaluate once now so invalid dtypes/operations fail before the dataset mutation.
        result = safe_eval_formula(df, clean_formula)
    except ValueError as exc:
        raise FeatureDerivationError(str(exc)) from exc
    if len(result) != len(df) or not result.index.equals(df.index):
        raise FeatureDerivationError("The formula did not preserve the dataset row/index contract.")
    if output in inputs:
        raise FeatureDerivationError("A derived feature cannot depend on itself.")
    return {
        "step_version": DERIVATION_STEP_VERSION,
        "kind": DERIVATION_KIND_FORMULA,
        "output_column": output,
        "inputs": list(map(str, inputs)),
        "formula": clean_formula,
    }


def normalise_feature_derivation_recipe(recipe: Any) -> list[dict[str, Any]]:
    """Return a strict, JSON-safe derivation recipe or raise on an unsafe contract."""
    if recipe in (None, "", [], ()):  # common empty states
        return []
    if isinstance(recipe, Mapping):
        if int(recipe.get("recipe_version", DERIVATION_RECIPE_VERSION)) != DERIVATION_RECIPE_VERSION:
            raise FeatureDerivationError("Unsupported feature-derivation recipe version.")
        steps = recipe.get("steps", [])
    else:
        steps = recipe
    if not isinstance(steps, (list, tuple)):
        raise FeatureDerivationError("Feature-derivation recipe must be an ordered list.")
    if len(steps) > MAX_DERIVATION_STEPS:
        raise FeatureDerivationError(
            f"Feature-derivation recipe exceeds the {MAX_DERIVATION_STEPS}-step safety limit."
        )

    clean: list[dict[str, Any]] = []
    outputs: set[str] = set()
    available_outputs: set[str] = set()
    for index, raw_step in enumerate(steps):
        if not isinstance(raw_step, Mapping):
            raise FeatureDerivationError(f"Derivation step {index + 1} is not an object.")
        version = int(raw_step.get("step_version", DERIVATION_STEP_VERSION))
        if version != DERIVATION_STEP_VERSION:
            raise FeatureDerivationError(f"Unsupported derivation step version at step {index + 1}.")
        kind = str(raw_step.get("kind", "")).strip()
        if kind not in SUPPORTED_DERIVATION_KINDS:
            raise FeatureDerivationError(f"Unsupported derivation kind '{kind}' at step {index + 1}.")
        output = _clean_column_name(raw_step.get("output_column"), field="output")
        if output in outputs:
            raise FeatureDerivationError(f"Derived column '{output}' is defined more than once.")
        inputs_raw = raw_step.get("inputs", [])
        if not isinstance(inputs_raw, (list, tuple)) or not inputs_raw:
            raise FeatureDerivationError(f"Derived column '{output}' has no recorded inputs.")
        inputs = [_clean_column_name(value, field="input") for value in inputs_raw]
        if len(inputs) != len(set(inputs)):
            raise FeatureDerivationError(f"Derived column '{output}' has duplicate inputs.")
        if output in inputs:
            raise FeatureDerivationError(f"Derived column '{output}' depends on itself.")

        if kind == DERIVATION_KIND_FORMULA:
            formula = str(raw_step.get("formula", "")).strip()
            if not formula or len(formula) > MAX_FORMULA_LENGTH:
                raise FeatureDerivationError(f"Derived column '{output}' has an invalid formula.")
            step = {
                "step_version": DERIVATION_STEP_VERSION,
                "kind": kind,
                "output_column": output,
                "inputs": inputs,
                "formula": formula,
            }
        elif kind == DERIVATION_KIND_REGEX_EXTRACT:
            source = _clean_column_name(raw_step.get("source_column"), field="source")
            if inputs != [source]:
                raise FeatureDerivationError(
                    f"Regex-derived column '{output}' has inconsistent recorded inputs."
                )
            pattern = str(raw_step.get("pattern", ""))
            compiled = _validate_regex_pattern(pattern)
            capture_group = int(raw_step.get("capture_group", 1))
            if capture_group < 1 or capture_group > compiled.groups:
                raise FeatureDerivationError(
                    f"Regex-derived column '{output}' has an invalid capture group."
                )
            step = {
                "step_version": DERIVATION_STEP_VERSION,
                "kind": kind,
                "output_column": output,
                "inputs": inputs,
                "source_column": source,
                "pattern": pattern,
                "capture_group": capture_group,
                "ignore_case": bool(raw_step.get("ignore_case", False)),
                "no_match_value": (
                    None if raw_step.get("no_match_value") is None else str(raw_step.get("no_match_value"))
                ),
            }
        elif kind == DERIVATION_KIND_VALUE_MAP:
            source = _clean_column_name(raw_step.get("source_column"), field="source")
            if inputs != [source]:
                raise FeatureDerivationError(
                    f"Mapped-derived column '{output}' has inconsistent recorded inputs."
                )
            mapping = _normalise_value_mapping(raw_step.get("mapping", {}))
            unmatched_strategy = str(raw_step.get("unmatched_strategy", "preserve")).strip().lower()
            if unmatched_strategy not in VALUE_MAP_UNMATCHED_STRATEGIES:
                raise FeatureDerivationError(
                    f"Mapped-derived column '{output}' has an invalid unmatched strategy."
                )
            unmatched_value = raw_step.get("unmatched_value")
            if unmatched_strategy == "constant" and unmatched_value is None:
                raise FeatureDerivationError(
                    f"Mapped-derived column '{output}' requires a constant unmatched value."
                )
            step = {
                "step_version": DERIVATION_STEP_VERSION,
                "kind": kind,
                "output_column": output,
                "inputs": inputs,
                "source_column": source,
                "mapping": mapping,
                "unmatched_strategy": unmatched_strategy,
                "unmatched_value": None if unmatched_value is None else str(unmatched_value),
            }
        else:  # pragma: no cover - guarded by SUPPORTED_DERIVATION_KINDS
            raise FeatureDerivationError(f"Unsupported derivation kind '{kind}'.")
        clean.append(step)
        outputs.add(output)
        available_outputs.add(output)
    return clean


def append_derivation_step(recipe: Any, step: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Append one new derivation step without allowing ambiguous output redefinition."""
    current = normalise_feature_derivation_recipe(recipe)
    candidate = normalise_feature_derivation_recipe([*current, copy.deepcopy(dict(step))])
    return candidate


def _step_map(recipe: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(step["output_column"]): step for step in recipe}


def source_columns_for_pipeline(
    recipe: Any,
    pipeline_required_columns: Sequence[str],
) -> list[str]:
    """Resolve the raw/source columns needed to recreate the pipeline input columns."""
    clean = normalise_feature_derivation_recipe(recipe)
    by_output = _step_map(clean)
    source: list[str] = []
    visiting: set[str] = set()
    resolved: set[str] = set()

    def resolve(column: str) -> None:
        if column in resolved:
            return
        if column in visiting:
            raise FeatureDerivationError("Feature-derivation recipe contains a dependency cycle.")
        step = by_output.get(column)
        if step is None:
            if column not in source:
                source.append(column)
            resolved.add(column)
            return
        visiting.add(column)
        for dependency in step.get("inputs", []):
            resolve(str(dependency))
        visiting.remove(column)
        resolved.add(column)

    for value in pipeline_required_columns:
        resolve(str(value))
    return source


def effective_derivation_recipe(
    recipe: Any,
    pipeline_required_columns: Sequence[str],
) -> list[dict[str, Any]]:
    """Keep only derivations required (directly or transitively) by the fitted pipeline."""
    clean = normalise_feature_derivation_recipe(recipe)
    by_output = _step_map(clean)
    needed_outputs: set[str] = set()
    visiting: set[str] = set()

    def visit(column: str) -> None:
        if column in visiting:
            raise FeatureDerivationError("Feature-derivation recipe contains a dependency cycle.")
        step = by_output.get(column)
        if step is None or column in needed_outputs:
            return
        visiting.add(column)
        for dependency in step.get("inputs", []):
            visit(str(dependency))
        visiting.remove(column)
        needed_outputs.add(column)

    for value in pipeline_required_columns:
        visit(str(value))
    return [copy.deepcopy(step) for step in clean if str(step["output_column"]) in needed_outputs]


def _replay_clean_recipe(df: pd.DataFrame, clean: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    work = df.copy(deep=True)
    for step in clean:
        output = str(step["output_column"])
        inputs = list(map(str, step.get("inputs", [])))
        missing = [column for column in inputs if column not in work.columns]
        if missing:
            raise FeatureDerivationError(
                f"Cannot derive '{output}'; missing source column(s): " + ", ".join(missing)
            )
        kind = str(step.get("kind"))
        if kind == DERIVATION_KIND_FORMULA:
            formula = str(step.get("formula", ""))
            try:
                actual_inputs = referenced_formula_columns(work, formula)
            except ValueError as exc:
                raise FeatureDerivationError(f"Cannot replay '{output}': {exc}") from exc
            if actual_inputs != inputs:
                raise FeatureDerivationError(
                    f"Recorded dependencies for derived feature '{output}' do not match its signed formula."
                )
            try:
                result = safe_eval_formula(work, formula)
            except ValueError as exc:
                raise FeatureDerivationError(f"Cannot replay '{output}': {exc}") from exc
        elif kind == DERIVATION_KIND_REGEX_EXTRACT:
            source = str(step.get("source_column", ""))
            if inputs != [source]:
                raise FeatureDerivationError(
                    f"Recorded dependencies for regex-derived feature '{output}' are inconsistent."
                )
            result = _regex_extract_series(
                work,
                source_column=source,
                pattern=str(step.get("pattern", "")),
                capture_group=int(step.get("capture_group", 1)),
                ignore_case=bool(step.get("ignore_case", False)),
                no_match_value=step.get("no_match_value"),
            )
        elif kind == DERIVATION_KIND_VALUE_MAP:
            source = str(step.get("source_column", ""))
            if inputs != [source]:
                raise FeatureDerivationError(
                    f"Recorded dependencies for mapped-derived feature '{output}' are inconsistent."
                )
            result = _value_map_series(
                work,
                source_column=source,
                mapping=step.get("mapping", {}),
                unmatched_strategy=str(step.get("unmatched_strategy", "preserve")),
                unmatched_value=step.get("unmatched_value"),
            )
        else:
            raise FeatureDerivationError(f"Unsupported derivation kind '{kind}'.")
        if len(result) != len(work) or not result.index.equals(work.index):
            raise FeatureDerivationError(
                f"Derived feature '{output}' did not preserve the row/index contract."
            )
        # Always recompute/overwrite internally. User-supplied derived values are not trusted.
        work[output] = result
    return work


def replay_feature_derivations(
    df: pd.DataFrame,
    recipe: Any,
    *,
    pipeline_required_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Replay signed derivations on an isolated copy of source/prediction data."""
    if not isinstance(df, pd.DataFrame):
        raise FeatureDerivationError("Feature derivation input must be a DataFrame.")
    clean = normalise_feature_derivation_recipe(recipe)
    if pipeline_required_columns is not None:
        clean = effective_derivation_recipe(clean, pipeline_required_columns)
    return _replay_clean_recipe(df, clean)


def validate_recipe_against_dataframe(
    df: pd.DataFrame,
    recipe: Any,
    pipeline_required_columns: Sequence[str],
) -> list[dict[str, Any]]:
    """Verify the active derived columns still match their recorded deterministic recipe."""
    if not isinstance(df, pd.DataFrame):
        raise FeatureDerivationError("Training data must be a DataFrame.")
    clean = effective_derivation_recipe(recipe, pipeline_required_columns)
    if not clean:
        return []
    for step in clean:
        output = str(step["output_column"])
        if output not in df.columns:
            raise FeatureDerivationError(
                f"Recorded derived feature '{output}' is missing from the training dataset."
            )

    # Remove the derived outputs so replay must reconstruct them from source columns.
    base = df.drop(columns=[str(step["output_column"]) for step in clean], errors="ignore").copy(deep=True)
    replayed = _replay_clean_recipe(base, clean)
    for step in clean:
        output = str(step["output_column"])
        expected = df[output]
        actual = replayed[output]
        if not actual.equals(expected):
            raise FeatureDerivationError(
                f"Derived feature '{output}' no longer matches its recorded recipe. "
                "Recreate the feature before training/export so production replay cannot diverge."
            )
    return clean


def feature_derivation_recipe_fingerprint(recipe: Any) -> str:
    clean = normalise_feature_derivation_recipe(recipe)
    raw = json.dumps(
        {"recipe_version": DERIVATION_RECIPE_VERSION, "steps": clean},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def recipe_output_columns(recipe: Any) -> list[str]:
    return [str(step["output_column"]) for step in normalise_feature_derivation_recipe(recipe)]


def prepare_training_dataframe_for_contract(
    df: pd.DataFrame,
    recipe: Any,
    pipeline_required_columns: Sequence[str],
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Validate any present derived columns and return a fully replayed training-contract frame.

    This accepts either the active training frame (where derivations already exist)
    or a fresh raw retraining frame (where they do not). Present derived values must
    match the deterministic recipe; missing derived values are recreated.
    """
    if not isinstance(df, pd.DataFrame):
        raise FeatureDerivationError("Training data must be a DataFrame.")
    clean = effective_derivation_recipe(recipe, pipeline_required_columns)
    if not clean:
        return df.copy(deep=True), []
    output_columns = [str(step["output_column"]) for step in clean]
    present = [column for column in output_columns if column in df.columns]
    base = df.drop(columns=output_columns, errors="ignore").copy(deep=True)
    replayed = _replay_clean_recipe(base, clean)
    for output in present:
        if not replayed[output].equals(df[output]):
            raise FeatureDerivationError(
                f"Derived feature '{output}' no longer matches its recorded recipe. "
                "Recreate the feature before training/export so production replay cannot diverge."
            )
    return replayed, clean
