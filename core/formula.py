# ════════════════════════════════════════════════════════
#  DataBridge AI — Safe Formula Evaluator
#  Uses a strict AST whitelist + pandas/numexpr evaluation.
# ════════════════════════════════════════════════════════
from __future__ import annotations

import ast
import math
import re
from typing import Any

import pandas as pd

MAX_FORMULA_LENGTH = 500

# Small numeric-only function whitelist. No attributes, imports, indexing, or builtins.
ALLOWED_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
}

ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.Call,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.BitAnd,
    ast.BitOr,
)

FORBIDDEN_TOKENS = (
    "__",
    "import",
    "lambda",
    "eval",
    "exec",
    "open",
    "read",
    "write",
    "globals",
    "locals",
    "builtins",
    "os",
    "sys",
    "subprocess",
    ";",
    "@",
    "[",
    "]",
    "{",
    "}",
    "\\",
)


def _alias_columns(formula: str, df: pd.DataFrame) -> tuple[str, dict[str, pd.Series], set[str]]:
    """Replace real column names with safe aliases and return expression + local dict."""
    expr = formula
    local_dict: dict[str, pd.Series] = {}
    allowed_names: set[str] = set(ALLOWED_FUNCTIONS)

    columns = [str(c) for c in df.columns]
    column_lookup = {str(c): c for c in df.columns}

    # Backtick syntax supports column names with spaces/symbols: `Total Amount` * 0.14
    def replace_backtick(match: re.Match[str]) -> str:
        col_name = match.group(1).strip()
        if col_name not in column_lookup:
            raise ValueError(f"Unknown column in formula: {col_name}")
        alias = f"_c{len(local_dict)}"
        local_dict[alias] = df[column_lookup[col_name]]
        allowed_names.add(alias)
        return alias

    expr = re.sub(r"`([^`]+)`", replace_backtick, expr)

    # Plain identifier columns are also supported: Price * Quantity
    valid_identifier_cols = [c for c in columns if re.fullmatch(r"[A-Za-z_]\w*", c)]
    for col in sorted(valid_identifier_cols, key=len, reverse=True):
        # Skip columns already added via backticks.
        if any(series is df[column_lookup[col]] for series in local_dict.values()):
            continue
        alias = f"_c{len(local_dict)}"
        pattern = rf"\b{re.escape(col)}\b"
        if re.search(pattern, expr):
            expr = re.sub(pattern, alias, expr)
            local_dict[alias] = df[column_lookup[col]]
            allowed_names.add(alias)

    return expr, local_dict, allowed_names


def _validate_ast(expr: str, allowed_names: set[str]) -> None:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid formula syntax: {exc.msg}") from exc

    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_AST_NODES):
            raise ValueError(f"Formula contains a blocked expression: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError(f"Formula contains an unknown or blocked name: {node.id}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in ALLOWED_FUNCTIONS:
                raise ValueError("Only whitelisted numeric functions are allowed.")


def safe_eval_formula(df: pd.DataFrame, formula: str) -> pd.Series:
    """
    Safely evaluate a user formula against a DataFrame.

    Security rules:
    - No Python engine.
    - No attribute access, indexing, imports, assignments, or builtins.
    - Columns must be plain identifiers or wrapped in backticks.
    - Evaluation runs through pandas with engine='numexpr'.
    """
    formula = (formula or "").strip()
    if not formula:
        raise ValueError("Formula is required.")
    if len(formula) > MAX_FORMULA_LENGTH:
        raise ValueError(f"Formula is too long. Maximum length is {MAX_FORMULA_LENGTH} characters.")

    lowered = formula.lower()
    if any(token in lowered for token in FORBIDDEN_TOKENS):
        raise ValueError("Formula contains blocked tokens or unsafe syntax.")

    expr, local_dict, allowed_names = _alias_columns(formula, df)
    if not local_dict:
        raise ValueError("Formula must reference at least one dataset column.")

    _validate_ast(expr, allowed_names)

    try:
        result = pd.eval(
            expr,
            engine="numexpr",
            parser="pandas",
            local_dict={**ALLOWED_FUNCTIONS, **local_dict},
        )
    except Exception as exc:
        raise ValueError(f"Formula could not be evaluated safely: {exc}") from exc

    if not isinstance(result, pd.Series):
        result = pd.Series(result, index=df.index)
    return result
