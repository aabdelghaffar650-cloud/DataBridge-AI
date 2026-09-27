from __future__ import annotations

import math
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


fake_streamlit = types.ModuleType("streamlit")
fake_streamlit.session_state = AttrDict()
fake_streamlit.cache_data = lambda *args, **kwargs: (
    args[0] if args and callable(args[0]) and len(args) == 1 and not kwargs
    else lambda func: func
)
sys.modules.setdefault("streamlit", fake_streamlit)

from modules.feature_pipeline import (  # noqa: E402
    BooleanCoercer,
    NumericCoercer,
    _parse_numeric_series,
)


def _assert_close(actual: float, expected: float) -> None:
    assert math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-9), (
        actual,
        expected,
    )


def test_numeric_parsing_is_storage_backend_independent() -> None:
    source = pd.Series(
        [" 1 234 ", "1\u00a0234", "$2,500", "(17.5)", "25%", None],
        dtype="object",
    )
    original = source.copy(deep=True)

    parsed = _parse_numeric_series(source)
    _assert_close(parsed.iloc[0], 1234.0)
    _assert_close(parsed.iloc[1], 1234.0)
    _assert_close(parsed.iloc[2], 2500.0)
    _assert_close(parsed.iloc[3], -17.5)
    _assert_close(parsed.iloc[4], 25.0)
    assert np.isnan(parsed.iloc[5])
    pd.testing.assert_series_equal(source, original)

    percentages = _parse_numeric_series(
        pd.Series(["25%", "0.5", " 10 % "], dtype="object"),
        percentage=True,
        percent_as_fraction=True,
    )
    _assert_close(percentages.iloc[0], 0.25)
    _assert_close(percentages.iloc[1], 0.5)
    _assert_close(percentages.iloc[2], 0.10)


def test_transformers_do_not_depend_on_global_string_storage() -> None:
    numeric_frame = pd.DataFrame(
        {
            "amount": ["1\u00a0234", "$2,500", "(17.5)"],
            "rate": ["10%", "25%", "0.5"],
        }
    )
    coercer = NumericCoercer(
        semantic_types={"amount": "Currency", "rate": "Percentage"},
        percent_as_fraction=True,
    )
    transformed = coercer.fit_transform(numeric_frame)
    assert transformed.shape == (3, 2)
    _assert_close(transformed[0, 0], 1234.0)
    _assert_close(transformed[1, 0], 2500.0)
    _assert_close(transformed[2, 0], -17.5)
    _assert_close(transformed[0, 1], 0.10)
    _assert_close(transformed[1, 1], 0.25)
    _assert_close(transformed[2, 1], 0.5)

    boolean_frame = pd.DataFrame({"flag": [" YES ", "no", None, "نعم"]})
    boolean_values = BooleanCoercer().fit_transform(boolean_frame)
    assert boolean_values.shape == (4, 1)
    _assert_close(boolean_values[0, 0], 1.0)
    _assert_close(boolean_values[1, 0], 0.0)
    assert np.isnan(boolean_values[2, 0])
    _assert_close(boolean_values[3, 0], 1.0)


def test_pyarrow_string_backend_when_available() -> None:
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return

    old_storage = pd.options.mode.string_storage
    old_infer = getattr(pd.options.future, "infer_string", None)
    try:
        pd.options.mode.string_storage = "pyarrow"
        if old_infer is not None:
            pd.options.future.infer_string = True

        arrow_series = pd.Series(
            ["1\u00a0234", " 2 500 ", "$3,000", "(40)"],
            dtype="string[pyarrow]",
        )
        parsed = _parse_numeric_series(arrow_series)
        assert list(parsed.astype(float)) == [1234.0, 2500.0, 3000.0, -40.0]

        frame = pd.DataFrame(
            {
                "amount": pd.Series(
                    ["1\u00a0234", "$2,500", "(17.5)"],
                    dtype="string[pyarrow]",
                )
            }
        )
        transformed = NumericCoercer(
            semantic_types={"amount": "Currency"}
        ).fit_transform(frame)
        assert transformed.shape == (3, 1)
        assert list(transformed[:, 0]) == [1234.0, 2500.0, -17.5]
    finally:
        pd.options.mode.string_storage = old_storage
        if old_infer is not None:
            pd.options.future.infer_string = old_infer


def test_incompatible_unicode_regex_is_not_present() -> None:
    source = (ROOT / "modules" / "feature_pipeline.py").read_text(encoding="utf-8")
    assert 'str.replace(r"[\\u00A0\\s]"' not in source
    assert 'pd.StringDtype(storage="python")' in source


def main() -> None:
    test_numeric_parsing_is_storage_backend_independent()
    test_transformers_do_not_depend_on_global_string_storage()
    test_pyarrow_string_backend_when_available()
    test_incompatible_unicode_regex_is_not_present()
    print(
        "PASS: Stage 10.1 forces Python-backed parsing for numeric and boolean "
        "feature transformers, supports non-breaking spaces, remains compatible "
        "with PyArrow string columns, and preserves source data."
    )


if __name__ == "__main__":
    main()
