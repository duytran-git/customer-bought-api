# app/preprocessing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreprocessConfig:
    feature_columns: List[str]


def preprocess_one(
    *,
    age: Optional[float],
    income: Optional[float],
    country: Optional[str],
    price: Optional[float],
    config: PreprocessConfig,
) -> pd.DataFrame:
    """
    Turn raw customer inputs into a single-row, model-ready DataFrame.

    IMPORTANT:
    - This must match the logic you used during training.
    - We do simple safe imputations here (0 / "Unknown") because we didn't save imputers.
      In a production pipeline, you'd usually save a full sklearn Pipeline.
    """

    # 1) Build a 1-row DataFrame (raw)
    row = pd.DataFrame(
        [
            {
                "age": age,
                "income": income,
                "country": country,
                "price": price,
            }
        ]
    )

    # 2) Basic missing handling (safe defaults)
    # (Your training used median for numeric + "Unknown" for categorical,
    #  but you didn't save those medians. For a demo service, 0 is acceptable.)
    # Force numeric columns to numeric (prevents FutureWarning about silent downcasting)
    row["age"] = pd.to_numeric(row["age"], errors="coerce").fillna(0.0)
    row["income"] = pd.to_numeric(row["income"], errors="coerce").fillna(0.0)
    row["price"] = pd.to_numeric(row["price"], errors="coerce").fillna(0.0)

    # Country is categorical
    row["country"] = row["country"].fillna("Unknown").astype(str)

    # 3) Feature engineering (same as your notebook)
    row["income_log1p"] = np.log1p(row["income"])
    row["income_per_age"] = row["income"] / np.maximum(row["age"], 1.0)

    # 4) One-hot encode categoricals (country)
    row_enc = pd.get_dummies(row)

    # 5) Align columns exactly to training schema
    # Missing columns -> filled with 0, extra columns -> dropped by reindex
    row_enc = row_enc.reindex(columns=config.feature_columns, fill_value=0)

    # Ensure numeric types (sklearn likes numeric arrays)
    for c in row_enc.columns:
        row_enc[c] = pd.to_numeric(row_enc[c], errors="coerce").fillna(0.0)

    return row_enc
