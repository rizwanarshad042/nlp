from __future__ import annotations

import os
from typing import Optional

import pandas as pd


CSV_PATH = "disease_symptoms.csv"


def ensure_csv_exists(csv_path: str = CSV_PATH) -> None:
    if os.path.exists(csv_path):
        return
    df = pd.DataFrame({"disease_name": [], "symptoms": []})
    df.to_csv(csv_path, index=False)


def get_symptoms(disease_name: str, csv_path: str = CSV_PATH) -> Optional[str]:
    ensure_csv_exists(csv_path)
    df = pd.read_csv(csv_path)
    row = df[df["disease_name"].str.lower() == disease_name.strip().lower()]
    if row.empty:
        return None
    return str(row.iloc[0]["symptoms"]) if "symptoms" in row.columns else None


def upsert_disease(disease_name: str, symptoms: str, csv_path: str = CSV_PATH) -> None:
    ensure_csv_exists(csv_path)
    df = pd.read_csv(csv_path)
    mask = df["disease_name"].str.lower() == disease_name.strip().lower()
    if mask.any():
        df.loc[mask, "symptoms"] = symptoms
    else:
        df = pd.concat([df, pd.DataFrame({"disease_name": [disease_name], "symptoms": [symptoms]})], ignore_index=True)
    df.to_csv(csv_path, index=False)


__all__ = ["ensure_csv_exists", "get_symptoms", "upsert_disease", "CSV_PATH"]


