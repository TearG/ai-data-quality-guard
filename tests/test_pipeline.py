"""
Suíte de testes unitários automatizados para o AI Data Quality Guard.
"""

import pytest
import pandas as pd
from data_quality_pipeline import validate_data_quality, detect_data_drift

def test_validate_data_quality_valid():
    df_valid = pd.DataFrame({
        "ID_Cliente": [1, 2, 3],
        "Idade": [25, 40, 60],
        "GastoMensal": [150.0, 300.0, 500.0],
        "TipoContrato": ["Mensal", "Anual", "Bienal"]
    })
    success, failures = validate_data_quality(df_valid)
    assert success is True
    assert len(failures) == 0

def test_validate_data_quality_invalid_age_and_contract():
    df_invalid = pd.DataFrame({
        "ID_Cliente": [1, None, 3],
        "Idade": [-5, 40, 150],  # fora da faixa
        "GastoMensal": [150.0, -10.0, 500.0],  # gasto negativo
        "TipoContrato": ["Mensal", "Semanal", "Bienal"]  # "Semanal" inválido
    })
    success, failures = validate_data_quality(df_invalid)
    assert success is False
    assert len(failures) >= 3

def test_detect_data_drift_no_drift():
    df_ref = pd.DataFrame({
        "Idade": [30, 32, 35, 40, 45, 50] * 10,
        "GastoMensal": [100, 150, 200, 250, 300, 350] * 10
    })
    df_new = df_ref.copy()
    drift_report = detect_data_drift(df_ref, df_new)
    assert "Idade" in drift_report
    assert drift_report["Idade"]["drift_detected"] == False
