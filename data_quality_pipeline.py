"""
AI Data Quality Guard — MLOps & QA Pipeline
===========================================
Pipeline robusto para validação de esquemas, asserções de dados e detecção de Data Drift
antes da etapa de inferência ou retreinamento em produção.
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

def validate_data_quality(df):
    """
    Executa testes de asserção de qualidade nos dados usando regras de negócio.
    
    Retorna:
        tuple: (sucesso: bool, relatório_de_falhas: list)
    """
    failures = []
    
    # 1. Validar colunas obrigatórias
    expected_cols = ["ID_Cliente", "Idade", "GastoMensal", "TipoContrato"]
    for col in expected_cols:
        if col not in df.columns:
            failures.append(f"Coluna ausente: {col}")
            
    if failures:
        return False, failures

    # 2. Validar ausência de nulos em ID_Cliente
    null_ids = df["ID_Cliente"].isnull().sum()
    if null_ids > 0:
        failures.append(f"ID_Cliente possui {null_ids} valores nulos.")
        
    # 3. Validar faixa de Idade (0 a 120 anos)
    null_idades = df["Idade"].isnull().sum()
    if null_idades > 0:
        failures.append(f"Idade possui {null_idades} valores nulos.")
    invalid_idades = df[(df["Idade"] < 0) | (df["Idade"] > 120)]
    if len(invalid_idades) > 0:
        sample_idades = invalid_idades['Idade'].head(10).tolist()
        failures.append(f"Idade fora do limite [0, 120]: {len(invalid_idades)} registros. Amostra: {sample_idades}")

    # 4. Validar GastoMensal (positivo e <= 1000)
    null_gastos = df["GastoMensal"].isnull().sum()
    if null_gastos > 0:
        failures.append(f"GastoMensal possui {null_gastos} valores nulos.")
    invalid_gastos = df[(df["GastoMensal"] < 0) | (df["GastoMensal"] > 1000)]
    if len(invalid_gastos) > 0:
        sample_gastos = invalid_gastos['GastoMensal'].head(10).tolist()
        failures.append(f"GastoMensal fora do limite [0, 1000]: {len(invalid_gastos)} registros. Amostra: {sample_gastos}")

    # 5. Validar domínio de TipoContrato
    allowed_contratos = {"Mensal", "Anual", "Bienal"}
    invalid_contratos = set(df["TipoContrato"].dropna().unique()) - allowed_contratos
    if invalid_contratos:
        failures.append(f"TipoContrato possui valores fora do domínio esperado: {list(invalid_contratos)}")

    is_success = len(failures) == 0
    return is_success, failures


def calculate_psi(ref_series, new_series, bins=10):
    """Calcula o Population Stability Index (PSI) entre a série de referência e a nova."""
    ref_arr = np.asarray(ref_series.dropna(), dtype=float)
    new_arr = np.asarray(new_series.dropna(), dtype=float)
    if len(ref_arr) == 0 or len(new_arr) == 0:
        return 0.0
    quantiles = np.linspace(0, 100, bins + 1)
    breakpoints = np.unique(np.percentile(ref_arr, quantiles))
    if len(breakpoints) < 2:
        return 0.0
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf
    ref_counts, _ = np.histogram(ref_arr, bins=breakpoints)
    new_counts, _ = np.histogram(new_arr, bins=breakpoints)
    ref_pct = np.clip(ref_counts / len(ref_arr), 1e-4, None)
    new_pct = np.clip(new_counts / len(new_arr), 1e-4, None)
    psi_val = np.sum((new_pct - ref_pct) * np.log(new_pct / ref_pct))
    return round(float(psi_val), 4)


def detect_data_drift(df_ref, df_new, alpha=0.05):
    """
    Calcula desvio estatístico (Data Drift) entre a base de referência e os novos dados
    utilizando o teste de Kolmogorov-Smirnov (KS-Test) e o Population Stability Index (PSI).
    
    Retorna:
        dict: Resultado dos testes por coluna
    """
    drift_report = {}
    num_cols = ["Idade", "GastoMensal"]
    
    for col in num_cols:
        if col in df_ref.columns and col in df_new.columns:
            ref_clean = df_ref[col].dropna()
            new_clean = df_new[col].dropna()
            
            if len(ref_clean) > 0 and len(new_clean) > 0:
                ks_stat, p_value = ks_2samp(ref_clean, new_clean)
                psi_val = calculate_psi(ref_clean, new_clean)
                has_drift = bool((p_value < alpha) or (psi_val >= 0.25))
                drift_report[col] = {
                    "ks_stat": round(float(ks_stat), 4),
                    "p_value": round(float(p_value), 4),
                    "psi": psi_val,
                    "drift_detected": has_drift
                }
                
    return drift_report


def run_pipeline(data_path, ref_path=None):
    """Executa o pipeline completo de QA e detecção de Drift."""
    print("=" * 60)
    print(f"🛡️  AI Data Quality Guard — Processando: {data_path}")
    print("=" * 60)
    
    if not os.path.exists(data_path):
        print(f"❌ Erro Crítico: Arquivo de entrada '{data_path}' não encontrado.")
        return 1

    df_new = pd.read_csv(data_path)
    print(f"Dataset carregado: {len(df_new)} linhas x {len(df_new.columns)} colunas\n")
    
    # Etapa 1: Asserções de Qualidade
    print("🔍 [1/2] Executando Asserções de Qualidade de Dados...")
    success, failures = validate_data_quality(df_new)
    
    if success:
        print("   ✅ APROVADO: Todos os testes de regras e esquema passaram.")
    else:
        print(f"   ❌ REJEITADO: Encontradas {len(failures)} falhas de qualidade:")
        for fail in failures:
            print(f"      └─ {fail}")
            
    # Etapa 2: Monitoramento de Data Drift
    drift_failed = False
    if ref_path and os.path.exists(ref_path):
        print("\n📊 [2/2] Analisando Data Drift contra Base de Referência...")
        df_ref = pd.read_csv(ref_path)
        drift_report = detect_data_drift(df_ref, df_new)
        
        for col, res in drift_report.items():
            status = "🚨 DRIFT DETECTADO" if res["drift_detected"] else "✅ OK (Sem Drift)"
            print(f"   └─ {col:<15}: KS-Stat = {res['ks_stat']:.4f} | p-valor = {res['p_value']:.4f} -> {status}")
            if res["drift_detected"]:
                drift_failed = True
    else:
        print("\n📊 [2/2] Análise de Drift ignorada (base de referência não fornecida).")
        
    print("\n" + "=" * 60)
    if not success or drift_failed:
        print("🛑 REPROVADO: O pipeline falhou no controle de qualidade.")
        return 1
    else:
        print("🎉 APROVADO: Dados validados e liberados para o modelo!")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Data Quality Guard CLI")
    parser.add_argument("file_path", nargs="?", default="data/novos_dados.csv", help="Caminho do arquivo CSV de entrada")
    parser.add_argument("--ref", default="data/base_referencia.csv", help="Caminho do arquivo CSV de referência para Drift")
    args = parser.parse_args()
    
    exit_code = run_pipeline(args.file_path, args.ref)
    sys.exit(exit_code)
