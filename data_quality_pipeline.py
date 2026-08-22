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
        failures.append(f"Idade fora do limite [0, 120]: {len(invalid_idades)} registros. Valores: {invalid_idades['Idade'].tolist()}")

    # 4. Validar GastoMensal (positivo e <= 1000)
    null_gastos = df["GastoMensal"].isnull().sum()
    if null_gastos > 0:
        failures.append(f"GastoMensal possui {null_gastos} valores nulos.")
    invalid_gastos = df[(df["GastoMensal"] < 0) | (df["GastoMensal"] > 1000)]
    if len(invalid_gastos) > 0:
        failures.append(f"GastoMensal fora do limite [0, 1000]: {len(invalid_gastos)} registros. Valores: {invalid_gastos['GastoMensal'].tolist()}")

    # 5. Validar domínio de TipoContrato
    allowed_contratos = {"Mensal", "Anual", "Bienal"}
    invalid_contratos = set(df["TipoContrato"].dropna().unique()) - allowed_contratos
    if invalid_contratos:
        failures.append(f"TipoContrato possui valores fora do domínio esperado: {list(invalid_contratos)}")

    is_success = len(failures) == 0
    return is_success, failures


def detect_data_drift(df_ref, df_new, alpha=0.05):
    """
    Calcula desvio estatístico (Data Drift) entre a base de referência e os novos dados
    utilizando o teste de Kolmogorov-Smirnov (KS-Test) para numéricas.
    
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
                has_drift = p_value < alpha
                drift_report[col] = {
                    "ks_stat": round(float(ks_stat), 4),
                    "p_value": round(float(p_value), 4),
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
