import sys
import os
import pandas as pd
import great_expectations as ge

def run_data_validation(file_path):
    print(f"=== Iniciando Validação de Qualidade para: {file_path} ===")
    
    if not os.path.exists(file_path):
        print(f"Erro: Arquivo {file_path} não encontrado.")
        sys.exit(1)

    # Carregar dataframe
    df = pd.read_csv(file_path)
    
    # Criar um PandasDataset do Great Expectations
    ge_df = ge.from_pandas(df)
    
    validation_results = []
    
    # 1. Validar se colunas críticas existem
    cols = ["ID_Cliente", "Idade", "GastoMensal", "TipoContrato"]
    for col in cols:
        res = ge_df.expect_column_to_exist(col)
        validation_results.append(res)
        
    # 2. Validar que ID_Cliente não tem valores nulos
    res = ge_df.expect_column_values_to_not_be_null("ID_Cliente")
    validation_results.append(res)
    
    # 3. Validar que Idade não tem nulos e está em faixa aceitável (0 a 120 anos)
    res = ge_df.expect_column_values_to_not_be_null("Idade")
    validation_results.append(res)
    res = ge_df.expect_column_values_to_be_between("Idade", min_value=0, max_value=120)
    validation_results.append(res)
    
    # 4. Validar que GastoMensal é positivo
    res = ge_df.expect_column_values_to_be_between("GastoMensal", min_value=0, max_value=1000)
    validation_results.append(res)
    
    # 5. Validar que TipoContrato pertence aos valores de domínio esperados
    res = ge_df.expect_column_values_to_be_in_set("TipoContrato", ["Mensal", "Anual", "Bienal"])
    validation_results.append(res)
    
    # Contar falhas
    failures = 0
    print("\nResultados dos Testes:")
    for res in validation_results:
        success = res["success"]
        expectation = res["expectation_config"]["expectation_type"]
        kwargs = res["expectation_config"]["kwargs"]
        
        if success:
            print(f" ✅ PASSOU: {expectation} {kwargs}")
        else:
            failures += 1
            print(f" ❌ FALHOU: {expectation} {kwargs}")
            # Mostrar resumo do erro
            result = res.get("result", {})
            unexpected_count = result.get("unexpected_count", 0)
            unexpected_values = result.get("unexpected_values", [])
            print(f"    └─ Registros inesperados: {unexpected_count} | Valores: {unexpected_values}")

    print(f"\nTotal de falhas encontradas: {failures}")
    if failures > 0:
        print("🛑 O Pipeline de dados falhou devido a problemas de qualidade dos dados!")
        sys.exit(1)
    else:
        print("🎉 Todos os testes de qualidade passaram! Dados aprovados para o Modelo.")
        sys.exit(0)

if __name__ == "__main__":
    # Por padrão, vamos tentar rodar com os novos dados que contêm falhas.
    # Mas o usuário pode passar o caminho de arquivo como argumento.
    target_file = sys.argv[1] if len(sys.argv) > 1 else r"data/novos_dados.csv"
    run_data_validation(target_file)
