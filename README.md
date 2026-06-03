# 🛡️ AI Data Quality Guard: MLOps & QA Pipeline

Este projeto é um pipeline prático de **MLOps & Data Quality** projetado para garantir que os dados alimentados em modelos de Machine Learning sejam consistentes, confiáveis e livres de anomalias. 

O projeto preenche uma lacuna crucial em produção: **a qualidade de dados antes da inferência ou do retreinamento**, atuando como um "Gatekeeper" de QA integrado à esteira de CI/CD (GitHub Actions).

---

## 🎯 Por que este projeto é relevante?
Modelos de Machine Learning sofrem com o princípio de *Garbage In, Garbage Out* (Entrada de lixo gera saída de lixo). Se o comportamento dos dados de produção mudar (Data Drift) ou se os dados vierem com formatos errados, nulos inesperados ou fora de intervalo:
1. O modelo fará predições erradas sem dar erro explícito no código.
2. Decisões de negócio serão tomadas com base em predições falhas.

O **AI Data Quality Guard** automatiza os testes de assertiva de dados antes do processamento.

---

## 🛠️ Tecnologias Utilizadas
*   **Linguagem:** Python
*   **Data Quality Framework:** **Great Expectations** (para testes de asserção de dados)
*   **Análise de Drift:** **Evidently AI** (para monitoramento de desvio de dados)
*   **Pipeline & Orquestração:** Pandas
*   **CI/CD:** **GitHub Actions** (para execução automatizada nas Pull Requests)

---

## 📁 Estrutura do Projeto
```text
ai-data-quality-guard/
├── .github/
│   └── workflows/
│       └── data_quality.yml        # CI/CD no GitHub Actions
├── data/
│   ├── base_referencia.csv         # Dados históricos de referência
│   └── novos_dados.csv             # Novos dados de produção para validar
├── data_quality_pipeline.py        # Script principal que executa a validação
├── requirements.txt                # Dependências do projeto
└── README.md                       # Documentação técnica
```

---

## ⚙️ Como Funciona o Pipeline

1.  **Carregamento de Dados:** O script lê a base de dados atual.
2.  **Validação por Regras (Great Expectations):** O pipeline aplica testes configurados:
    *   Verifica se não há valores nulos em colunas críticas.
    *   Confirma se tipos de dados estão corretos (ex: inteiros, decimais).
    *   Verifica se valores estão dentro de intervalos estatísticos esperados (ex: idade entre 0 e 120, valores de cobrança positivos).
    *   Garante que colunas essenciais estão presentes no DataFrame.
3.  **Relatório e Alerta:**
    *   Se todos os testes passarem, o pipeline prossegue para a inferência/treino do modelo.
    *   Se algum teste falhar, o script encerra com código de saída de erro (`sys.exit(1)`), forçando a esteira de CI/CD do GitHub Actions a falhar e alertando a equipe de engenharia de dados.

---

## 🚀 Como Executar Localmente

### 1. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 2. Rodar o Pipeline de Validação
```bash
python data_quality_pipeline.py
```

---

*Desenvolvido por Vanessa de Carvalho Faria.*
