# 🛡️ AI Data Quality Guard: MLOps & QA Pipeline

Este projeto é um pipeline profissional de **MLOps & Data Quality** projetado para garantir que os dados alimentados em modelos de Machine Learning sejam consistentes, confiáveis e livres de anomalias ou desvios estatísticos (*Data Drift*).

O projeto atua como um **Gatekeeper de QA** integrado à esteira de CI/CD (GitHub Actions), impedindo que dados corrompidos ou com desvio estatístico cheguem às etapas de inferência ou retreinamento em produção.

---

## 🎯 Funcionalidades Principais

1. **Asserção de Esquema e Regras de Negócio:**
   - Validação de presença de colunas obrigatórias (`ID_Cliente`, `Idade`, `GastoMensal`, `TipoContrato`).
   - Verificação de ausência de nulos em identificadores e variáveis críticas.
   - Controle de intervalo estatístico (ex: `Idade` $\in [0, 120]$, `GastoMensal` $\in [0, 1000]$).
   - Validação de domínio de variáveis categóricas (`TipoContrato` $\in \{\text{"Mensal"}, \text{"Anual"}, \text{"Bienal"}\}$).

2. **Monitoramento de Data Drift (KS-Test):**
   - Comparação estatística entre a base de referência (`base_referencia.csv`) e novos dados de produção (`novos_dados.csv`).
   - Aplicação do teste **Kolmogorov-Smirnov (KS-Test)** com nível de significância $\alpha = 0.05$.

3. **Suíte de Testes Automáticos (`pytest`):**
   - Testes unitários cobrindo cenários com dados válidos, dados corrompidos e ausência de drift.

4. **Automação CI/CD (GitHub Actions):**
   - Execução automática dos testes unitários e das validações de qualidade a cada `push` ou `pull_request`.

---

## 📁 Estrutura do Repositório

```text
ai-data-quality-guard/
├── .github/
│   └── workflows/
│       └── data_quality.yml        # CI/CD no GitHub Actions
├── data/
│   ├── base_referencia.csv         # Dados históricos de referência
│   └── novos_dados.csv             # Novos dados de produção para validar
├── tests/
│   └── test_pipeline.py            # Suíte de testes unitários com pytest
├── data_quality_pipeline.py        # Pipeline principal de QA e Drift
├── requirements.txt                # Dependências com versões fixadas
└── README.md                       # Documentação técnica do repositório
```

---

## 🚀 Como Executar Localmente

### 1. Instalar Dependências
```bash
pip install -r requirements.txt
```

### 2. Rodar a Suíte de Testes Unitários
```bash
python -m pytest
```

### 3. Rodar a Validação nos Dados de Referência (Aprovados)
```bash
python data_quality_pipeline.py data/base_referencia.csv --ref data/base_referencia.csv
```

### 4. Rodar a Validação em Novos Dados com Drift/Falhas
```bash
python data_quality_pipeline.py data/novos_dados.csv --ref data/base_referencia.csv
```
