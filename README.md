# Análise de Violência contra Mulheres em MG

Repositório do Trabalho de Conclusão (MBA em Data Science & Analytics). O projeto explora dados públicos de Minas Gerais sobre violência contra mulheres, realizando leitura, limpeza, análises exploratórias e séries temporais para identificar padrões e apoiar discussões sobre políticas públicas.

## Objetivos
- Consolidar bases anuais do Governo de Minas Gerais em um único conjunto tratado.
- Explorar o perfil das vítimas e das ocorrências (idade, raça/cor, cidade, local da ocorrência, reincidência).
- Analisar diferenças entre tipos de violência e relações com variáveis categóricas (ex.: teste do qui-quadrado).
- Gerar séries temporais por tipo de violência para observar tendências.

## Estrutura do repositório
- `notebooks/`
  - `data_exploration.ipynb`: leitura dos dados brutos, limpeza/tratamento, análises exploratórias, testes estatísticos e séries temporais por tipo de violência.
  - `read_files.ipynb`: leitura e inspeção inicial dos CSVs brutos (verificação de colunas, formatos e codificação).
- `data/`
  - `raw/`: CSVs originais (separador `;`) obtidos de portais públicos de MG. Ex.: `data/raw/dados_violencia_mulheres_ses_2010.csv`.
  - `processed/`: saídas geradas pelos notebooks, como `data/processed/violencia_mulheres_ses.csv`.
- `requirements.txt`: dependências necessárias para rodar os notebooks.

## Fontes de dados
CSVs anuais disponibilizados por órgãos públicos de Minas Gerais. Os arquivos usam ponto e vírgula como separador e incluem variáveis demográficas, de local/ocorrência e classificação do tipo de violência. Não há dados sensíveis individualizados; todas as análises são feitas em nível agregado.

## Ambiente e instalação
1. Tenha Python 3.11+ instalado.
2. Crie e ative um ambiente virtual (recomendado):
   ```bash
   python -m venv .venv
   .\\.venv\\Scripts\\activate  # Windows
   ```
3. Instale dependências:
   ```bash
   pip install -r requirements.txt
   ```

## Como reproduzir
1. Coloque os CSVs originais em `data/raw/` (não versionados).
2. Abra o Jupyter na pasta de notebooks:
   ```bash
   jupyter notebook notebooks/
   ```
3. Execute `read_files.ipynb` para validar estrutura e encoding dos CSVs.
4. Execute `data_exploration.ipynb` na ordem das células:
   - limpeza/normalização (separadores, tipos e valores ausentes);
   - análise exploratória (distribuições numéricas, contingência para variáveis categóricas);
   - séries temporais por tipo de violência (física, psicológica, sexual, autolesão).
5. Os arquivos tratados são gravados em `data/processed/` e podem ser reutilizados em outras etapas ou estudos.

## Principais saídas
- Conjunto consolidado `data/processed/violencia_mulheres_ses.csv`.
- Gráficos de distribuição para variáveis numéricas e categóricas.
- Resultados de testes de independência entre variáveis categóricas.
- Séries temporais por tipo de violência para visualização de tendências.

## Observações e limites
- Dados públicos podem ter lacunas de cobertura ou diferenças de coleta entre anos.
- Ajuste caminhos ou separadores caso novos CSVs venham em formatos distintos.
- O projeto foca reprodutibilidade acadêmica; não inclui modelagem preditiva ou dashboards interativos.
