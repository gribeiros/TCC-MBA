"""Analysis of notifications of violence against women from SES-MG.

This script loads the processed dataset, summarises variables and generates
exploratory visualisations. The final cleaned dataset is saved under
``data/processed/cleaned_data.csv``.
"""

from __future__ import annotations

import math
from pathlib import Path
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency, shapiro

pd.set_option("display.max_rows", None)

# ---------------------------------------------------------------------------
# 1. Introdução/Contexto
# ---------------------------------------------------------------------------
# Este script analisa notificações de violência contra mulheres reportadas ao
# Sistema Único de Saúde de Minas Gerais (SES-MG). O conjunto de dados
# ``notifications_ses.csv`` foi previamente limpo e está disponível em
# ``data/processed``.

VARIABLE_INFO = {
    "notification_date": "Data em que o caso foi notificado",
    "birth_date": "Data de nascimento da vítima",
    "age": "Idade da vítima em anos",
    "race": "Raça/Cor auto declarada",
    "city_residence": "Município de residência",
    "occurrence_place": "Local onde ocorreu o fato",
    "previous_occurrences": "Indica ocorrência prévia de violência",
    "self_harm": "Registro de autolesão",
    "physical_violence": "Violência física",
    "psychological_violence": "Violência psicológica",
    "sexual_violence": "Violência sexual",
    "num_perpetrators": "Número de agressores",
    "perpetrator_sex": "Sexo do(s) agressor(es)",
    "sexual_orientation": "Orientação sexual da vítima",
    "gender_identity": "Identidade de gênero da vítima",
    "age_category": "Faixa etária categorizada",
}

# ---------------------------------------------------------------------------
# 2. Carregamento e revisão de dados
# ---------------------------------------------------------------------------

def load_dataset() -> pd.DataFrame:
    """Load processed CSV file with proper date parsing."""
    path = Path(__file__).resolve().parents[1] / "data" / "processed" / "notifications_ses.csv"
    df = pd.read_csv(path, parse_dates=["notification_date", "birth_date"])
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    return df


def variable_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a descriptive summary for each column of ``df``."""
    records: list[dict[str, object]] = []
    for col in df.columns:
        dtype = df[col].dtype
        desc = VARIABLE_INFO.get(col, "-")
        missing = df[col].isna().sum()
        if pd.api.types.is_numeric_dtype(dtype):
            series = df[col].dropna()
            stats = series.describe()
            sample = series.sample(n=min(len(series), 5000), random_state=42)
            # Shapiro-Wilk (n <= 5000)
            if len(sample) > 3:
                shapiro_p = shapiro(sample).pvalue
            else:
                shapiro_p = math.nan
            records.append(
                {
                    "variable": col,
                    "description": desc,
                    "type": "numeric",
                    "mean": stats.get("mean"),
                    "median": series.median(),
                    "std": stats.get("std"),
                    "missing": missing,
                    "shapiro_p": shapiro_p,
                }
            )
        else:
            counts = df[col].value_counts()
            proportion = counts.iloc[0] / len(df) if not counts.empty else math.nan
            records.append(
                {
                    "variable": col,
                    "description": desc,
                    "type": "category",
                    "unique": df[col].nunique(),
                    "top": counts.index[0] if not counts.empty else None,
                    "top_prop": proportion,
                    "missing": missing,
                }
            )
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# 3. Funções auxiliares
# ---------------------------------------------------------------------------

def categorize_age(age: float | int | None) -> str:
    if pd.isna(age):
        return "Ignorado"
    if age < 18:
        return "Menor de idade"
    if age < 60:
        return "Maior de idade"
    return "Idoso"


def remove_age_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outliers da idade utilizando IQR."""
    q1 = df["age"].quantile(0.25)
    q3 = df["age"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return df[(df["age"] >= lower) & (df["age"] <= upper)]


def chi2_associations(vars: list[str], df: pd.DataFrame) -> pd.DataFrame:
    """Calculate chi-square association for all variable pairs."""
    results = []
    for var1, var2 in combinations(vars, 2):
        table = pd.crosstab(df[var1], df[var2])
        chi2, p, _, _ = chi2_contingency(table)
        results.append({"var1": var1, "var2": var2, "chi2": chi2, "p_value": p})
    return pd.DataFrame(results).sort_values("p_value")


# ---------------------------------------------------------------------------
# 4. Visualizações
# ---------------------------------------------------------------------------

def plot_age_distribution(df: pd.DataFrame) -> None:
    sns.boxplot(x=df["age"])
    plt.title("Distribuição de Idades")
    plt.tight_layout()
    plt.show()

    plt.figure()
    sns.histplot(df["age"], kde=True)
    plt.title("Histograma de Idades")
    plt.tight_layout()
    plt.show()


def plot_age_category(df: pd.DataFrame) -> None:
    plt.figure()
    sns.countplot(
        data=df,
        x="age_category",
        order=df["age_category"].value_counts().index,
        color="teal",
    )
    plt.title("Distribuição por Categoria de Idade")
    plt.xlabel("Categoria")
    plt.ylabel("Quantidade")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_time_series(df: pd.DataFrame) -> None:
    by_year = df["notification_date"].dt.year.value_counts().sort_index()
    by_year.plot(kind="bar")
    plt.title("Distribuição por Ano de Notificação")
    plt.xlabel("Ano")
    plt.ylabel("Número de casos")
    plt.tight_layout()
    plt.show()

    plt.figure()
    by_month = df["notification_date"].dt.month.value_counts().sort_index()
    by_month.plot(kind="bar")
    plt.title("Distribuição por Mês de Notificação")
    plt.xlabel("Mês")
    plt.ylabel("Número de casos")
    plt.tight_layout()
    plt.show()


def plot_cities(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    df["city_residence"].value_counts().head(20).plot(kind="bar")
    plt.title("Top 20 Municípios por Número de Casos")
    plt.xlabel("Município")
    plt.ylabel("Número de casos")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_race(df: pd.DataFrame) -> None:
    plt.figure()
    df["race"].value_counts().plot(kind="bar")
    plt.title("Distribuição por Raça/Cor")
    plt.xlabel("Raça/Cor")
    plt.ylabel("Número de casos")
    plt.tight_layout()
    plt.show()


def plot_correlation(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 8))
    corr = df.select_dtypes(include=[np.number]).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlação entre Variáveis Numéricas")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 5. Pipeline principal
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_dataset()
    print("Dataset shape:", df.shape)
    print(variable_summary(df))

    df_clean = remove_age_outliers(df)
    df_clean["age_category"] = df_clean["age"].apply(categorize_age)

    plot_age_distribution(df_clean)
    plot_age_category(df_clean)
    plot_time_series(df_clean)
    plot_cities(df_clean)
    plot_race(df_clean)
    plot_correlation(df_clean)

    cat_vars = [c for c in df_clean.columns if df_clean[c].dtype == "object"]
    cat_results = chi2_associations(cat_vars, df_clean)
    print(cat_results)

    output = Path(__file__).resolve().parents[1] / "data" / "processed" / "cleaned_data.csv"
    df_clean.to_csv(output, index=False)


if __name__ == "__main__":
    main()
