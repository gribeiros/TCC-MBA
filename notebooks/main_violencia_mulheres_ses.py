# %%
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from itertools import combinations
pd.set_option("display.max_rows", None)

# %% Util Functions

def to_date_format(df: pd.DataFrame)-> pd.DataFrame:
    df['DT_NOTIFIC'] = pd.to_datetime(
        df['DT_NOTIFIC'], dayfirst=True)
    df['DT_NASC'] = pd.to_datetime(
        df['DT_NASC'], dayfirst=True)
    return df


def chi2_function(vars: list[str], df: pd.DataFrame,order_by: str = 'chi2')-> None:
    results = []

    for var1, var2 in combinations(vars, 2):
        tabela = pd.crosstab(df[var1], df[var2])
        
        chi2, p, _, _ = chi2_contingency(tabela)
        results.append({
            'var1': var1,
            'var2': var2,
            'chi2': chi2,
            'association': '✅ Yes' if p < 0.05 else '❌ No'
        })

    # Exibir resultados ordenados por p-valor
    df_results = pd.DataFrame(results).sort_values(by=order_by)
    print(df_results)

# %% Load Data
path = "../data/dados_violencia_mulheres_ses/to_clean"

files = [f for f in os.listdir(path) if f.endswith('.csv')]

dfs = [to_date_format(pd.read_csv(os.path.join(path, file), sep=';'))
       for file in files]


main = pd.concat(dfs, ignore_index=True)

main.info()

# %% Rename colums and create a copy
column_mapping = {
    'DT_NOTIFIC': 'notification_date',
    'DT_NASC': 'birth_date',
    'NU_IDADE_N': 'age',
    'CS_SEXO': 'sex',
    'CS_RACA': 'race',
    'ID_MN_RESI': 'city_residence',
    'LOCAL_OCOR': 'occurrence_place',
    'OUT_VEZES': 'previous_occurrences',
    'LES_AUTOP': 'self_harm',
    'VIOL_FISIC': 'physical_violence',
    'VIOL_PSICO': 'psychological_violence',
    'VIOL_SEXU': 'sexual_violence',
    'NUM_ENVOLV': 'num_perpetrators',
    'AUTOR_SEXO': 'perpetrator_sex',
    'ORIENT_SEX': 'sexual_orientation',
    'IDENT_GEN': 'gender_identity'
}

main = main.rename(columns=column_mapping)
main['age'] = pd.to_numeric(main['age'], errors='coerce').astype('Int64')

df_cleaned = main.copy()

main.head()

# %%
print(main['age'].describe())

main['age'] = pd.to_numeric(main['age'], errors='coerce').astype('Int64')
sns.boxplot(data=main, y='age')
main['age'].unique()
# Remover outliers de 'age' utilizando o método do Intervalo Interquartil (IQR)
Q1 = main['age'].quantile(0.25)
Q3 = main['age'].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df_cleaned = df_cleaned[(df_cleaned['age'] >= lower_bound) & (df_cleaned['age'] <= upper_bound)]
sns.boxplot(data=df_cleaned, y='age')
# %%
# Get value counts and create a bar plot
plt.figure(figsize=(15, 8))
age_counts = df_cleaned['age'].value_counts().head(20)
age_counts.plot(kind='bar')
plt.title('Top 20 Ages by Number of Cases')
plt.xlabel('Age')
plt.ylabel('Number of Cases')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Categoriza idades
def categorize_age(age):
    if age < 18:
        return 'Menor de idade'
    elif age < 60:
        return 'Maior de idade'
    else:
        return 'Idoso'

df_cleaned['age_category'] = df_cleaned['age'].apply(categorize_age)

main['age_category'] = main['age'].apply(categorize_age)

plt.figure(figsize=(10, 6))
sns.countplot(
    data=df_cleaned,
    x='age_category',
    order=df_cleaned['age_category'].value_counts().index,
    color='teal'
)
plt.title('Distribuição por Categoria de Idade')
plt.xlabel('Categoria')
plt.ylabel('Quantidade')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

main['notification_date'].dt.year.value_counts().plot(kind='bar', title='Distribuição por Ano de Notificação')

main['notification_date'].dt.month.value_counts().plot(kind='bar', title='Distribuição por Ano de Notificação')
main.drop(columns=['notification_date', 'birth_date', 'age']).describe()
vars = main.select_dtypes(include=['object']).columns.tolist()

filtered_vars = [v for v in vars if main[v].nunique() > 1]
chi2_function(filtered_vars, main)
main['sex'].value_counts().plot(kind='bar', title='Distribuição por Sexo')
# Por possui somente uma categoria, e não passar no teste de qui-quadrado se faz necessário remover
df_cleaned.drop(columns=['sex'],inplace=True)

main['race'].value_counts().plot(kind='bar', title='Distribuição por Raça')

# Get value counts and create a bar plot
plt.figure(figsize=(15, 8))
city_counts = main['city_residence'].value_counts().head(20)
city_counts.plot(kind='bar')
plt.title('Top 20 Cities by Number of Cases')
plt.xlabel('City')
plt.ylabel('Number of Cases')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
city_categories = main['city_residence'].dropna().unique()
print(f"Total of cities: {len(city_categories)}")
main['gender_identity'].value_counts().plot(kind='bar', title='Distribuição por Identidade de Gênero')
main['occurrence_place'].value_counts().plot(kind='bar', title='Distribuição por Local de Ocorrência')
main['previous_occurrences'].value_counts()
plt.figure(figsize=(10, 6))
plt.pie(main['previous_occurrences'].value_counts(),
        labels=main['previous_occurrences'].value_counts().index,
        autopct='%1.1f%%',
        startangle=140)
plt.title('Distribuição de Ocorrências Anteriores')
plt.show()

plt.figure(figsize=(10, 6))
plt.pie(main['self_harm'].value_counts(),
        labels=main['self_harm'].value_counts().index,
        autopct='%1.1f%%',
        startangle=140)
plt.title('Distribuição de Autolesão')
plt.show()
plt.figure(figsize=(10, 6))
plt.pie(main['physical_violence'].value_counts(),
        labels=main['physical_violence'].value_counts().index,
        autopct='%1.1f%%',
        startangle=140)
plt.title('Distribuição de Violência Física')
plt.show()
plt.figure(figsize=(10, 6))
plt.pie(main['psychological_violence'].value_counts(),
        labels=main['psychological_violence'].value_counts().index,
        autopct='%1.1f%%',
        startangle=140)
plt.title('Distribuição de Violência Psicológica')
plt.show()
plt.figure(figsize=(10, 6))
plt.pie(main['sexual_violence'].value_counts(),
        labels=main['sexual_violence'].value_counts().index,
        autopct='%1.1f%%',
        startangle=140)
plt.title('Distribuição de Violência Sexual')
plt.show()
plt.figure(figsize=(10, 6))
plt.pie(main['num_perpetrators'].value_counts(),
        labels=main['num_perpetrators'].value_counts().index,
        autopct='%1.1f%%',
        startangle=140)
plt.title('Distribuição do Número de Agressões')
plt.show()
plt.figure(figsize=(10, 6))
plt.pie(main['perpetrator_sex'].value_counts(),
        labels=main['perpetrator_sex'].value_counts().index,
        autopct='%1.1f%%',
        startangle=140)
plt.title('Distribuição do Sexo do Agressor')
plt.show()
sns.countplot(data=main, x='sexual_orientation')
plt.xticks(rotation=45, ha='right')
plt.show()
df_cleaned.to_csv('../data/dados_violencia_mulheres_ses/cleaned/cleaned_data.csv', index=False)