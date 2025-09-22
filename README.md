# Violence Against Women Analysis

This repository contains the material used in my MBA in Data Science and Analytics final thesis. The goal is to explore public datasets from the state of Minas Gerais (Brazil) related to violence against women.

## Project structure

- `notebooks/` – Jupyter notebooks with the exploratory analysis.
- `src/` – Python modules with helper functions used in the notebooks.
- `data/` – Raw CSV files and intermediate processed datasets.
- `docs/` – Additional documentation such as the project overview.

## Data sources

The CSV files under the `data/raw/` directory were downloaded from open data portals of the Government of Minas Gerais:

- `notifications_ses/` – notifications of violence against women from the state health secretariat (SES-MG).
- `domestic_violence/` – domestic violence police reports.
- `feminicide/` – feminicide records.

Each subfolder contains yearly CSV files in Portuguese separated by semicolons. Cleaned datasets generated from the notebooks are saved under `data/processed/`.

## Notebooks

Two Jupyter notebooks are provided in the `notebooks/` directory:

- **`main_violencia_domestica.ipynb`** – cleaning and plots for the domestic violence reports.
- **`main_violencia_mulheres_ses.ipynb`** – analysis of notifications from the SES dataset including chi-square tests.

They were created with Python 3.12 kernels and rely on common scientific libraries such as pandas and seaborn.

## Reproducing the analysis

1. Install Python 3.11 or newer.
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter and open the desired notebook:
   ```bash
   jupyter notebook notebooks/
   ```
4. Execute the cells in order. The notebooks expect the CSV files to be present in the `data/raw/` and `data/processed/` directories.

For a short summary of the project, see [docs/project_overview.md](docs/project_overview.md).

## Requirements

See `requirements.txt` for the package versions. The notebooks were tested with pandas 2, numpy 1.26, matplotlib 3.8, seaborn 0.13 and scipy 1.11.

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.
