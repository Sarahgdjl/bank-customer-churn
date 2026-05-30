# Bank Customer Churn Prediction

An end-to-end data science project predicting which bank customers are likely to churn — from raw data exploration through SQL data warehouse design, machine learning, and an interactive dashboard.

**Live dashboard →** [yourusername.github.io/bank-customer-churn](https://yourusername.github.io/bank-customer-churn)

---

## Results at a glance

| Model | ROC-AUC | F1 Score | CV-AUC |
|---|---|---|---|
| **XGBoost** ★ | **0.8688** | **0.6230** | **0.8525** |
| LightGBM | 0.8669 | 0.6168 | 0.8513 |
| Logistic Regression | 0.8500 | 0.5776 | 0.8342 |

---

## Project structure

```
bank-customer-churn/
├── index.html                        ← Interactive dashboard (GitHub Pages)
├── README.md
│
├── data/
│   └── Customer-Churn-Records.csv    ← Source dataset (10,000 rows, 18 cols)
│
├── notebooks/
│   ├── eda.ipynb                     ← Phase 1: Exploratory data analysis
│   └── phase3_modeling.ipynb         ← Phase 3: Model training & evaluation
│
├── sql/
│   └── phase2_mysql.sql              ← Phase 2: 3-layer MySQL data warehouse
│
├── outputs/
│   ├── plot_01_distributions.png
│   ├── plot_02_categorical_churn.png
│   ├── plot_03_age_analysis.png
│   ├── plot_04_balance_analysis.png
│   ├── plot_05_correlation_heatmap.png
│   ├── plot_06_interaction_heatmaps.png
│   ├── plot_07_boxplots.png
│   ├── plot_08_satisfaction.png
│   ├── plot_p3_01_model_comparison.png
│   ├── plot_p3_02_confusion_matrices.png
│   ├── plot_p3_03_feature_importance.png
│   └── plot_p3_04_shap.png
│
└── reports/
    └── executive_summary.pdf         ← Phase 5: Business report
```

---

## Five phases

### Phase 1 — EDA & Feature Engineering
Deep exploratory analysis covering distributions, churn rates by segment, correlation heatmap, interaction effects, and leakage investigation on the `Complain` column.

Key findings:
- **Germany** churns at 32.4% — nearly 2× France (16.2%) and Spain (16.7%)
- **Ages 48–58** show 55.3% churn — the highest-risk life stage
- **Products 3 & 4** have 82.7% and 100% churn respectively
- **Zero balance** customers churn less (13.8%) than positive balance (24.1%)
- `Complain` identified as a leakage feature — excluded from all modelling

### Phase 2 — SQL Data Warehouse
Three-layer MySQL schema built and loaded via XAMPP/phpMyAdmin:

- `staging_raw` — raw CSV ingest, zero transformations
- `customers_cleaned` — validated, typed, deduped; `Complain` quarantined
- `customer_features` — 30 engineered features mirroring the ML pipeline

Includes 7 analytical views: `v_model_features`, `v_churn_by_geo_gender`, `v_churn_by_age_band`, `v_churn_by_balance`, `v_engagement_segments`, `v_complain_leakage_audit`, `v_load_quality`.

### Phase 3 — Machine Learning
Trained three models on an 80/20 stratified split with 30 features. Class imbalance (80/20) handled via `class_weight='balanced'` and `scale_pos_weight`. SHAP TreeExplainer used for feature explainability.

```python
# Install dependencies
pip install xgboost lightgbm shap scikit-learn pandas numpy matplotlib seaborn
```

### Phase 4 — Dashboard
Interactive HTML/CSS/JS dashboard with:
- Animated KPI cards
- Churn rate bar charts by geography and product count
- Gender donut chart
- Age-band column chart
- Geography × Gender interaction heatmap
- Model performance comparison panel
- Key insights sidebar

No frameworks required — opens directly in any browser.

### Phase 5 — Report & GitHub
Executive summary PDF (4 pages), model card, and this README.

---

## Key features engineered

| Group | Features | Rationale |
|---|---|---|
| Age | `age_band`, `is_peak_churn_age` | 48–58 band = 55.3% churn |
| Balance | `has_zero_balance`, `log_balance`, `balance_to_salary_ratio` | Zero balance = lower churn; log for skew |
| Products | `is_high_risk_products` (≥3), `is_single_product` | Products 3 & 4 near-100% churn |
| Tenure | `is_new_customer`, `products_per_year` | New customers differ in risk |
| Engagement | `engagement_score` (0–3), `is_disengaged` | Activity + card + satisfaction composite |
| Geography | `geography_risk_score`, `is_high_churn_geography` | Germany=3, Spain=2, France=1 |
| Interactions | `is_german_female`, `is_inactive_high_products` | Joint effects from EDA heatmaps |

---

## Leakage note

The `Complain` column was flagged early in EDA as a potential leakage feature — complaints recorded at the same time as churn would give the model information it wouldn't have in a real prediction scenario. It is preserved in the SQL `customers_cleaned` table for post-hoc business analysis but is **excluded from all model training and the `v_model_features` view**.

---

## How to reproduce

**EDA & modelling:**
```bash
git clone https://github.com/yourusername/bank-customer-churn
cd bank-customer-churn
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm shap sqlalchemy mysql-connector-python
jupyter notebook
```

**SQL warehouse:**
1. Open XAMPP → start MySQL
2. Open phpMyAdmin → create database `churn-warehouse`
3. Paste and run `sql/phase2_mysql.sql`
4. Run the data load cell in `notebooks/phase3_modeling.ipynb`

**Dashboard:**
Open `index.html` in any browser — no server needed.

---

## Tech stack

`Python 3.x` · `pandas` · `numpy` · `scikit-learn` · `XGBoost` · `LightGBM` · `SHAP` · `matplotlib` · `seaborn` · `MySQL` · `SQLAlchemy` · `HTML/CSS/JS` · `Jupyter`

---

## Dataset

[Bank Customer Churn Dataset](https://www.kaggle.com/) — 10,000 rows, 18 columns, zero missing values. Synthetic retail banking dataset covering customers in France, Germany, and Spain.

---

*Built as a full end-to-end portfolio project covering data engineering, EDA, machine learning, and business communication.*
