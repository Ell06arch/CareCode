# **Predicting Postpartum Depression Risk with Machine Learning**

*A data-driven approach to early maternal mental health support*

---

## **1. Why This Project?**

Postpartum depression (PPD) affects up to 1 in 5 mothers, yet in many communities, it goes undetected due to limited screening and stigma. Early identification is critical — the sooner support is offered, the better the outcomes.

This project asks a practical question:
> **Can we use accessible, non-invasive data to predict PPD risk six months after childbirth?**

The goal isn’t to replace clinicians, but to **support them with timely insights** — especially in resource-constrained settings.

---

## **2. The Data Behind It**

The dataset includes **1,203 participants** from a longitudinal maternal health study. To ensure a reliable outcome, **385 entries without 6-month HAM-D scores were excluded**, leaving **818 records** for modeling.

Each record contained hundreds of features across domains like:
- Mental health history
- Social and family support
- Economic status
- Treatment engagement

After cleaning, **14 columns with >50% missingness were removed**, and the remaining data was rigorously preprocessed for modeling.

---

## **3. How We Approached the Problem**

### 1. **Feature Engineering & Discovery**
- Conducted deep EDA to uncover patterns in resilience and risk.
- Engineered **clinically meaningful features** like:
  - `high_stress_home`: Low parenting preparedness + poor marital quality
  - `treatment_effective`: Composite of key interaction effects
  - `recover_perm`: Indicator of full recovery from past depression

### 2. **Feature Selection**
- Used a **dual-path strategy**:
  - **LassoCV**: For automatic sparsity
  - **RFECV with Ridge**: For stable, model-driven selection
- Final set: **33 features** — a balance of signal, interpretability, and generalizability

### 3. **Modeling**
- Tested **XGBoost, LightGBM, and Lasso** on the same CV framework
- Used **Stratified K-Fold CV** (5 splits) on binned `hamd_6m` to ensure balanced representation of depression severity

### 4. **Evaluation**
- Primary metrics: **R²** (explained variance), **RMSE** (prediction error)
- Secondary: **MAE, MedAE** for clinical interpretability

---

## **4. What We Found**

XGBoost outperformed all models:

| Model        | R² (↑) | RMSE (↓) | MAE (↓) |
| ------------ | ------ | -------- | ------- |
| **XGBoost**  | **0.673** | **4.014** | **3.085** |
| LightGBM     | 0.642  | 4.202    | 3.210   |
| Lasso        | 0.223  | 6.321    | 5.232   |

This means the final model **explains 67.3% of the variance** in PPD — a **strong result** for a behavioral health outcome influenced by complex, overlapping factors.

### **Top Predictors**
1. `recover_perm` — Recovery from past depression
2. `employed_mo_baselineXtreat` — Treatment amplifies employment’s protective effect
3. `Mental_Health_composite` — Baseline psychological burden
4. `recover_never` — Major risk factor — indicates chronic vulnerability
5. `discussed` — Openness to dialogue may reflect awareness and early help-seeking

---

## **5. Why It Matters**

This model doesn’t just predict — it **reveals**.

It shows that:
- **Resilience** (e.g., past recovery) is the strongest protective factor
- **Social support** and **treatment interactions** are modifiable levers

Rather than a black box, this is a **transparent decision-support tool** that can:
- Help health workers **prioritize follow-ups**
- Guide **targeted interventions** for high-risk mothers
- Inform **policy and program design** in maternal mental health

---

## **6. What’s Next**

- **Test on external datasets** to assess generalizability
- **Develop a baseline-only model** for earlier prediction (before 6-month data)
- **Add SHAP explanations** for real-time interpretability in clinical settings
- Explore **fairness and bias** across subgroups (e.g., education, income)

---

## **7. How to Explore This Project**

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ppd-prediction.git
   cd ppd-prediction
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook:

   ```bash
   jupyter notebook notebooks/carecode.ipynb
   ```

---

## **8. Key Takeaway**

This project demonstrates that with the right data and careful modeling, we can move closer to **timely, data-driven maternal mental health support** — one prediction at a time.

---

## **Acknowledgments**

This work was developed as part of a maternal mental health data science challenge. Dataset provided by **SheCode Africa AI/ML Track**.
