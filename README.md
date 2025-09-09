# **Predicting Postpartum Depression Risk with Machine Learning**
*A practical look at using data to support early maternal mental health care*

---

## **Why I Took On This Challenge**
Postpartum depression (PPD) affects up to 1 in 5 mothers. In many communities, screening is rare, and stigma delays help. I set out to explore a simple question:
> *Can accessible, non-invasive data flag mothers at risk six months after childbirth?*

The aim isn’t to replace clinicians, but to help them focus where support is most urgently needed.
**Bottom line?** The final model explained **67% of the variance in PPD outcomes** — a promising step for early intervention.

---

## **Where the Data Came From**
This project used a longitudinal maternal health study with **1,203 participants**. I chose to predict the **6-month HAM-D score** (`hamd_6m`) because it was both clinically meaningful and the most complete.
After removing **385 entries missing this target**, I worked with **818 participants**. The dataset included:
* Mental health history
* Social and family support
* Economic status
* Treatment engagement

Columns with **>50% missingness (14 total)** were dropped, and the remaining data was cleaned for modeling.

---

## **How I Broke Down the Problem**
### **Screening and Reducing Features**
Starting with **311 columns**, I cut redundancy within each domain:
* **Numeric features** – filtered using Variance Inflation Factor (VIF) to remove multicollinearity
* **Binary features** – combined within domains using PCA composites to retain signal

This produced **61 core features** — a more manageable set.

### **Selecting the Most Useful Features**
From those 61, I used two complementary paths:
* **LassoCV** for sparse, linear selection
* **RFECV with Ridge** for stable, model-driven selection

This gave:
* **8 features (intersection)** – compact, high-confidence
* **33 features (union)** – broader, richer

A quick classification experiment (predicting PPD severity) showed the **33-feature set performed better**, so I adopted it and used **Stratified K-Fold cross-validation** to keep severity levels balanced.

### **Adding Meaningful Interactions**
I engineered features to reflect combined risk factors, such as:
* `high_stress_home` – low parenting preparedness × poor marital quality
* `treatment_effective` – effect of treatment when employment was stable

### **Modeling Approach**
I tested three models under the same cross-validation setup:
* XGBoost
* LightGBM (with Optuna tuning)
* Lasso (baseline)

### **Performance Metrics**
* **R²** – how much variance is explained
* **RMSE** – average prediction error
* Secondary: **MAE** and **Median AE** for clinical context

---

## **What the Model Showed**
XGBoost gave the best results:
| Model       | R² (↑)    | RMSE (↓)  | MAE (↓)   |
| ----------- | --------- | --------- | --------- |
| **XGBoost** | **0.673** | **4.014** | **3.085** |
| LightGBM    | 0.642     | 4.202     | 3.210     |
| Lasso       | 0.223     | 6.321     | 5.232     |

**Key predictors included:**
1. `recover_perm` – recovery from past depression
2. `employed_mo_baselineXtreat` – treatment amplifying employment’s protective effect
3. `Mental_Health_composite` – baseline psychological burden
4. `recover_never` – marker of chronic vulnerability
5. `discussed` – early dialogue with providers linked to detection

---

## **Why This Work Matters**
PPD isn’t just about screening — it’s about **timely, informed care**.
This model highlights:
* **Resilience factors** (like recovery history) as major protectors
* **Social and treatment interactions** as levers for change

It’s transparent enough to:
* Guide follow-ups
* Shape targeted interventions
* Support maternal health planning in resource-limited settings

---

## **What I’d Do Next**
* Test on **external datasets** for robustness
* Build a **baseline-only model** for earlier prediction
* Add **SHAP explanations** for interpretability
* Examine **bias and fairness** across income and education groups

---

## **How to Explore the Project**
1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/ppd-prediction.git
   cd ppd-prediction
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebook:

   ```bash
   jupyter notebook notebooks/carecode.ipynb
   ```

---

## **Key Takeaway**
With careful feature work and ensemble modeling, it’s possible to flag postpartum depression risk early — not perfectly, but in a way that could **help prevent suffering before it escalates**.

---

## **Acknowledgments**
Developed as part of the **SheCode Africa AI/ML Track** challenge on maternal mental health.
