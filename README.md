# **Predicting Postpartum Depression Risk with Machine Learning**
Postpartum depression (PPD) affects about 1 in 7 mothers globally, yet in many settings, fewer than half of cases are detected early due to limited screening and persistent stigma. I set out to explore a simple question:

> *Can routinely collected maternal health and social factors predict who is likely to develop PPD six months after childbirth?*

This work looks at how early patterns in mothers' health and environment could guide timely follow-up and support not to replace care but to make it more proactive.

The model captured about **67% of the key differences between mothers who did and didn’t develop PPD**, showing it can meaningfully flag risk early based on the data available.

*This project was developed as part of the **SheCode Africa AI/ML Challenge (Aug–Sep 2025)**.*

---

## **Where the Data Came From**
The dataset was provided as part of the competition — a longitudinal maternal health study with **1,203 participants**. The task was framed as a **regression problem**, so I used the **6-month HAM-D score** (`hamd_6m`) as the target because it’s a standard measure of depression severity and had the most complete data for this period.

### **Schema Alignment**
A schema was provided describing what each column represented. I cross-checked all dataset columns against this schema to ensure interpretability.
* After this step, the feature set dropped from **394 (including the target)** to **236 (including the target)**.
* Columns with over 50% missingness (14 in total) were then removed, leaving **222 (including the target)**.

After removing **385 entries missing the target**, I worked with **818 participants**. Final retained domains included, but were not limited to:
* Mental health history
* Social and family support
* Economic status
* Treatment engagement

---

## **How I Broke Down the Problem**
### **Reducing and Screening Features**
Starting with 221 usable features (excluding the target), I applied a series of steps:
* Ran domain-level VIF to reduce collinearity among numeric features
* Created PCA composites from binary groups within the same domain
* Preserved relevant interaction terms (17 in total) to retain the study’s experimental design effects
* Converted certain binary variables for consistency
* Excluded two domains—Cognitive & Behavioral and School Education—to prevent leakage, since they reflect child outcomes that occur after the six-month window

After deduplication, this left **103 unique features**. A final global VIF check was then applied to the combined set, which further reduced the count to **61 core features** — a more manageable foundation for selection and modeling.

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
The model highlighted several patterns that can shape how postpartum depression is followed up and managed:
* **Recovery history is a strong signal** – Mothers who had recovered well from past depression were far less likely to develop PPD again, while those with a history of chronic vulnerability needed closer monitoring.
* **Early discussions with care providers matter** – When conversations happened early, risk was often flagged sooner and support could start earlier.
* **Treatment works better with stability** – Its impact was stronger when employment and household stability were already in place.

These signals point to areas where follow-up can be more focused:
* Identify mothers with a history of recurring depression for early check-ins.
* Encourage early conversations during routine care.
* Pair treatment plans with social and economic support when possible.

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
Developed as part of the **SheCode Africa AI/ML Challenge (August–September 2025)** on maternal mental health.
