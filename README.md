# Fuzzy Modeling and Pattern Discovery in Strength Sports
## A Comparative Analysis of Crisp vs. Fuzzy Logic on Open Powerlifting Data
## **Author:** Todoran Ciprian Bogdan
## **Course:** Advanced Methods in Data Analysis

---

## Overview
This research investigates the limitations of standard "crisp" statistical algorithms (Standard PCA, K-Means, OLS Regression) when applied to biological and athletic performance data. Using the massive **Open Powerlifting** dataset, this project demonstrates that human performance is inherently "fuzzy"—a continuous, noisy manifold rather than a set of distinct, precise clusters.

The project compares standard methods against **Fuzzy Logic counterparts** (Fuzzy PCA, Fuzzy Clustering, Fuzzy Regression) to determine which approach better models the "normal spectrum" of raw power versus elite outliers.

---

## Key Features & Findings

###1. Dimensionality Reduction: The "Manifold" Discovery* **Objective:** Compare Standard PCA vs. Fuzzy PCA in handling extreme outliers (e.g., super-heavyweight lifters).
* **Finding:**     * **Fuzzy PCA** consistently outperformed Standard PCA, explaining significantly higher variance (Female: **89.42%**, Male: **91.87%**) by down-weighting noise and outliers.
* Standard PCA was skewed by "freak" performances, whereas Fuzzy PCA captured the true core trend.

### 2. Clustering: Density vs. Connectivity* **Objective:** Determine if athletes fall into natural "types" or exist on a continuum.
* **Finding:**
* **Male Subset (High Density):** Supported **DBSCAN** (Silhouette=0.5371), identifying a massive, cohesive "normal" core separated from noise.
* **Female Subset (Lower Density):** DBSCAN failed (< 2 clusters). Structure had to be forced via **Agglomerative Clustering**, proving that natural distinct clusters do not exist in the data.

### 3. Prediction: The "Green Zone"* **Objective:** Move beyond predicting a single "mean" value to defining a valid performance range.
* **Finding:**     * **Soft Margin Fuzzy Regression** created a robust "Corridor of Expected Performance" (Green Zone).
* This band allows for quantitative talent identification: athletes above the band are "Elite," while those inside represent standard variance.

---

## Repository Structure & Python ScriptsThis project is built using a modular Python workflow.

| Script | Description |
| --- | --- |
| `1_Data_PreProcessing_Cleaning.py` | **Robust Loading & Cleaning:** Handles European CSV formats, imputes missing Age/Sex data, and applies strict biological boundary filters (e.g., removing negative weights). |
| `2_Data_PreProcessing_Transform.py` | **Feature Engineering:** Constructs the `Strength_to_Weight_Ratio` feature and applies Z-score Normalization to center the data for PCA. |
| `3_Data_Analysis_Algorithms.py` | **Core Analysis Engine:** <br>

<br>• **Fuzzy PCA:** Custom implementation with iterative membership updates.<br>
<br>• **Clustering:** Comparative execution of DBSCAN vs. Agglomerative.<br>
<br>• **Regression:** Solves Soft Margin Fuzzy Regression using Linear Programming (`scipy.optimize.linprog`). |

---

## Methodology
### Data Source
The study utilizes the **Open Powerlifting** dataset (~1,000,000 records), sampled into two distinct subsets to ensure statistical significance:

* **Female Subset:** ~35,700 records 
* **Male Subset:** ~66,300 records 

###Technologies Used* **Language:** Python 3.9+
* **Libraries:** `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `scipy`
* **Techniques:** Fuzzy Logic, Principal Component Analysis (PCA), Density-Based Clustering (DBSCAN), Linear Programming (Simplex Method).

---

 ![Tux, the Linux mascot](/Images/2_)

### 1. Variance AnalysisFuzzy PCA explains ~10% more variance than standard PCA by filtering out biological noise.
### 2. Clustering CompositionThe contrast between Density-Based (DBSCAN) and Connectivity-Based clustering highlights the continuous nature of the data.
### 3. Fuzzy Regression PredictionThe "Green Band" (Soft Fuzzy) provides a usable scouting range, unlike the "Red Dotted" (Hard Fuzzy) which is broken by outliers.

---

## Conclusion
The results confirm that athletic performance data is a **continuous, noisy manifold**.

1. **Use Fuzzy PCA:** It is the superior feature extraction method for biological data.
2. **Avoid Rigid Clustering:** Distinct groups (Cluster A vs B) are often artificial; performance scales fluidly.
3. **Benchmarking:** Soft Margin Fuzzy Regression provides the most actionable insight for coaches and analysts.

---

# **Acknowledgement:**

* Dataset provided by [OpenPowerlifting.org](https://www.openpowerlifting.org).
* Research conducted at **Babes-Bolyai University**.
