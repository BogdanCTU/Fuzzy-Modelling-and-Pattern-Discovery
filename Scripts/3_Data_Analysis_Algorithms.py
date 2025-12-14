import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.linear_model import LinearRegression
from scipy.optimize import linprog
import os

# ==========================================
# CONFIGURATION
# ==========================================
# Use 'Agg' backend to prevent plots from displaying on screen
plt.switch_backend('Agg')
plt.style.use('ggplot')

# ==========================================
# 0. ROBUST DATA LOADING
# ==========================================
def load_and_clean_data(filename='Phase3_OPD_PisitiveScale_Reducetd_Sex_F.csv'):
    """
    Loads data robustly, handling European formats.
    """
    print(f"Attempting to load: {filename}")
    
    df = None # Initialize to avoid UnboundLocalError

    # 1. Load File
    if os.path.exists(filename):
        try:
            if filename.endswith('.xlsx'):
                df = pd.read_excel(filename)
            else:
                # Try European CSV (semicolon sep, comma decimal)
                try:
                    df = pd.read_csv(filename, sep=';', decimal=',')
                except:
                    # Fallback to standard CSV
                    df = pd.read_csv(filename, sep=',', decimal='.')
        except Exception as e:
            print(f"Error reading file: {e}")
            return None
    else:
        print(f"ERROR: File '{filename}' not found in current directory.")
        print(f"Current Directory is: {os.getcwd()}")
        return None

    # If loading failed for any reason
    if df is None:
        return None

    # 2. Select & Clean Numeric Columns
    target_cols = ['BodyweightKg_Scaled','TotalKg_Scaled', 'Best3SquatKg_Scaled',
                   'Best3BenchKg_Scaled','Best3DeadliftKg_Scaled','Age_Scaled']
    
    # Check if columns exist
    available_cols = [c for c in target_cols if c in df.columns]
    
    if not available_cols:
        print(f"CRITICAL: None of the target columns found in {filename}.")
        print(f"Columns found: {list(df.columns)}")
        return None
        
    df_clean = df[available_cols].copy()
    
    # Force numeric conversion
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str).str.replace(',', '.')
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
    df_clean.dropna(inplace=True)
    return df_clean

# Load Data
df = load_and_clean_data()

# SAFETY CHECK: Stop script if df is empty/None
if df is None or df.empty:
    raise ValueError("Data loading failed. Please check the file path and column names.")

features_pca = ['BodyweightKg_Scaled','TotalKg_Scaled', 'Best3SquatKg_Scaled',
                'Best3BenchKg_Scaled','Best3DeadliftKg_Scaled','Age_Scaled']

X = df[features_pca].values

# ==========================================
# 1. PCA vs FUZZY PCA
# ==========================================
print("\n--- Running Analysis 1: PCA vs Fuzzy PCA ---")

# A. Standard PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
std_explained_variance = pca.explained_variance_ratio_

# B. Fuzzy PCA Class
class FuzzyPCA:
    def __init__(self, n_components=2, m=2, max_iter=20):
        self.n_components = n_components
        self.m = m
        self.max_iter = max_iter
        self.explained_variance_ratio_ = None
        self.components_ = None
        self.weights_ = None
        
    def fit_transform(self, X):
        n, p = X.shape
        u = np.ones(n) / n  # Initial weights
        
        for _ in range(self.max_iter):
            # 1. Weighted Center
            center = np.average(X, axis=0, weights=u**self.m)
            X_centered = X - center
            
            # 2. Weighted Covariance
            weights = np.sqrt(u**self.m).reshape(-1, 1)
            X_weighted = X_centered * weights
            # Covariance matrix C
            C = (X_weighted.T @ X_weighted) / np.sum(u**self.m)
            
            # 3. Eigendecomposition
            vals, vecs = np.linalg.eigh(C)
            idx = np.argsort(vals)[::-1]
            self.components_ = vecs[:, idx[:self.n_components]]
            
            # Store explained variance ratio based on eigenvalues of Fuzzy Covariance
            total_var = np.sum(vals)
            self.explained_variance_ratio_ = vals[idx[:self.n_components]] / total_var
            
            # 4. Update Memberships
            X_proj = X_centered @ self.components_
            X_rec = X_proj @ self.components_.T
            dist_sq = np.sum((X_centered - X_rec)**2, axis=1)
            
            u_new = 1 / (1 + dist_sq)
            u_new /= np.max(u_new) # Normalize to 0-1 range
            
            if np.allclose(u, u_new, atol=1e-4): break
            u = u_new
            
        self.weights_ = u
        return (X - center) @ self.components_

fpca = FuzzyPCA(n_components=2)
X_fpca = fpca.fit_transform(X)

# --- OUTPUT: VARIANCE COMPARISON ---
print("\n[Variance Analysis Output]")
print(f"{'Component':<15} | {'Standard PCA':<15} | {'Fuzzy PCA':<15}")
print("-" * 50)
for i in range(2):
    print(f"PC{i+1:<14} | {std_explained_variance[i]:.4f} ({std_explained_variance[i]*100:.1f}%) | {fpca.explained_variance_ratio_[i]:.4f} ({fpca.explained_variance_ratio_[i]*100:.1f}%)")
print("-" * 50)

# --- EXPORT 1: VARIANCE COMPARISON ---
fig_var, ax_var = plt.subplots(figsize=(8, 5))
indices = np.arange(2)
width = 0.35

ax_var.bar(indices - width/2, std_explained_variance, width, label='Standard PCA', color='steelblue')
ax_var.bar(indices + width/2, fpca.explained_variance_ratio_, width, label='Fuzzy PCA', color='darkorange')

ax_var.set_ylabel('Explained Variance Ratio')
ax_var.set_title('Variance Explained: Standard vs Fuzzy PCA')
ax_var.set_xticks(indices)
ax_var.set_xticklabels(['PC1', 'PC2'])
ax_var.legend()

filename1 = "PCA_vs_FuzzyPCA_Variance.png"
plt.savefig(filename1, dpi=300, bbox_inches='tight')
plt.close(fig_var)
print(f"Saved: {filename1}")

# --- EXPORT 2: PROJECTIONS ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, color='steelblue')
ax1.set_title("Standard PCA\n(Equal weights for all)")
ax1.set_xlabel("PC1")

sc = ax2.scatter(X_fpca[:, 0], X_fpca[:, 1], c=fpca.weights_, cmap='viridis', alpha=0.6)
plt.colorbar(sc, ax=ax2, label="Membership Degree")
ax2.set_title("Fuzzy PCA\n(Outliers down-weighted)")
ax2.set_xlabel("Fuzzy PC1")

filename2 = "PCA_vs_FuzzyPCA.png"
plt.savefig(filename2, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {filename2}")

# ==========================================
# 2. CLUSTERING COMPARISON
# ==========================================
print("\n--- Running Analysis 2: Clustering ---")

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_db = dbscan.fit_predict(X_pca)

# Agglomerative
agg = AgglomerativeClustering(n_clusters=3)
labels_agg = agg.fit_predict(X_pca)

# --- EXPORT 3: CLUSTERING ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
unique_labels = set(labels_db)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1: col = [0, 0, 0, 1] 
    mask = (labels_db == k)
    ax1.plot(X_pca[mask, 0], X_pca[mask, 1], 'o', markerfacecolor=tuple(col), 
             markeredgecolor='k', markersize=6, label=f"Cluster {k}")
ax1.set_title("Density-Based (DBSCAN)\n(Detects Noise/Outliers)")

ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=labels_agg, cmap='viridis', edgecolor='k')
ax2.set_title("Connectivity-Based (Agglomerative)\n(Forces Hierarchy)")

filename3 = "DBSCAN_vs_Connectivity_Clustering.png"
plt.savefig(filename3, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {filename3}")

# ==========================================
# 3. REGRESSION COMPARISON
# ==========================================
print("\n--- Running Analysis 3: Regression ---")

X_reg = df['BodyweightKg_Scaled'].values.reshape(-1, 1)
y_reg = df['TotalKg_Scaled'].values

# Standard Linear
lin = LinearRegression()
lin.fit(X_reg, y_reg)
y_pred_lin = lin.predict(X_reg)

# Fuzzy Linear (Tanaka)
n = len(y_reg)
c_obj = [0, 0, n, np.sum(np.abs(X_reg))] 
A_ub, b_ub = [], []
for i in range(n):
    x_i, y_i = X_reg[i][0], y_reg[i]
    A_ub.append([-1, -x_i, -1, -abs(x_i)])
    b_ub.append(-y_i)
    A_ub.append([1, x_i, -1, -abs(x_i)])
    b_ub.append(y_i)

res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=[(None,None)]*2 + [(0,None)]*2, method='highs')

# --- EXPORT 4: REGRESSION ---
fig = plt.figure(figsize=(10, 6))
plt.scatter(X_reg, y_reg, color='gray', alpha=0.5, label='Data')
plt.plot(X_reg, y_pred_lin, color='blue', linewidth=2, label='Standard Linear Reg.')

if res.success:
    a0, a1, c0, c1 = res.x
    x_sort = np.sort(X_reg.flatten())
    y_c = a0 + a1 * x_sort
    y_s = c0 + c1 * np.abs(x_sort)
    plt.plot(x_sort, y_c, 'r--', linewidth=2, label='Fuzzy Center')
    plt.fill_between(x_sort, y_c - y_s, y_c + y_s, color='red', alpha=0.15, label='Fuzzy Possibility Band')

plt.title("Standard vs Fuzzy Linear Regression")
plt.xlabel("Bodyweight (Scaled)")
plt.ylabel("Total (Scaled)")
plt.legend()
plt.grid(True)

filename4 = "LinearRegression_vs_FuzzyRegression.png"
plt.savefig(filename4, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {filename4}")

print("\nAll Analysis Complete. Images exported.")
