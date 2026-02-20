#!/usr/bin/env python
# coding: utf-8

# # Google Play Store Rating Prediction (Linear Regression)
# 
# This notebook:
# - Cleans and validates the Google Play Store dataset
# - Performs univariate and bivariate EDA (matplotlib only)
# - Treats outliers
# - Builds a Linear Regression model to predict **Rating**
# - Reports **R²** on train and test sets
# 
# > Dataset path used here: `/mnt/data/googleplaystore[1].csv`
# 

# In[3]:


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

DATA_PATH = "C:/Users/dell/Downloads/1569582940_googleplaystore (2)/googleplaystore.csv"


# In[4]:


# 1) Load data
df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
df.head()


# In[5]:


# 2) Null checks
null_counts = df.isna().sum().sort_values(ascending=False)
print("Null values per column:")
display(null_counts[null_counts > 0])

# 3) Drop records with nulls in ANY column
df0 = df.dropna(axis=0, how="any").copy()
print("After dropna:", df0.shape)


# ## 4) Fix types & inconsistent formatting
# 
# We will:
# - **Size**: convert `k`/`M` into numeric KB (per instructions: 1 MB = 1000 KB)
# - **Reviews**: to numeric
# - **Installs**: remove `+` and `,` then numeric
# - **Price**: remove `$` then numeric
# 

# In[6]:


df1 = df0.copy()

# --- Size: convert to numeric KB ---
def parse_size_to_kb(x):
    s = str(x).strip()
    if s.lower() == "varies with device":
        return np.nan
    if s.endswith("M"):
        # MB -> KB (multiply by 1000 as requested)
        try:
            return float(s[:-1]) * 1000.0
        except:
            return np.nan
    if s.endswith(("k", "K")):
        try:
            return float(s[:-1])
        except:
            return np.nan
    # sometimes size might be numeric already
    try:
        return float(s)
    except:
        return np.nan

df1["Size"] = df1["Size"].apply(parse_size_to_kb)

# --- Reviews: numeric ---
df1["Reviews"] = pd.to_numeric(df1["Reviews"], errors="coerce")

# --- Installs: numeric ---
df1["Installs"] = (
    df1["Installs"].astype(str)
    .str.replace("+", "", regex=False)
    .str.replace(",", "", regex=False)
)
df1["Installs"] = pd.to_numeric(df1["Installs"], errors="coerce")

# --- Price: numeric ---
df1["Price"] = df1["Price"].astype(str).str.replace("$", "", regex=False)
df1["Price"] = pd.to_numeric(df1["Price"], errors="coerce")

# Rating to numeric (safe)
df1["Rating"] = pd.to_numeric(df1["Rating"], errors="coerce")

# Drop rows that turned invalid during conversion
before = df1.shape[0]
df1 = df1.dropna(axis=0, how="any").copy()
print(f"Dropped {before - df1.shape[0]} rows after type fixes. New shape:", df1.shape)

df1[["Rating","Reviews","Size","Installs","Price"]].describe()


# ## 5) Sanity checks
# - Rating must be **between 1 and 5**
# - Reviews should be **<= Installs**
# - If Type = **Free**, Price must be **0**
# 

# In[7]:


df2 = df1.copy()

# Rating in [1,5]
df2 = df2[(df2["Rating"] >= 1) & (df2["Rating"] <= 5)]

# Reviews <= Installs
df2 = df2[df2["Reviews"] <= df2["Installs"]]

# Free apps must have Price == 0
df2 = df2[~((df2["Type"].str.lower() == "free") & (df2["Price"] > 0))]

print("After sanity checks:", df2.shape)


# ## 6) Univariate analysis (matplotlib)
# 
# - Boxplot: Price
# - Boxplot: Reviews
# - Histogram: Rating
# - Histogram: Size (KB)
# 
# Write down your observations below each plot.
# 

# In[8]:


# Price boxplot
plt.figure()
plt.boxplot(df2["Price"], vert=True)
plt.title("Boxplot: Price")
plt.ylabel("Price ($)")
plt.show()

# Reviews boxplot
plt.figure()
plt.boxplot(df2["Reviews"], vert=True)
plt.title("Boxplot: Reviews")
plt.ylabel("Number of Reviews")
plt.show()

# Rating histogram
plt.figure()
plt.hist(df2["Rating"].values, bins=30)
plt.title("Histogram: Rating")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

# Size histogram (KB)
plt.figure()
plt.hist(df2["Size"].values, bins=30)
plt.title("Histogram: Size (KB)")
plt.xlabel("Size (KB)")
plt.ylabel("Count")
plt.show()


# ## 7) Outlier treatment
# 
# - **Price**: drop apps with Price > 200 (suspicious/junk)
# - **Reviews**: drop apps with Reviews > 2,000,000
# - **Installs**: compute percentiles and choose a cutoff; drop above the cutoff
# 
# > Cutoff used below: **99th percentile** of Installs (top 1% dropped).  
# You can change to 95th/97th/etc. if your distribution suggests it.
# 

# In[9]:


df3 = df2.copy()

# Price outliers (>200)
high_price = df3[df3["Price"] > 200].sort_values("Price", ascending=False)
print("High price records (Price > 200):", high_price.shape[0])
display(high_price[["App","Category","Type","Price","Rating","Reviews","Installs"]].head(20))

df3 = df3[df3["Price"] <= 200]

# Reviews outliers (>2 million)
df3 = df3[df3["Reviews"] <= 2_000_000]

# Installs percentiles
percentiles = [10, 25, 50, 70, 90, 95, 99]
perc_values = np.percentile(df3["Installs"], percentiles)
inst_perc = pd.DataFrame({"percentile": percentiles, "installs": perc_values.astype(int)})
display(inst_perc)

cutoff = np.percentile(df3["Installs"], 99)
print("Chosen installs cutoff (99th percentile):", int(cutoff))

df3 = df3[df3["Installs"] <= cutoff]

print("After outlier treatment:", df3.shape)


# ## 8) Bivariate analysis
# 
# - Scatter: Rating vs Price
# - Scatter: Rating vs Size
# - Scatter: Rating vs Reviews
# - Boxplot: Rating vs Content Rating
# - Boxplot: Rating vs Category
# 

# In[10]:


# Scatter: Rating vs Price
plt.figure()
plt.scatter(df3["Price"], df3["Rating"], alpha=0.4)
plt.title("Rating vs Price")
plt.xlabel("Price ($)")
plt.ylabel("Rating")
plt.show()

# Scatter: Rating vs Size
plt.figure()
plt.scatter(df3["Size"], df3["Rating"], alpha=0.4)
plt.title("Rating vs Size (KB)")
plt.xlabel("Size (KB)")
plt.ylabel("Rating")
plt.show()

# Scatter: Rating vs Reviews
plt.figure()
plt.scatter(df3["Reviews"], df3["Rating"], alpha=0.4)
plt.title("Rating vs Reviews")
plt.xlabel("Reviews")
plt.ylabel("Rating")
plt.show()

# Boxplot: Rating vs Content Rating
plt.figure(figsize=(10, 5))
df3.boxplot(column="Rating", by="Content Rating", rot=45)
plt.title("Rating vs Content Rating")
plt.suptitle("")
plt.xlabel("Content Rating")
plt.ylabel("Rating")
plt.show()

# Boxplot: Rating vs Category (may be many categories)
plt.figure(figsize=(14, 6))
df3.boxplot(column="Rating", by="Category", rot=90)
plt.title("Rating vs Category")
plt.suptitle("")
plt.xlabel("Category")
plt.ylabel("Rating")
plt.show()


# ## 9) Data preprocessing (inp1 → inp2)
# 
# - Create **inp1** copy
# - Log-transform `Reviews` and `Installs` using `np.log1p`
# - Drop columns: `App`, `Last Updated`, `Current Ver`, `Android Ver`
# - Dummy encode: `Category`, `Genres`, `Content Rating` → **inp2**
# 

# In[11]:


# inp1 copy
inp1 = df3.copy()

# log transformation to reduce skew
inp1["Reviews"] = np.log1p(inp1["Reviews"])
inp1["Installs"] = np.log1p(inp1["Installs"])

# Drop unneeded columns
drop_cols = ["App", "Last Updated", "Current Ver", "Android Ver"]
inp1 = inp1.drop(columns=[c for c in drop_cols if c in inp1.columns])

# Dummy encoding
inp2 = pd.get_dummies(
    inp1,
    columns=["Category", "Genres", "Content Rating"],
    drop_first=True
)

print("inp2 shape:", inp2.shape)
inp2.head()


# ## 10) Train-test split (70-30) and modeling
# 
# - Split into **df_train** and **df_test**
# - Separate into **X_train, y_train, X_test, y_test**
# - Fit **LinearRegression**
# - Report **R²** on train and test
# 

# In[14]:


# Dummy encoding (include Type too!)
inp2 = pd.get_dummies(
    inp1,
    columns=["Category", "Genres", "Content Rating", "Type"],
    drop_first=True
)

# 9) Train-test split 70-30
df_train, df_test = train_test_split(inp2, test_size=0.30, random_state=42)

# 10) Separate X and y
target = "Rating"
X_train = df_train.drop(columns=[target])
y_train = df_train[target]

X_test = df_test.drop(columns=[target])
y_test = df_test[target]

print("Train:", X_train.shape, "Test:", X_test.shape)

# 11) Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# R2 on train
train_pred = lr.predict(X_train)
r2_train = r2_score(y_train, train_pred)
print("R² (train):", round(r2_train, 4))

# 12) Predictions on test & R2
test_pred = lr.predict(X_test)
r2_test = r2_score(y_test, test_pred)
print("R² (test):", round(r2_test, 4))


# In[15]:


# Safety check: no object/string columns should remain
obj_cols = X_train.select_dtypes(include=["object"]).columns
print("Object columns still in X_train:", list(obj_cols))

# If any exist, convert them to dummies
if len(obj_cols) > 0:
    X_train = pd.get_dummies(X_train, columns=obj_cols, drop_first=True)
    X_test  = pd.get_dummies(X_test,  columns=obj_cols, drop_first=True)

    # align train and test columns (very important)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)


# ## 11) Save artifacts (optional)
# 
# This cell saves:
# - cleaned dataset (`cleaned_playstore.csv`)
# - model coefficients (`linear_regression_coeffs.csv`)
# 

# In[17]:


# Save cleaned dataset (after outliers & fixes)
clean_path = "cleaned_playstore.csv"
df3.to_csv(clean_path, index=False)
print("Saved:", clean_path)

# Ensure model is fitted before accessing coef_
if hasattr(lr, "coef_"):
    coef = pd.Series(
        lr.coef_,
        index=X_train.columns
    ).sort_values(key=lambda s: np.abs(s), ascending=False)

    coef_path = "linear_regression_coeffs.csv"
    coef.to_csv(coef_path, header=["coefficient"])
    print("Saved:", coef_path)

    display(coef.head(20))
else:
    print("Model is not fitted yet. Run the training cell first.")


# In[ ]:




