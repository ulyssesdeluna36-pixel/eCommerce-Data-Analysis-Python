import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. DATA FILE SETUP ---

# IMPORTANT: You confirmed this file name is correct and the file is in the folder.
file_name = 'ecommerce_customer_behavior_dataset.xlsx'
# You confirmed your data loaded when specifying the sheet name was missing.
# If your sheet name is 'Sheet1' (default), use 'Sheet1'. If it's different (e.g., 'Data'), use that.
sheet_name = 'ecommerce_customer_behavior_dat'
# Initialize df as None before the try block
df = None

# --- 2. DATA LOADING & INITIAL INSPECTION ---

try:
    # Load the data, using the sheet name fix
    df = pd.read_excel(file_name, sheet_name=sheet_name)

    # Initial Inspection Output
    print("\n--- Data successfully loaded! ---")
    print("\n--- First 5 Rows of Data ---")
    print(df.head())
    print("\n--- Data Information (Types, Missing Values) ---")
    df.info()

# Handle specific errors
except FileNotFoundError:
    print(f"\nERROR: File '{file_name}' not found. Please check the spelling!")
except ValueError:
    print(f"\nERROR: Sheet '{sheet_name}' not found in the file. Check the sheet name!")
except Exception as e:
    print(f"\nOTHER ERROR: Data loading failed due to a different issue: {e}")

# --- 3. DATA CLEANING & PREPARATION (This runs ONLY if loading was successful) ---

if df is not None:
    print("\n\n--- PHASE 1: DATA CLEANING ---")

    # 1. Standardize column names (lowercase and use underscores)
    df.columns = df.columns.str.lower().str.replace('[^a-z0-9_]', '_', regex=True)

    # 2. Rename specific messy/truncated columns
    rename_map = {
        'ustomer_rating': 'customer_rating',  # Fixes the truncated name
        'product_c_unit_price': 'unit_price',
        'discount_a_total_amount': 'total_amount',
        'payment_meth_device_type': 'payment_method',
        't_customer_rating': 'customer_rating_t'
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    print("\nColumn names standardized and renamed.")

    # 3. Handle Missing Data (NaN)
    print("\n--- Missing Value Count (Before Drop) ---")
    print(df.isnull().sum())

    # Strategy 1: Remove rows where the Total Amount is missing
    df.dropna(subset=['total_amount'], inplace=True)
    print(f"\nTotal rows after dropping missing 'total_amount': {len(df)}")

    # Strategy 2: Fill remaining numerical data gaps (like Age) with the mean
    if 'age' in df.columns:
        age_mean = df['age'].mean()
        df['age'] = df['age'].fillna(round(age_mean))
        df['age'] = df['age'].astype(int)

        # Final check of the cleaned data types and structure
    print("\n--- Data Cleaning Complete. Final Info ---")
    df.info()

print("\n\n--- PHASE 2: EXPLORATORY DATA ANALYSIS (EDA) ---")

# 1. Descriptive Statistics
print("\n--- Key Statistical Summary of Numerical Data ---")
# 'Total_amount' shows spending patterns, 'Age' shows demographics.
print("\n--- DIAGNOSTIC: CLEANED COLUMN NAMES ---")
print(df.columns.tolist())
# ----------------------------------------
print(df[['total_amount', 'age', 'session_duration_minutes', 'customer_rating']].describe())
print(df.columns.tolist())

# 2. Categorical Value Counts
print("\n--- Top Cities by Transaction Count ---")
print(df['city'].value_counts().head(5))

print("\n--- Top 5 Products by Transaction Count ---")
print(df['product_category'].value_counts().head(5))

print("\n--- Payment Method Counts ---")
print(df['payment_method'].value_counts())

print("\n\n--- PHASE 2: EXPLORATORY DATA ANALYSIS (EDA) ---")

# 1. Descriptive Statistics
print("\n--- Key Statistical Summary of Numerical Data ---")
# 'total_amount' shows spending patterns, 'age' shows demographics.
print(df[['total_amount', 'age', 'session_duration_minutes', 'customer_rating']].describe())

# 2. Categorical Value Counts
print("\n--- Top Cities by Transaction Count ---")
print(df['city'].value_counts().head(5))

print("\n--- Top 5 Products by Transaction Count ---")
print(df['product_category'].value_counts().head(5))
print("\n\n--- PHASE 3: DEEP-DIVE ANALYSIS ---")

# Calculate Average Order Value (AOV) overall
aov_overall = df['total_amount'].mean()
print(f"\n1. Overall Average Order Value (AOV): ${aov_overall:.2f}")

# Average Order Value (AOV) by Returning Customer status
aov_by_return = df.groupby('is_returning_customer')['total_amount'].mean()
print("\nAverage Order Value by Returning Customer:")
print(aov_by_return)
# Calculate Total Revenue by Product Category
product_revenue = df.groupby('product_category')['total_amount'].sum().sort_values(ascending=False)
print("\nTop 5 Product Categories by Total Revenue:")
print(product_revenue.head(5))
# Calculate the correlation between discount and price
correlation = df[['discount_amount', 'unit_price']].corr().iloc[0, 1]
print(f"\nCorrelation between Discount Amount and Unit Price: {correlation:.4f}")
# A strong negative correlation might suggest discounts are targeted at higher-priced items.
# Calculate the average number of pages viewed per minute of session duration
df['pages_per_minute'] = df['pages_viewed'] / df['session_duration_minutes']
avg_pages_per_minute = df['pages_per_minute'].mean()
print(f"\nAverage Pages Viewed per Minute of Session: {avg_pages_per_minute:.2f}")
print("\n\n--- PHASE 4: ADVANCED VISUALIZATION ---")

plt.figure(figsize=(9, 6))
sns.scatterplot(x='session_duration_minutes', y='pages_viewed', data=df, hue='device_type', alpha=0.6)
plt.title('Session Duration vs. Pages Viewed (Colored by Device)')
plt.xlabel('Session Duration (Minutes)')
plt.ylabel('Pages Viewed')
plt.show() #
# Group data to plot Total Revenue by Payment Method
payment_revenue = df.groupby('payment_method')['total_amount'].sum().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=payment_revenue.index, y=payment_revenue.values, palette='viridis')
plt.title('Total Revenue by Payment Method')
plt.xlabel('Payment Method')
plt.ylabel('Total Revenue (Units)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


print("\n--- Payment Method Counts ---")
print(df['payment_method'].value_counts())

# 3. Visualization: Age Distribution
plt.figure(figsize=(9, 5))
# KDE (Kernel Density Estimate) smooths the distribution curve.
sns.histplot(df['age'], bins=30, kde=True, color='skyblue')
plt.title('Distribution of Customer Ages')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show() #

# 4. Visualization: Spending by Gender
plt.figure(figsize=(7, 6))
# We use a box plot to see the median and range of spending for each gender.
sns.boxplot(x='gender', y='total_amount', data=df, palette='pastel')
plt.title('Total Amount Spent by Gender')
plt.show() #