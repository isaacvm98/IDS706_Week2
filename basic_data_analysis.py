import yfinance as yf
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import polars.selectors as cs
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

## Selected stocks with their respective sectors

mexican_stocks =  [
            # Financial Sector (Peso GFNORTE, BBVA, Santander, Inbursa)
            'GFNORTEO.MX',    # Grupo Financiero Banorte - Leading Mexican bank
            'GFINBURO.MX',    # Grupo Financiero Inbursa - Carlos Slim's bank
            
            # Retail & Consumer Staples
            'WALMEX.MX',      # Wal-Mart de México - Largest retailer
            'BIMBOA.MX',      # Grupo Bimbo - Global bakery leader
            'FEMSAUBD.MX',     # Fomento Económico Mexicano - FEMSA conglomerate
            'KOFUBL.MX',      # Coca-Cola FEMSA - Largest Coke bottler in LatAm
            'AC.MX',          # Arca Continental - Coca-Cola bottler
            'LIVEPOL1.MX',    # Liverpool - Premium department stores
            
            # Telecommunications & Media
            'TLEVISACPO.MX',  # Grupo Televisa - Media conglomerate
            
            # Materials & Mining (Commodity Exposure)
            'CEMEXCPO.MX',    # Cemex - Global cement giant
            'GMEXICOB.MX',    # Grupo México - Mining and transportation
            'PE&OLES.MX',     # Industrias Peñoles - Precious metals
            'ALPEKA.MX',      # Alfa - Industrial conglomerate
            
            # Transportation & Infrastructure
            'GAPB.MX',        # Grupo Aeroportuario del Pacífico - Airport operator
            'OMAB.MX',        # Grupo Aeroportuario del Centro Norte - Airports
            'ASURB.MX',       # Grupo Aeroportuario del Sureste - Airport group
            
            # Real Estate Investment Trusts (FIBRAs)
            'FIBRAMQ12.MX',    # Fibra Mty - Industrial real estate
            
            # Healthcare & Consumer Discretionary
            'LABB.MX',        # Genomma Lab Internacional - Pharma & personal care
            'GCARSOA1.MX',     # Grupo Carso - Retail and industrial conglomerate
            
            # Additional Diversification
            'GRUMAB.MX',      # Grupo Financiero Multiva
        ]


raw_data = yf.download(mexican_stocks, "2020-01-01", "2025-09-05")
df = pl.from_pandas(raw_data,include_index=True)

print(df.head())
print(df.columns)
print(df.describe())

# Basic Filtering and Grouping
## Only keep the 'Close' prices
df  = df.select(
    pl.col('Date'),
    cs.starts_with("('Close'").name.map(lambda col_name: col_name.split("', '")[1].rstrip("')"))
)

print('Verifying Close Prices DataFrame:')
print(df.head())
print(df.columns)
print(df.describe())


corr_matrix = df.select(df.columns[1:]).corr()
mask = np.triu(np.ones_like(corr_matrix), k=1)

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, 
            mask=mask, 
            annot=True, 
            fmt=".2f", 
            cmap="coolwarm",
            square=True)
plt.xticks(rotation=90, ha='right',ticks=np.arange(len(corr_matrix.columns))+0.5, labels=corr_matrix.columns)
plt.yticks(rotation=0, ticks=np.arange(len(corr_matrix.columns))+0.5, labels=corr_matrix.columns)
plt.tight_layout()
plt.savefig('visualizations/stock_correlation.png', dpi=300, bbox_inches='tight')

inertias = []
for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(corr_matrix)
    inertias.append(kmeans.inertia_)
plt.figure(figsize=(12, 10))
plt.plot(range(2, 8), inertias, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (WCSS)')
plt.title('Elbow Method for Optimal k')
plt.savefig('visualizations/elbow.png', dpi=300, bbox_inches='tight')

# From the elbow plot, we choose k=4
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(corr_matrix)

cluster_groups = {}
for stock, cluster in zip(corr_matrix.columns, kmeans.labels_):
    if cluster not in cluster_groups:
        cluster_groups[cluster] = []
    cluster_groups[cluster].append(stock)

# Create a more readable format
print("Mexican Stock Correlation Clusters")
print("=" * 50)
for cluster_id in sorted(cluster_groups.keys()):
    print(f"\nCluster {cluster_id} ({len(cluster_groups[cluster_id])} stocks):")
    print("-" * 30)
    for stock in sorted(cluster_groups[cluster_id]):
        print(f"  • {stock}")


## Forecasting Cluster 2 (Consumer Staples - 2 stocks): BIMBOA.MX (Grupo Bimbo - food) + WALMEX.MX (Walmart México - retail)

#It is usually better to model returns than prices
df_returns = df.select([
    pl.col('Date'),
    (pl.col('WALMEX.MX').pct_change()).alias('WALMEX_returns'),
    (pl.col('BIMBOA.MX').pct_change()).alias('BIMBOA_returns')
]).drop_nulls()

plt.figure(figsize=(12, 10))
sns.lineplot(data=df_returns, x='Date', y='BIMBOA_returns', label='BIMBOA_returns')
sns.lineplot(data=df_returns, x='Date', y='WALMEX_returns', label='WALMEX_returns')
plt.title('Stock Returns Over Time for Cluster 2')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.savefig('visualizations/stock_returns.png', dpi=300, bbox_inches='tight')

# Create train/test split (last 30 days for testing)
n_test = 30
n_train = len(df) - n_test

# Extract data using Polars syntax
X_train = df_returns['WALMEX_returns'][:n_train].to_numpy().reshape(-1, 1)
y_train = df_returns['BIMBOA_returns'][:n_train].to_numpy()

X_test = df_returns['WALMEX_returns'][n_train:].to_numpy().reshape(-1, 1)
y_test = df_returns['BIMBOA_returns'][n_train:].to_numpy()

# Fit model
stock_model = LinearRegression()
stock_model.fit(X_train, y_train)

# Predictions
y_pred = stock_model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Predicting BIMBOA using WALMEX:")
print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.2f}")
print(f"Equation: BIMBOA = {stock_model.coef_[0]:.3f} * WALMEX + {stock_model.intercept_:.3f}")

