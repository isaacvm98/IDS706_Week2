"""Basic Data Analysis of Mexican Stocks using yfinance and Polars"""

import os
import yfinance as yf
import polars as pl
import polars.selectors as cs
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import networkx as nx


# Create visualizations directory if it doesn't exist
os.makedirs("visualizations", exist_ok=True)


def get_stock_symbols():
    """Return list of Mexican stock symbols for analysis."""
    return [
        "GFNORTEO.MX",
        "GFINBURO.MX",
        "WALMEX.MX",
        "BIMBOA.MX",
        "FEMSAUBD.MX",
        "KOFUBL.MX",
        "AC.MX",
        "LIVEPOL1.MX",
        "TLEVISACPO.MX",
        "CEMEXCPO.MX",
        "GMEXICOB.MX",
        "PE&OLES.MX",
        "ALPEKA.MX",
        "GAPB.MX",
        "OMAB.MX",
        "ASURB.MX",
        "FIBRAMQ12.MX",
        "LABB.MX",
        "GCARSOA1.MX",
        "GRUMAB.MX",
    ]


def download_stock_data(
    symbols,
    start_date="2020-01-01",
    end_date="2025-09-05"
):
    """Download stock data for given symbols and date range."""
    if not symbols:
        raise ValueError("Symbols list cannot be empty")

    raw_data = yf.download(symbols, start_date, end_date, auto_adjust=False)
    if raw_data.empty:
        raise ValueError("No data downloaded for given symbols and date range")

    return raw_data


def extract_close_prices(raw_data):
    """Extract close prices from raw yfinance data and return as Polars
    DataFrame."""
    df = pl.from_pandas(raw_data, include_index=True)

    # Only keep the 'Close' prices
    df = df.select(
        pl.col("Date"),
        cs.starts_with("('Close'").name.map(
            lambda col_name: col_name.split("', '")[1].rstrip("')")
        ),
    )

    return df


def calculate_correlation_matrix(df):
    """Calculate correlation matrix from price DataFrame."""
    if df.shape[1] < 2:
        raise ValueError("DataFrame must have at least 2 columns")

    # Remove Date column for correlation calculation
    price_columns = [col for col in df.columns if col != "Date"]
    corr_matrix = df.select(price_columns).corr()

    return corr_matrix


def find_optimal_clusters(corr_matrix, max_k=8):
    """Find optimal number of clusters using elbow method."""
    inertias = []
    k_range = range(2, max_k)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(corr_matrix)
        inertias.append(kmeans.inertia_)

    return list(k_range), inertias


def perform_kmeans_clustering(corr_matrix, n_clusters=4, random_state=42):
    """Perform K-means clustering on correlation matrix."""
    if n_clusters < 1:
        raise ValueError("Number of clusters must be positive")
    if n_clusters > corr_matrix.shape[0]:
        raise ValueError("Number of clusters cannot exceed number of stocks")

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(corr_matrix)

    # Create cluster groups
    cluster_groups = {}
    for stock, cluster in zip(corr_matrix.columns, kmeans.labels_):
        if cluster not in cluster_groups:
            cluster_groups[cluster] = []
        cluster_groups[cluster].append(stock)

    return cluster_groups


def calculate_returns(df, stock1_col, stock2_col):
    """Calculate returns for two stocks."""
    df_returns = df.select(
        [
            pl.col("Date"),
            (pl.col(stock1_col).pct_change()).alias(f"{stock1_col}_returns"),
            (pl.col(stock2_col).pct_change()).alias(f"{stock2_col}_returns"),
        ]
    ).drop_nulls()

    return df_returns


def build_linear_model(df_returns, predictor_col, target_col, test_size=30):
    """Build linear regression model to predict one stock from another."""
    if len(df_returns) <= test_size:
        raise ValueError("Not enough data for train/test split")

    # Create train/test split
    n_train = len(df_returns) - test_size

    X_train = df_returns[predictor_col][:n_train].to_numpy().reshape(-1, 1)
    y_train = df_returns[target_col][:n_train].to_numpy()

    X_test = df_returns[predictor_col][n_train:].to_numpy().reshape(-1, 1)
    y_test = df_returns[target_col][n_train:].to_numpy()

    # Fit model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions and metrics
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "model": model,
        "mse": mse,
        "r2": r2,
        "coefficient": model.coef_[0],
        "intercept": model.intercept_,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def create_correlation_heatmap(corr_matrix, save_path=None):
    """Create and optionally save correlation heatmap."""
    mask = np.triu(np.ones_like(corr_matrix), k=1)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
    )
    plt.xticks(
        rotation=90,
        ha="right",
        ticks=np.arange(len(corr_matrix.columns)) + 0.5,
        labels=corr_matrix.columns,
    )
    plt.yticks(
        rotation=0,
        ticks=np.arange(len(corr_matrix.columns)) + 0.5,
        labels=corr_matrix.columns,
    )
    plt.title("Mexican Stocks Correlation Matrix")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def create_elbow_plot(k_range, inertias, save_path=None):
    """Create and optionally save elbow plot."""
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, "bo-")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (WCSS)")
    plt.title("Elbow Method for Optimal k")
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def create_returns_plot(df_returns, stock1_col, stock2_col, save_path=None):
    """Create and optionally save returns time series plot."""
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=df_returns, x="Date", y=f"{stock1_col}_returns", label=f"{stock1_col}"
    )
    sns.lineplot(
        data=df_returns,
        x="Date",
        y=f"{stock2_col}_returns",
        label=f"{stock2_col}",
    )
    plt.title(f"Stock Returns Over Time: {stock1_col} vs {stock2_col}")
    plt.xlabel("Date")
    plt.ylabel("Daily Returns")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def print_cluster_results(cluster_groups):
    """Print clustering results in a formatted way."""
    print("Mexican Stock Correlation Clusters")
    print("=" * 50)
    for cluster_id in sorted(cluster_groups.keys()):
        for stock in sorted(cluster_groups[cluster_id]):
            print(f"  • {stock}")


def simple_correlation_network(corr_matrix):
    """Create basic network showing strongest correlations"""
    G = nx.Graph()
    # Add all stocks as nodes
    for stock in corr_matrix.columns:
        G.add_node(stock.replace('.MX', ''))  # Clean names
    # Add edges for strong correlations (>0.6)
    for i in range(len(corr_matrix)):
        for j in range(i + 1, len(corr_matrix)):
            corr_val = corr_matrix.item(i, j)
            if corr_val > 0.6:  # Only strong correlations
                stock1 = corr_matrix.columns[i].replace('.MX', '')
                stock2 = corr_matrix.columns[j].replace('.MX', '')
                G.add_edge(stock1, stock2, weight=corr_val)
    # Simple visualization
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=2, iterations=50)
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')
    nx.draw_networkx_labels(G, pos, font_size=8)
    nx.draw_networkx_edges(G, pos, width=2, alpha=0.6, edge_color='gray')
    plt.title("Mexican Stocks Network (Correlation > 0.6)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def print_model_results(model_results, predictor_name, target_name):
    """Print linear model results in a formatted way."""
    print(f"Predicting {target_name} using {predictor_name}:")
    print(f"MSE: {model_results['mse']:.6f}")
    print(f"R²: {model_results['r2']:.3f}")
    print(
        f"Equation: {target_name} = {model_results['coefficient']:.3f} * "
        f"{predictor_name} + {model_results['intercept']:.3f}"
    )


def main():
    """Main analysis pipeline."""
    # Get data
    mexican_stocks = get_stock_symbols()
    raw_data = download_stock_data(mexican_stocks)
    df = extract_close_prices(raw_data)

    print("Basic Data Info:")
    print(df.head())
    print(f"Columns: {df.columns}")
    print(f"Shape: {df.shape}")

    # Correlation analysis
    corr_matrix = calculate_correlation_matrix(df)
    create_correlation_heatmap(
        corr_matrix, "visualizations/stock_correlation.png")

    # Clustering
    k_range, inertias = find_optimal_clusters(corr_matrix)
    create_elbow_plot(k_range, inertias, "visualizations/elbow.png")

    cluster_groups = perform_kmeans_clustering(corr_matrix)
    print_cluster_results(cluster_groups)

    # Time series modeling (Cluster 2: Consumer Staples)
    df_returns = calculate_returns(df, "WALMEX.MX", "BIMBOA.MX")
    create_returns_plot(
        df_returns, "WALMEX.MX", "BIMBOA.MX", "visualizations/stock_returns.png"
    )
    simple_correlation_network(corr_matrix)

    model_results = build_linear_model(
        df_returns, "WALMEX.MX_returns", "BIMBOA.MX_returns"
    )
    print_model_results(model_results, "WALMEX", "BIMBOA")


if __name__ == "__main__":
    main()
