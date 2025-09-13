# test_stock_analysis.py
import unittest
import numpy as np
from basic_data_analysis import (
    get_mexican_stock_symbols,
    extract_close_prices,
    download_stock_data,
    calculate_correlation_matrix,
    perform_kmeans_clustering,
    calculate_returns,
    build_linear_model,
)


# =====================================================
# 1. DATA PROCESSING TESTS
# =====================================================


class TestDataProcessing(unittest.TestCase):
    """Tests for data processing functions."""

    def setUp(self):
        self.stocks = ["WALMEX.MX", "BIMBOA.MX", "GFNORTEO.MX"]
        self.raw_data = download_stock_data(self.stocks)
        self.df = extract_close_prices(self.raw_data)

    def test_get_mexican_stock_symbols_returns_list(self):
        """Test that stock symbols function returns a list."""
        symbols = get_mexican_stock_symbols()
        self.assertIsInstance(symbols, list)
        self.assertGreater(len(symbols), 0)
        self.assertTrue(all(symbol.endswith(".MX") for symbol in symbols))

    def test_extract_close_prices_structure(self):
        """Test that close price extraction maintains correct structure."""
        self.assertIn("Date", self.df.columns)
        self.assertTrue(
            all(col.endswith(".MX") for col in self.df.columns if col != "Date")
        )
        self.assertGreater(len(self.df.columns), 2)  # At least Date + 1 stock

    def test_download_returns_valid_data(self):
        """Test that downloaded data is not empty and has expected structure."""
        self.assertFalse(self.raw_data.empty)
        self.assertGreater(len(self.raw_data), 100)  # Should have many trading days

    def test_extract_close_prices_removes_other_columns(self):
        """Test that only Date and Close prices are kept."""
        # Should only have Date + stock symbols, no Open/High/Low/Volume
        expected_columns = len(self.stocks) + 1  # stocks + Date
        self.assertEqual(len(self.df.columns), expected_columns)


# =====================================================
# 2. CORRELATION MATRIX TESTS
# =====================================================


class TestCorrelationMatrix(unittest.TestCase):
    """Tests for correlation matrix calculation."""

    def setUp(self):
        self.stocks = ["WALMEX.MX", "BIMBOA.MX", "GFNORTEO.MX"]
        self.raw_data = download_stock_data(self.stocks)
        self.df = extract_close_prices(self.raw_data)
        self.corr_matrix = calculate_correlation_matrix(self.df)

    def test_correlation_matrix_shape(self):
        """Test that correlation matrix has correct shape."""
        num_stocks = len(self.df.columns) - 1  # Exclude Date column
        self.assertEqual(self.corr_matrix.shape, (num_stocks, num_stocks))

    def test_correlation_values_in_valid_range(self):
        """Test that all correlation values are between -1 and 1."""
        corr_array = self.corr_matrix.to_numpy()
        self.assertTrue(np.all(corr_array >= -1.0))
        self.assertTrue(np.all(corr_array <= 1.0))


# =====================================================
# 3. CLUSTERING TESTS
# =====================================================


class TestClustering(unittest.TestCase):
    """Tests for K-means clustering functionality."""

    def setUp(self):
        self.stocks = ["WALMEX.MX", "BIMBOA.MX", "GFNORTEO.MX", "CEMEXCPO.MX"]
        self.raw_data = download_stock_data(self.stocks)
        self.df = extract_close_prices(self.raw_data)
        self.corr_matrix = calculate_correlation_matrix(self.df)

    def test_clustering_returns_correct_number_of_clusters(self):
        """Test that clustering returns the requested number of clusters."""
        n_clusters = 2
        kmeans, cluster_groups = perform_kmeans_clustering(
            self.corr_matrix, n_clusters=n_clusters
        )
        self.assertEqual(len(cluster_groups), n_clusters)

    def test_all_stocks_assigned_to_clusters(self):
        """Test that every stock is assigned to exactly one cluster."""
        kmeans, cluster_groups = perform_kmeans_clustering(
            self.corr_matrix, n_clusters=2
        )

        # Count total stocks in all clusters
        total_assigned = sum(len(stocks) for stocks in cluster_groups.values())
        expected_stocks = len(self.corr_matrix.columns)
        self.assertEqual(total_assigned, expected_stocks)

    def test_clustering_with_invalid_parameters(self):
        """Test that invalid clustering parameters raise appropriate errors."""
        with self.assertRaises(ValueError):
            perform_kmeans_clustering(self.corr_matrix, n_clusters=0)

        with self.assertRaises(ValueError):
            perform_kmeans_clustering(self.corr_matrix, n_clusters=-1)


# =====================================================
# 4. TIME SERIES MODELING TESTS
# =====================================================


class TestTimeSeriesModeling(unittest.TestCase):
    """Tests for time series modeling and regression functionality."""

    def setUp(self):
        self.stocks = ["WALMEX.MX", "BIMBOA.MX"]
        self.raw_data = download_stock_data(self.stocks)
        self.df = extract_close_prices(self.raw_data)
        self.df_returns = calculate_returns(self.df, "WALMEX.MX", "BIMBOA.MX")

    def test_calculate_returns_structure(self):
        """Test that returns calculation produces correct structure."""
        expected_columns = ["Date", "WALMEX.MX_returns", "BIMBOA.MX_returns"]
        self.assertEqual(list(self.df_returns.columns), expected_columns)

        # Should have fewer rows than original (due to pct_change removing first row)
        self.assertLess(len(self.df_returns), len(self.df))

    def test_returns_are_reasonable(self):
        """Test that calculated returns are within reasonable ranges."""
        walmex_returns = self.df_returns["WALMEX.MX_returns"].to_numpy()
        bimboa_returns = self.df_returns["BIMBOA.MX_returns"].to_numpy()

        # Daily returns should generally be between -30% and +30%
        self.assertTrue(np.all(walmex_returns > -0.3))
        self.assertTrue(np.all(walmex_returns < 0.3))
        self.assertTrue(np.all(bimboa_returns > -0.3))
        self.assertTrue(np.all(bimboa_returns < 0.3))

    def test_linear_model_produces_valid_results(self):
        """Test that linear regression model produces valid statistical results."""
        model_results = build_linear_model(
            self.df_returns, "WALMEX.MX_returns", "BIMBOA.MX_returns", test_size=30
        )

        # Check that all required metrics are present
        required_keys = ["model", "mse", "r2", "coefficient", "intercept"]
        for key in required_keys:
            self.assertIn(key, model_results)

        # R^2 should be between 0 and 1
        self.assertGreaterEqual(model_results["r2"], 0.0)
        self.assertLessEqual(model_results["r2"], 1.0)

        # MSE should be positive
        self.assertGreaterEqual(model_results["mse"], 0)

        # Coefficient and intercept should be finite numbers
        self.assertTrue(np.isfinite(model_results["coefficient"]))
        self.assertTrue(np.isfinite(model_results["intercept"]))

    def test_model_with_insufficient_data_raises_error(self):
        """Test that model building fails gracefully with insufficient data."""
        # Create minimal dataset
        small_data = self.df_returns.head(10)  # Only 10 rows

        with self.assertRaises(ValueError):
            build_linear_model(
                small_data, "WALMEX.MX_returns", "BIMBOA.MX_returns", test_size=30
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
