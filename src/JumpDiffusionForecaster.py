import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional, Tuple

class JumpDiffusionForecaster:
    def __init__(self, data_path="data", dt=1/390, T=1/6.5, n_paths=1000):
        self.data_path = Path(data_path)
        self.dt = dt
        self.T = T
        self.n_paths = n_paths
        self.N = int(self.T / self.dt)
        self.df = pd.read_csv(self.data_path / "es_futures_1m_all.csv")
        self.df["Datetime"] = pd.to_datetime(self.df["Datetime"])
        self.df.set_index("Datetime", inplace=True)
        self.df["date"] = self.df.index.date
        self.df["time"] = self.df.index.time

    def _compute_overnight_returns(self, window_size=10) -> pd.Series:
        close_df = self.df[self.df["time"] == pd.to_datetime("16:00:00").time()][["Close"]].copy()
        open_df = self.df[self.df["time"] == pd.to_datetime("09:30:00").time()][["Close"]].copy()

        close_df["date"] = close_df.index.date
        open_df["date"] = open_df.index.date

        merged = pd.merge(
            close_df.rename(columns={"Close": "close"}),
            open_df.rename(columns={"Close": "open"}),
            on="date",
            how="inner"
        ).shift(-1).dropna()

        merged["overnight_return"] = np.log(merged["open"] / merged["close"])
        return merged["overnight_return"].tail(window_size)

    def simulate_paths(self, S0: float, mu: float, sigma: float, lambda_: float, jump_mean: float, jump_std: float) -> np.ndarray:
        paths = np.zeros((self.n_paths, self.N))
        paths[:, 0] = S0

        for i in range(self.n_paths):
            for t in range(1, self.N):
                dW = np.random.normal(0, np.sqrt(self.dt))
                dN = np.random.poisson(lambda_ * self.dt)
                J = -abs(np.random.normal(jump_mean, jump_std)) if dN else 0
                paths[i, t] = paths[i, t - 1] * np.exp((mu - 0.5 * sigma**2) * self.dt + sigma * dW + J)
        return paths

    def forecast_volatility(self, window_size=10, threshold=3, jump_std_scale=1.5, lambda_scale=2.0) -> Tuple[float, float, float, float, float]:
        log_returns = self._compute_overnight_returns(window_size)
        mu = log_returns.mean()
        sigma = log_returns.std()

        jumps = log_returns[np.abs(log_returns - mu) > threshold * sigma]
        lambda_ = (len(jumps) / len(log_returns)) * lambda_scale
        jump_mean = -abs(jumps.mean()) if len(jumps) > 0 else -0.0001
        jump_std = jumps.std() * jump_std_scale if len(jumps) > 0 else 0.0002

        latest_date = self.df["date"].max()
        intraday = self.df[(self.df["date"] == latest_date) & (self.df["time"] >= pd.to_datetime("09:30:00").time())]
        es_prices = intraday["Close"].values
        S0 = es_prices[0] if len(es_prices) > 0 else self.df["Close"].iloc[-1]

        paths = self.simulate_paths(S0, mu, sigma, lambda_, jump_mean, jump_std)
        log_returns_paths = np.log(paths[:, 1:] / paths[:, :-1])
        path_vols = np.std(log_returns_paths, axis=1) * np.sqrt(390)
        mean_vol = path_vols.mean()
        std_vol = path_vols.std()

        return mean_vol, std_vol, lambda_, jump_mean, jump_std

    def grid_search(self, n_samples=25, z_threshold=0.2) -> pd.DataFrame:
        log_returns = self._compute_overnight_returns()
        mu = log_returns.mean()
        sigma = log_returns.std()

        thresholds = np.random.uniform(2.0, 3.5, n_samples)
        std_scales = np.random.uniform(0.9, 1.6, n_samples)
        lambda_scales = np.random.uniform(1.0, 2.0, n_samples)

        latest_date = self.df["date"].max()
        intraday = self.df[(self.df["date"] == latest_date) & (self.df["time"] >= pd.to_datetime("09:30:00").time())]
        es_prices = intraday["Close"].values
        S0 = es_prices[0] if len(es_prices) > 0 else self.df["Close"].iloc[-1]

        results = []

        for th, s_scale, l_scale in zip(thresholds, std_scales, lambda_scales):
            jumps = log_returns[np.abs(log_returns - mu) > th * sigma]
            lambda_ = len(jumps) / len(log_returns) * l_scale
            jump_mean = -abs(jumps.mean()) if len(jumps) > 0 else -0.0001
            jump_std = jumps.std() * s_scale if len(jumps) > 0 else 0.0002

            paths = self.simulate_paths(S0, mu, sigma, lambda_, jump_mean, jump_std)
            lower_band = np.percentile(paths, 5, axis=0)
            upper_band = np.percentile(paths, 95, axis=0)

            if len(es_prices) >= self.N:
                es_prices_clipped = es_prices[:self.N]
                within_band = ((es_prices_clipped >= lower_band) & (es_prices_clipped <= upper_band)).mean()

                path_vols = np.std(np.log(paths[:, 1:] / paths[:, :-1]), axis=1) * np.sqrt(390)
                mean_vol = path_vols.mean()
                std_vol = path_vols.std()
                realized_vol = np.std(np.log(es_prices_clipped[1:] / es_prices_clipped[:-1])) * np.sqrt(390)
                z_score = (realized_vol - mean_vol) / std_vol

                results.append((th, s_scale, l_scale, within_band, z_score))

        return pd.DataFrame(results, columns=["Threshold", "StdScale", "LambdaScale", "Coverage", "ZScore"]).sort_values("ZScore", key=lambda x: x.abs())

    def plot_simulations(self, paths: np.ndarray, es_prices: Optional[np.ndarray] = None):
        N = paths.shape[1]
        plt.figure(figsize=(12, 6))
        for i in range(min(50, len(paths))):
            plt.plot(paths[i], linewidth=0.7, alpha=0.5, color="steelblue")

        lower_band = np.percentile(paths, 5, axis=0)
        upper_band = np.percentile(paths, 95, axis=0)
        plt.fill_between(range(N), lower_band, upper_band, color="cornflowerblue", alpha=0.2, label="5–95% Sim Band")

        if es_prices is not None and len(es_prices) >= N:
            plt.plot(es_prices[:N], linewidth=2.2, color="black", label="Actual ES Path")

        plt.title("Simulated vs Actual SPX Price Paths")
        plt.xlabel("Time Steps (1-min)")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
