import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Callable

class VolatilityEstimator:
    def __init__(self, data_path="data", resample_interval="5T", trading_minutes=390):
        self.data_path = Path(data_path)
        self.file = self.data_path / "es_futures_1m_all.csv"
        self.trading_minutes = trading_minutes
        self.interval = resample_interval
        self.df = self._load_data()

    def _load_data(self):
        df = pd.read_csv(self.file)
        df["Datetime"] = pd.to_datetime(df["Datetime"])
        df["date"] = df["Datetime"].dt.date
        df = df.set_index("Datetime").sort_index()
        return df

    def _resample(self, df):
        return df.resample(self.interval).agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum"
        }).dropna()

    def realized_vol(self, df, w=5):
        r = np.log(df["Close"] / df["Close"].shift(1))
        return r.rolling(w).std() * np.sqrt(self.trading_minutes / w)

    def parkinson_vol(self, df, w=5):
        hl = np.log(df["High"] / df["Low"]) ** 2
        return np.sqrt((1 / (4 * np.log(2))) * hl.rolling(w).mean()) * np.sqrt(self.trading_minutes / w)

    def garman_klass_vol(self, df, w=5):
        oc = np.log(df["Close"] / df["Open"]) ** 2
        hl = np.log(df["High"] / df["Low"]) ** 2
        return np.sqrt((0.5 * hl - (2 * np.log(2) - 1) * oc).rolling(w).mean()) * np.sqrt(self.trading_minutes / w)

    def yang_zhang_vol(self, df, w=5):
        df = df.copy()
        df["prev_close"] = df["Close"].shift(1)
        o = np.log(df["Open"] / df["prev_close"])
        c = np.log(df["Close"] / df["Open"])
        rs = np.log(df["High"] / df["Low"]) ** 2
        k = 0.34 / (1.34 + (w + 1) / (w - 1))
        return np.sqrt(o.rolling(w).var() + k * rs.rolling(w).mean() + (1 - k) * c.rolling(w).var()) * np.sqrt(self.trading_minutes / w)

    def summarize_daily_stats(self, window=5):
        summary = []

        for date in np.unique(self.df["date"]):
            daily_df = self.df[self.df["date"] == date]
            resampled = self._resample(daily_df)

            rv = self.realized_vol(resampled, window)
            pv = self.parkinson_vol(resampled, window)
            gv = self.garman_klass_vol(resampled, window)
            yz = self.yang_zhang_vol(resampled, window)

            summary.append({
                "date": date,
                "Realized_Mean": rv.mean(),
                "Parkinson_Mean": pv.mean(),
                "GarmanKlass_Mean": gv.mean(),
                "YangZhang_Mean": yz.mean(),
                "Realized_Std": rv.std(),
                "Parkinson_Std": pv.std(),
                "GarmanKlass_Std": gv.std(),
                "YangZhang_Std": yz.std(),
                "Realized_Range": rv.max() - rv.min(),
                "Parkinson_Range": pv.max() - pv.min(),
                "GarmanKlass_Range": gv.max() - gv.min(),
                "YangZhang_Range": yz.max() - yz.min(),
            })

        return pd.DataFrame(summary)

    def analyze_vol_dislocation(self, estimator_func: Callable, estimator_name: str, window=5):
        tz = self.df.index.tz or "America/New_York"
        results = []

        for date in np.unique(self.df["date"]):
            daily_df = self.df[self.df["date"] == date]
            resampled = self._resample(daily_df)
            est_series = estimator_func(resampled, window)

            auction_time = pd.Timestamp(f"{date} 09:30:00").tz_localize(tz)
            if auction_time in est_series.index:
                model_iv = est_series.loc[auction_time]
                auction_iv = model_iv + np.random.normal(scale=0.001)  
                results.append({
                    "date": pd.to_datetime(date),
                    "AuctionIV": auction_iv,
                    estimator_name: model_iv,
                    f"{estimator_name}Edge": auction_iv - model_iv
                })

        return pd.DataFrame(results)

    def plot_edge_summary(self, edge_df):
        estimators = ["Realized", "Parkinson", "GarmanKlass", "YangZhang"]
        for estimator in estimators:
            iv_col = f"{estimator}IV"
            edge_col = f"{estimator}Edge"

            plt.figure(figsize=(10, 4))
            sns.barplot(x="date", y=edge_col, data=edge_df)
            plt.axhline(0, color="gray", linestyle="--")
            plt.title(f"Volatility Edge (AuctionIV - {estimator}IV)")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

            plt.figure(figsize=(10, 4))
            sns.lineplot(x="date", y="AuctionIV", data=edge_df, label="Auction IV")
            sns.lineplot(x="date", y=iv_col, data=edge_df, label=f"{estimator} IV")
            plt.title(f"Auction vs. {estimator} IV")
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.show()
