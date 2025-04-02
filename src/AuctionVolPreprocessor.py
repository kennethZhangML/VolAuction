import pandas as pd
import numpy as np
from pathlib import Path

class AuctionVolPreprocessor:
    def __init__(self, data_path="data"):
        self.data_path = Path(data_path)
        self.futures_file = self.data_path / "es_futures_1m_all.csv"
        self.spx_file = self.data_path / "spx_spot_daily.csv"
        self.atm_file = self.data_path / "atm_strike_map.csv"
        self.quotes_file = self.data_path / "auction_daily_quotes.parquet"

    def _load_and_align_timezones(self):
        print("Loading and aligning timezones...")

        es_futures = pd.read_csv(self.futures_file, index_col=0, parse_dates=True)
        es_futures.index = pd.to_datetime(es_futures.index).tz_localize("UTC").tz_convert("America/New_York")

        spx_spot = pd.read_csv(self.spx_file, parse_dates=["date"])
        spx_spot["date"] = pd.to_datetime(spx_spot["date"]).dt.tz_localize("UTC").dt.tz_convert("America/New_York")

        atm_strike_map = pd.read_csv(self.atm_file, parse_dates=["date"])
        atm_strike_map["date"] = pd.to_datetime(atm_strike_map["date"]).dt.tz_localize("UTC").dt.tz_convert("America/New_York")

        auction_quotes = pd.read_parquet(self.quotes_file)
        auction_quotes["date"] = pd.to_datetime(auction_quotes["date"]).dt.tz_localize("UTC").dt.tz_convert("America/New_York")

        return es_futures, spx_spot, atm_strike_map, auction_quotes

    def compute_overnight_realized_vol(self, es_futures, spx_spot, auction_quotes):
        print("Computing overnight realized volatility...")
        results = []

        for auction_day in auction_quotes["date"]:
            auction_day = pd.Timestamp(auction_day)
            t_minus_1 = auction_day - pd.Timedelta(days=1)

            spx_before = spx_spot[spx_spot["date"] < auction_day]
            spx_close_t_minus_1 = spx_before.iloc[-1]["Close"] if not spx_before.empty else None

            t1_str = t_minus_1.strftime("%Y-%m-%d")
            t_str = auction_day.strftime("%Y-%m-%d")
            es_start = pd.Timestamp(f"{t1_str} 18:00", tz="America/New_York")
            es_end = pd.Timestamp(f"{t_str} 09:30", tz="America/New_York")

            es_window = es_futures.loc[(es_futures.index >= es_start) & (es_futures.index <= es_end)]
            log_returns = np.log(es_window["Close"] / es_window["Close"].shift(1)).dropna()
            realized_std = log_returns.std() * np.sqrt(len(log_returns))

            results.append({
                "date": auction_day,
                "spx_close_t_minus_1": spx_close_t_minus_1,
                "realized_std": realized_std,
                "n_ticks": len(log_returns)
            })

        overnight_df = pd.DataFrame(results)
        overnight_df.to_csv(self.data_path / "overnight_realized_vol.csv", index=False)
        return overnight_df

    def run(self):
        es_futures, spx_spot, atm_strike_map, auction_quotes = self._load_and_align_timezones()
        overnight_df = self.compute_overnight_realized_vol(es_futures, spx_spot, auction_quotes)
        return overnight_df
