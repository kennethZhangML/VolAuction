import time
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

class ESFuturesFetcher:
    def __init__(self, start_date, end_date, save_path="data"):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

    def fetch_spx_daily(self):
        print("Fetching SPX daily spot data...")
        spx = yf.Ticker("^GSPC")
        spx_hist = spx.history(start=self.start_date, end=self.end_date, interval="1d")
        spx_hist = spx_hist[["Open", "Close", "High", "Low", "Volume"]].reset_index()
        spx_hist.rename(columns={"Date": "date"}, inplace=True)
        spx_hist.to_csv(self.save_path / "spx_spot_daily.csv", index=False)
        return spx_hist

    def fetch_es_1min(self, sleep_time=1):
        print("Fetching ES 1-minute futures data...")
        current = self.start_date
        all_data = []
        es = yf.Ticker("ES=F")

        while current <= self.end_date:
            print(f"→ {current.date()}")

            try:
                hist = es.history(period="7d", interval="1m") 
                filtered = hist[hist.index.date == current.date()]
                if not filtered.empty:
                    filtered = filtered.copy()
                    filtered["date"] = current.date()
                    all_data.append(filtered)
            except Exception as e:
                print(f"Error on {current.date()}: {e}")

            current += timedelta(days=1)
            time.sleep(sleep_time)

        df_all = pd.concat(all_data)
        df_all.to_csv(self.save_path / "es_futures_1m_all.csv")
        print("Saved ES data to es_futures_1m_all.csv")
        return df_all

    def run(self):
        spx = self.fetch_spx_daily()
        es_1m = self.fetch_es_1min()
        return spx, es_1m
