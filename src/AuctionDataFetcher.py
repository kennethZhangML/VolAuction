import requests
import pandas as pd
from datetime import datetime, timedelta, time
from pathlib import Path

class AuctionDataFetcher:
    def __init__(self, api_base="http://127.0.0.1:25510", save_path="data"):
        self.api_base = api_base
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.option_root = "SPXW"
        self.ivl = 12000 
        self.atm_strike_map = {}

    def _fetch_expirations(self, days_back=63):
        url = f"{self.api_base}/v2/list/expirations?root={self.option_root}"
        response = requests.get(url).json()["response"]
        today = datetime.today()
        cutoff = today - timedelta(days=days_back)
        return [str(exp) for exp in response if cutoff <= datetime.strptime(str(exp), "%Y%m%d") <= today]

    def _fetch_daily_quotes(self, expirations):
        all_quotes = []
        for exp in expirations:
            params = {
                "root": self.option_root,
                "exp": exp,
                "start_date": exp,
                "end_date": exp,
                "ivl": self.ivl
            }
            r = requests.get(f"{self.api_base}/v2/bulk_hist/option/quote", params=params)
            data = r.json()["response"]
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(exp)
            all_quotes.append(df)
        df_all = pd.concat(all_quotes, ignore_index=True)
        df_all["date"] = pd.to_datetime(df_all["date"])
        df_all.set_index("date", inplace=True)
        return df_all

    def _flatten_ticks(self, df_all):
        rows = []
        for idx, row in df_all.iterrows():
            contract = row["contract"]
            for tick in row["ticks"]:
                rows.append({
                    "timestamp": tick[0],
                    "bid_sz": tick[1],
                    "bid_px": tick[3],
                    "ask_sz": tick[5],
                    "ask_px": tick[7],
                    "strike": contract["strike"],
                    "right": contract["right"],
                    "expiration": contract["expiration"],
                    "date": idx
                })
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df["expiration"] = pd.to_datetime(df["expiration"])
        return df

    def _extract_0dte_auction(self, df_flat):
        df_0dte = df_flat[df_flat["date"] == df_flat["expiration"]].copy()
        df_0dte["timestamp"] = df_0dte["timestamp"].apply(
            lambda x: (datetime.min + timedelta(milliseconds=x)).time()
        )
        window_start = (datetime.combine(datetime.today(), time(9, 30)) - timedelta(minutes=5)).time()
        window_end = (datetime.combine(datetime.today(), time(9, 30)) + timedelta(minutes=5)).time()
        df_auction = df_0dte[(df_0dte["timestamp"] >= window_start) & (df_0dte["timestamp"] <= window_end)].copy()
        df_auction["mid_px"] = (df_auction["bid_px"] + df_auction["ask_px"]) / 2
        df_auction = df_auction[df_auction["mid_px"] > 0]
        return df_auction

    def _find_atm_quotes(self, df_auction):
        results = []
        for date, group in df_auction.groupby("date"):
            group["spread"] = group["ask_px"] - group["bid_px"]
            best = group.loc[group["spread"].idxmin()]
            atm_strike = best["strike"]
            atm_call = group[(group["strike"] == atm_strike) & (group["right"] == "C")]
            atm_put = group[(group["strike"] == atm_strike) & (group["right"] == "P")]

            if not atm_call.empty and not atm_put.empty:
                results.append({
                    "date": date,
                    "strike": atm_strike,
                    "call_mid": atm_call["mid_px"].values[0],
                    "put_mid": atm_put["mid_px"].values[0],
                    "call_bid": atm_call["bid_px"].values[0],
                    "call_ask": atm_call["ask_px"].values[0],
                    "put_bid": atm_put["bid_px"].values[0],
                    "put_ask": atm_put["ask_px"].values[0],
                })
                self.atm_strike_map[date] = atm_strike

        df_daily = pd.DataFrame(results).sort_values("date")
        df_daily.to_parquet(self.save_path / "auction_daily_quotes.parquet", index=False)
        pd.DataFrame([
            {"date": k, "strike": v} for k, v in self.atm_strike_map.items()
        ]).to_csv(self.save_path / "atm_strike_map.csv", index=False)
        return df_daily

    def _save_intraday_quotes(self, df_flat):
        all_intraday = []
        by_day_path = self.save_path / "intraday_by_day"
        by_day_path.mkdir(parents=True, exist_ok=True)

        for date in df_flat["date"].unique():
            strike = self.atm_strike_map.get(date)
            if not strike:
                continue
            df_day = df_flat[(df_flat["date"] == date) &
                             (df_flat["strike"] == strike) &
                             (df_flat["expiration"] == date)]
            df_day["mid_px"] = (df_day["bid_px"] + df_day["ask_px"]) / 2
            df_day["time"] = pd.to_timedelta(df_day["timestamp"], unit="ms")
            all_intraday.append(df_day)
            df_day.to_parquet(by_day_path / f"intraday_quotes_{date.strftime('%Y-%m-%d')}.parquet", index=False)

        df_all = pd.concat(all_intraday)
        df_all.to_parquet(self.save_path / "all_intraday_quotes.parquet", index=False)
        return df_all

    def run(self):
        expirations = self._fetch_expirations()
        df_all = self._fetch_daily_quotes(expirations)
        df_flat = self._flatten_ticks(df_all)
        df_auction = self._extract_0dte_auction(df_flat)
        df_daily_quotes = self._find_atm_quotes(df_auction)
        df_intraday = self._save_intraday_quotes(df_flat)
        return df_daily_quotes, df_intraday
