import pandas as pd
from pathlib import Path

class ESOvernightWindowExtractor:
    def __init__(self, data_path="data"):
        self.data_path = Path(data_path)
        self.auction_file = self.data_path / "auction_daily_quotes.parquet"
        self.es_file = self.data_path / "es_futures_1m_all.csv"
        self.timezone = "America/New_York"
        self._load_data()

    def _load_data(self):
        self.auction_quotes = pd.read_parquet(self.auction_file)
        self.auction_quotes["date"] = pd.to_datetime(self.auction_quotes["date"]).dt.tz_localize(None)
        self.auction_dates = pd.to_datetime(self.auction_quotes["date"]).dt.tz_localize(None)

        self.es_futures = pd.read_csv(self.es_file, index_col=0, parse_dates=True)
        self.es_futures.index = pd.to_datetime(self.es_futures.index).tz_localize("UTC").tz_convert(self.timezone)

    def get_es_slice(self, auction_date):
        if auction_date.tzinfo is None:
            auction_date = auction_date.tz_localize(self.timezone)

        t_minus_1 = auction_date - pd.Timedelta(days=1)
        start = pd.Timestamp(f"{t_minus_1.date()} 16:00", tz=self.timezone)
        end = pd.Timestamp(f"{auction_date.date()} 09:30", tz=self.timezone)

        return self.es_futures.loc[(self.es_futures.index >= start) & (self.es_futures.index <= end)].copy()

    def get_all_slices(self):
        slices = {}
        for date in self.auction_dates:
            auction_date = pd.Timestamp(date).tz_localize(self.timezone)
            es_window = self.get_es_slice(auction_date)
            slices[auction_date.strftime("%Y-%m-%d")] = es_window
        return slices
