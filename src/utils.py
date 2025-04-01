import pandas as pd 

def get_es_slice(auction_date):
    t_minus_1 = auction_date - pd.Timedelta(days=1)
    start = pd.Timestamp(f"{t_minus_1.date()} 16:00", tz="America/New_York")
    end = pd.Timestamp(f"{auction_date.date()} 09:30", tz="America/New_York")
    return es_futures_df.loc[(es_futures_df.index >= start) & (es_futures_df.index <= end)].copy()