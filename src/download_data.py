
import os
import pandas as pd
import yfinance as yf


from config import (
    TICKER,
    TRAIN_START_DATE,
    TEST_END_DATE,
    DATA_DIR
)


class DataDownloader:


    def __init__(self):

        self.start_date = TRAIN_START_DATE
        self.end_date = TEST_END_DATE
        self.ticker_list = TICKER


    def download(self):

        print("=" * 60)
        print("Downloading data from Yahoo Finance...")
        print("=" * 60)


        ticker = self.ticker_list[0]


        df = yf.download(
            ticker,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=False
        )


        print("\nDownload completed.")


        # yfinance returns MultiIndex columns

        if isinstance(df.columns, pd.MultiIndex):

            df.columns = df.columns.droplevel(1)


        df.reset_index(inplace=True)


        df.columns = [
            col.lower().replace(" ", "_")
            for col in df.columns
        ]


        if "adj_close" in df.columns:

            df = df.drop(
                columns=["adj_close"]
            )


        # FinRL format requires tic column

        df["tic"] = ticker


        return df

    @staticmethod
    def clean(df):

        print("\nCleaning data...")


        df = df.sort_values(
            ["date", "tic"]
        ).reset_index(drop=True)

        # Remove duplicates

        df = df.drop_duplicates()

        # Forward fill missing values

        df = df.ffill()

        df = df.dropna()

        print("Cleaning completed.")

        return df



    @staticmethod
    def save(df):

        path = os.path.join(
            DATA_DIR,
            "raw_data.csv"
        )


        df.to_csv(
            path,
            index=False
        )


        print(
            f"\nData saved to:\n{path}"
        )



    @staticmethod
    def summary(df):

        print("\n")
        print("=" * 60)
        print("DATA SUMMARY")
        print("=" * 60)

        print(df.head())
        print("\n")
        print(df.info())
        print("\nStatistics")
        print(df.describe())
        print("\nMissing values")
        print(df.isna().sum())

        print("=" * 60)



def main():

    downloader = DataDownloader()

    df = downloader.download()

    df = downloader.clean(df)

    downloader.summary(df)

    downloader.save(df)



if __name__ == "__main__":

    main()