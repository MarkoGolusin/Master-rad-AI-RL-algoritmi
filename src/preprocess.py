
import os

import numpy as np
import pandas as pd

from finrl.meta.preprocessor.preprocessors import FeatureEngineer

from config import (
    DATA_DIR,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TECHNICAL_INDICATORS,
    VOLATILITY_WINDOW
)


class DataPreprocessor:

    def __init__(self):

        self.raw_file = os.path.join(
            DATA_DIR,
            "raw_data.csv"
        )

    def load(self):

        print("=" * 60)
        print("Loading raw data...")
        print("=" * 60)

        df = pd.read_csv(self.raw_file)

        df["date"] = pd.to_datetime(df["date"])

        return df

    def feature_engineering(self, df):

        print("\nGenerating technical indicators...")

        fe = FeatureEngineer(

            use_technical_indicator=True,

            tech_indicator_list=TECHNICAL_INDICATORS,

            use_vix=False,

            use_turbulence=False,

            user_defined_feature=False

        )

        df = fe.preprocess_data(df)

        print("Technical indicators generated.")

        return df

    def add_returns(self, df):

        print("Calculating daily returns...")

        df["daily_return"] = (
            df.groupby("tic")["close"]
            .pct_change()
        )

        return df

    def add_atr(self, df, window=14):

        """
        Average True Range
        """

        print("Calculating ATR...")

        high = df["high"]
        low = df["low"]
        close = df["close"]

        previous_close = close.shift(1)

        tr1 = high - low

        tr2 = (high - previous_close).abs()

        tr3 = (low - previous_close).abs()

        true_range = pd.concat(
            [tr1, tr2, tr3],
            axis=1
        ).max(axis=1)

        df["ATR"] = (
            true_range
            .rolling(window)
            .mean()
        )

        return df

    def add_volatility(self, df):

        print("Calculating rolling volatility...")

        df["volatility"] = (

            df.groupby("tic")["daily_return"]

            .rolling(VOLATILITY_WINDOW)

            .std()

            .reset_index(level=0, drop=True)

        )

        return df

    def clean(self, df):

        print("Cleaning dataset...")

        df = df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        df = df.dropna()

        df = df.reset_index(drop=True)

        return df

    def split(self, df):

        print("Creating train/test datasets...")

        train = df[

            (df["date"] >= TRAIN_START_DATE)

            &

            (df["date"] < TRAIN_END_DATE)

        ]

        test = df[

            (df["date"] >= TEST_START_DATE)

            &

            (df["date"] <= TEST_END_DATE)

        ]

        return train, test

    def save(self, processed, train, test):

        processed.to_csv(

            os.path.join(DATA_DIR, "processed.csv"),

            index=False

        )

        train.to_csv(

            os.path.join(DATA_DIR, "train.csv"),

            index=False

        )

        test.to_csv(

            os.path.join(DATA_DIR, "test.csv"),

            index=False

        )

        print("\nFiles saved successfully.")

    def summary(self, train, test):

        print("\n")

        print("=" * 60)

        print("TRAIN SET")

        print("=" * 60)

        print(train.head())

        print()

        print(train.describe())

        print()

        print("=" * 60)

        print("TEST SET")

        print("=" * 60)

        print(test.head())

        print()

        print(test.describe())

        print()

        print("Train size:", len(train))

        print("Test size :", len(test))

        print("=" * 60)


def main():

    processor = DataPreprocessor()

    df = processor.load()

    df = processor.feature_engineering(df)

    df = processor.add_returns(df)

    df = processor.add_atr(df)

    df = processor.add_volatility(df)

    df = processor.clean(df)

    train, test = processor.split(df)

    processor.summary(train, test)

    processor.save(df, train, test)


if __name__ == "__main__":

    main()