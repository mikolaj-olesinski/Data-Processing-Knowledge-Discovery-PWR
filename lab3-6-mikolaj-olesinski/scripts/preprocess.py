import glob
import os
from dataclasses import dataclass

import pandas as pd
import yaml

RATING_CATEGORY_MAP = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}

CSV_OPTIONS = {"on_bad_lines": "skip", "encoding_errors": "replace", "engine": "python"}

REVIEWS_DUPLICATE_COLS = ["product_name", "brand_name", "price_usd"]


@dataclass
class PreprocessConfig:
    reviews_dir: str
    product_info: str
    output: str
    drop_columns: list[str]

    @classmethod
    def from_yaml(cls) -> "PreprocessConfig":
        with open("params.yaml") as f:
            cfg = yaml.safe_load(f)["preprocess"]
        return cls(**cfg)


def load_reviews(reviews_dir: str, product_info_path: str) -> pd.DataFrame:
    review_files = sorted(glob.glob(os.path.join(reviews_dir, "reviews_*.csv")))
    reviews = pd.concat(
        [pd.read_csv(f, **CSV_OPTIONS) for f in review_files],
        ignore_index=True,
    )
    reviews["LABEL-rating"] = pd.to_numeric(reviews["LABEL-rating"], errors="coerce").astype("Int32")

    # drop before merge to avoid _x/_y duplicate columns
    reviews = reviews.drop(columns=REVIEWS_DUPLICATE_COLS, errors="ignore")

    products = pd.read_csv(product_info_path, **CSV_OPTIONS)
    return reviews.merge(products, on="product_id", how="left")


def add_rating_category(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df[f"{target_col}-category"] = df[target_col].map(RATING_CATEGORY_MAP)
    return df


def main():
    cfg = PreprocessConfig.from_yaml()

    df = load_reviews(cfg.reviews_dir, cfg.product_info)
    print(f"Loaded:          {len(df)} rows")

    df = add_rating_category(df, "LABEL-rating")

    before = len(df)
    df = df.drop(columns=cfg.drop_columns, errors="ignore").drop_duplicates()
    print(f"After drop/dedup:{len(df)} rows  (dropped {before - len(df)})")

    before = len(df)
    df = df.dropna(subset=["LABEL-rating-category"])
    print(f"After dropna:    {len(df)} rows  (dropped {before - len(df)})")

    print(f"\nClass distribution:\n{df['LABEL-rating-category'].value_counts().sort_index()}")
    print(f"\nColumns ({len(df.columns)}):\n{df.dtypes.to_string()}")

    os.makedirs(os.path.dirname(cfg.output), exist_ok=True)
    df.to_csv(cfg.output, index=False)
    print(f"\nSaved {len(df)} rows to {cfg.output}")


if __name__ == "__main__":
    main()