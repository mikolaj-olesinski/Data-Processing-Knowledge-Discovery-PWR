import glob
import os
from dataclasses import dataclass, field

import pandas as pd

from utils import load_yaml_section, save_csv, setup_logger

logger = setup_logger()

RATING_CATEGORY_MAP = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1}

CSV_OPTIONS = {"on_bad_lines": "skip", "encoding_errors": "replace", "engine": "python"}

REVIEWS_DUPLICATE_COLS = ["product_name", "brand_name", "price_usd"]


@dataclass
class PreprocessConfig:
    reviews_dir: str
    product_info: str
    output: str
    drop_columns: list[str]
    dropna_subsets: list[str] = field(default_factory=list)
    fill_unknown_cols: list[str] = field(default_factory=list)
    fill_empty_cols: list[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls) -> "PreprocessConfig":
        return cls(**load_yaml_section("preprocess"))


def load_reviews(reviews_dir: str, product_info_path: str) -> pd.DataFrame:
    review_files = sorted(glob.glob(os.path.join(reviews_dir, "reviews_*.csv")))
    reviews = pd.concat(
        [pd.read_csv(f, **CSV_OPTIONS) for f in review_files],
        ignore_index=True,
    )
    reviews["LABEL-rating"] = pd.to_numeric(
        reviews["LABEL-rating"], errors="coerce"
    ).astype("Int32")

    # drop before merge to avoid _x/_y duplicate columns
    reviews = reviews.drop(columns=REVIEWS_DUPLICATE_COLS, errors="ignore")

    products = pd.read_csv(product_info_path, **CSV_OPTIONS)
    return reviews.merge(products, on="product_id", how="left")


def add_rating_category(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    df[f"{target_col}-category"] = df[target_col].map(RATING_CATEGORY_MAP)
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    text = df["review_text"].fillna("")
    df["review_text_length"] = text.str.len()
    df["has_exclamation"] = text.str.contains("!", regex=False).astype(int)
    alpha = text.str.replace(r"[^a-zA-Z]", "", regex=True)
    df["caps_ratio"] = text.str.count(r"[A-Z]") / alpha.str.len().replace(0, 1)
    return df


def handle_missing(df: pd.DataFrame, cfg: "PreprocessConfig") -> pd.DataFrame:
    # Drop rows with NaN in critical columns
    before = len(df)
    existing_subsets = [c for c in cfg.dropna_subsets if c in df.columns]
    df = df.dropna(subset=existing_subsets)
    logger.info(
        f"\tdropna({existing_subsets}): {len(df)} rows (removed {before - len(df)})"
    )

    # Fill categorical NaN with "unknown"
    for col in cfg.fill_unknown_cols:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")

    # Fill text NaN with ""
    for col in cfg.fill_empty_cols:
        if col in df.columns:
            df[col] = df[col].fillna("")

    return df


def main():
    cfg = PreprocessConfig.from_yaml()

    # 1. Load and merge reviews with product info
    logger.info("=== 1. Loading data ===")
    df = load_reviews(cfg.reviews_dir, cfg.product_info)
    logger.info(f"\trows: {len(df)}, cols: {len(df.columns)}")

    # 2. Create sentiment target (ratings 1-2-3 → 0, ratings 4-5 → 1)
    logger.info("\n=== 2. Creating target label ===")
    df = add_rating_category(df, "LABEL-rating")
    logger.info(
        f"\tclass distribution:\n{df['LABEL-rating-category'].value_counts().sort_index().to_string()}"
    )

    # 3. Add engineered features
    logger.info("\n=== 3. Adding features ===")
    df = add_features(df)
    new_features = ["review_text_length", "has_exclamation", "caps_ratio"]
    logger.info(f"\tadded: {new_features}")

    # 4. Drop unused columns and deduplicate rows
    logger.info("\n=== 4. Dropping columns and deduplicating ===")
    before = len(df)
    df = df.drop(columns=cfg.drop_columns, errors="ignore").drop_duplicates()
    logger.info(f"\tdropped columns: {cfg.drop_columns}")
    logger.info(f"\trows after dedup: {len(df)} (removed {before - len(df)})")

    # 5. Drop rows without a valid target label
    logger.info("\n=== 5. Dropping rows with missing target ===")
    before = len(df)
    df = df.dropna(subset=["LABEL-rating-category"])
    logger.info(f"\trows after dropna: {len(df)} (removed {before - len(df)})")

    # 6. Handle missing values (drop critical, fill categorical/text)
    logger.info("\n=== 6. Handling missing values ===")
    df = handle_missing(df, cfg)

    # 7. Save
    logger.info("\n=== 7. Saving ===")
    logger.info(f"\tfinal shape: {len(df)} rows x {len(df.columns)} cols")
    logger.info(f"\tcolumns: {list(df.columns)}")
    save_csv(df, cfg.output)
    logger.info(f"\tsaved to {cfg.output}")


if __name__ == "__main__":
    main()
