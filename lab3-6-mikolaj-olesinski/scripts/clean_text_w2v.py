"""Minimal text cleaning for Word2Vec input.

Steps: lowercase → remove HTML/URLs → remove special chars.
No lemmatization, no stopword removal, no negation tagging.
"""

import re
from dataclasses import dataclass

import pandas as pd
from tqdm import tqdm

from utils import load_yaml_section, save_csv, setup_logger

logger = setup_logger()


@dataclass
class CleanTextW2VConfig:
    input_train: str
    input_test: str
    train_output: str
    test_output: str
    text_columns: list[str]

    @classmethod
    def from_yaml(cls) -> "CleanTextW2VConfig":
        return cls(**load_yaml_section("clean_text_w2v"))


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)  # remove HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)  # keep only letters and spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            continue
        texts = df[col].fillna("").tolist()
        df[col] = [clean_text(t) for t in tqdm(texts, desc=f"  cleaning '{col}'")]
    return df


def main():
    cfg = CleanTextW2VConfig.from_yaml()

    for split_name, input_path, output_path in [
        ("train", cfg.input_train, cfg.train_output),
        ("test", cfg.input_test, cfg.test_output),
    ]:
        logger.info(f"\n=== Processing {split_name} set ===")

        logger.info("  === 1. Loading ===")
        df = pd.read_csv(input_path, low_memory=False)
        logger.info(f"\tpath: {input_path}")
        logger.info(f"\trows: {len(df)}, cols: {len(df.columns)}")

        logger.info("  === 2. Cleaning text columns ===")
        logger.info(f"\tcolumns: {cfg.text_columns}")
        logger.info(f"\tsteps: lowercase → remove HTML/URLs → remove special chars")
        df = clean_columns(df, cfg.text_columns)

        logger.info("  === 3. Saving ===")
        save_csv(df, output_path)
        logger.info(f"\tpath: {output_path}")
        logger.info(f"\trows saved: {len(df)}")


if __name__ == "__main__":
    main()
