import re
from dataclasses import dataclass, field

import contractions as contractions_lib
import spacy
import pandas as pd
from tqdm import tqdm

from utils import load_yaml_section, save_csv, setup_logger

logger = setup_logger()

nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])


@dataclass
class CleanTextConfig:
    input_train: str
    input_test: str
    train_output: str
    test_output: str
    text_columns: list[str]
    min_token_len: int = 2

    @classmethod
    def from_yaml(cls) -> "CleanTextConfig":
        return cls(**load_yaml_section("clean_text"))


def expand_contractions(text: str) -> str:
    return contractions_lib.fix(text)


def remove_html_and_urls(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    return text


def tag_negations(text: str) -> str:
    # "not good" / "never again" → "NOT_good" / "NOT_again"
    return re.sub(r"\b(not|never|no)\s+(\w+)", r"NOT_\2", text)


def extract_not_tokens(text: str) -> tuple[str, list[str]]:
    not_tokens = re.findall(r"NOT_\w+", text)
    text = re.sub(r"NOT_\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return text, not_tokens


def preprocess_text(text: str) -> tuple[str, list[str]]:
    """Steps before spaCy: lowercase, contractions, HTML/URLs, negation tagging."""
    text = text.lower()
    text = expand_contractions(text)
    text = remove_html_and_urls(text)
    text = tag_negations(text)
    return extract_not_tokens(text)


def load_split(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def clean_columns(
    df: pd.DataFrame, columns: list[str], min_token_len: int
) -> pd.DataFrame:
    for col in columns:
        if col not in df.columns:
            continue
        texts = df[col].fillna("").tolist()

        # Pre-process: contractions, HTML/URLs, negation tagging
        preprocessed, not_tokens_list = zip(*[preprocess_text(t) for t in texts])

        # Batch spaCy: lemmatize + remove stopwords
        results = []
        pipe = tqdm(
            nlp.pipe(preprocessed, batch_size=1000),
            total=len(texts),
            desc=f"  spaCy '{col}'",
        )
        for doc, not_tokens in zip(pipe, not_tokens_list):
            tokens = [
                token.lemma_
                for token in doc
                if not token.is_stop
                and not token.is_space
                and len(token.text) >= min_token_len
            ]
            results.append(" ".join(tokens + not_tokens))

        df[col] = results
    return df


def main():
    cfg = CleanTextConfig.from_yaml()

    for split_name, input_path, output_path in [
        ("train", cfg.input_train, cfg.train_output),
        ("test", cfg.input_test, cfg.test_output),
    ]:
        logger.info(f"\n=== Processing {split_name} set ===")

        # 1. Load
        logger.info("  === 1. Loading ===")
        df = load_split(input_path)
        logger.info(f"\tpath: {input_path}")
        logger.info(f"\trows: {len(df)}, cols: {len(df.columns)}")

        # 2. Clean text columns
        logger.info("  === 2. Cleaning text columns ===")
        logger.info(f"\tcolumns: {cfg.text_columns}")
        logger.info(
            f"\tsteps: lowercase → expand contractions → remove HTML/URLs → negation tagging → spaCy lemmatize + stopwords"
        )
        df = clean_columns(df, cfg.text_columns, cfg.min_token_len)

        # 3. Save
        logger.info("  === 3. Saving ===")
        save_csv(df, output_path)
        logger.info(f"\tpath: {output_path}")
        logger.info(f"\trows saved: {len(df)}")


if __name__ == "__main__":
    main()
