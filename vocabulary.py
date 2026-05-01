"""
Special tokens:
  PAD   (0) — padding token, tells the loss function to ignore this position
  START (1) — prepended to every caption sequence
  END   (2) — appended to every caption sequence
  UNK   (3) — replaces words below the frequency threshold

Creates a map between words in the config.TRAIN_ANN and indices, basically building a Tokenizer.
Which will be used for converting tokens to Sequences and vice versa.
"""

import re
import json
import pickle
from collections import Counter
from pathlib import Path

import nltk
from tqdm import tqdm

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

#     Special tokens
PAD_TOKEN   = "<PAD>"      # Padding token, will be used to make all sequences of same length.
START_TOKEN = "<START>"    # <START> token, will be prepended to every caption sequence
END_TOKEN   = "<END>"      # <END> token, will be appended to every caption sequence
UNK_TOKEN   = "<UNK>"      # <UNK> token, replaces words below the frequency threshold
SPECIAL_TOKENS = [PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN]


def clean_caption(text: str) -> str:
    """Lower case and clean the caption string."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return text.strip()


class Vocabulary:
    """
    Bidirectional word ↔ index mapping.
    """

    def __init__(self, threshold: int = 5):
        self.threshold = threshold
        self.word2idx: dict[str, int] = {}
        self.idx2word: dict[int, str] = {}
        self._add_special_tokens()

    # Add special tokens: START at the start of each caption and END at the end of each caption.
    def _add_special_tokens(self):
        """Add special tokens to the vocabulary."""
        for tok in SPECIAL_TOKENS:
            idx = len(self.word2idx)
            self.word2idx[tok] = idx
            self.idx2word[idx] = tok

    def _add_word(self, word: str):
        idx = len(self.word2idx)
        self.word2idx[word] = idx
        self.idx2word[idx]  = word


    # Building Vocabulary from Val2017 captions.
    def build_from_coco(self, ann_file: Path | str) -> "Vocabulary":

        ann_file = Path(ann_file)
        assert ann_file.exists(), f"Annotation file not found: {ann_file}"

        print(f"[Vocab] Reading {ann_file.name} …")
        with open(ann_file) as f:
            data = json.load(f)

        counter: Counter = Counter()
        for ann in tqdm(data["annotations"], desc="[Vocab] Tokenising"):
            tokens = nltk.tokenize.word_tokenize(clean_caption(ann["caption"]))
            counter.update(tokens)

        print(f"[Vocab] Unique tokens before threshold : {len(counter):,}")
        for word, freq in sorted(counter.items()):
            if freq >= self.threshold:
                self._add_word(word)

        print(f"[Vocab] Final vocabulary size          : {len(self):,}")
        return self


    def save(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[Vocab] Saved → {path}")

    @classmethod
    def load(cls, path: Path | str) -> "Vocabulary":
        with open(path, "rb") as f:
            vocab = pickle.load(f)
        print(f"[Vocab] Loaded ({len(vocab):,} words) from {path}")
        return vocab



    def __len__(self) -> int:
        return len(self.word2idx)

    def __getitem__(self, word: str) -> int:
        """Return the index for the word. if not in vocabulary, return UNK_TOKEN."""
        return self.word2idx.get(word, self.word2idx[UNK_TOKEN])

    def decode(self, indices: list[int], skip_special: bool = True) -> str:
        """
        Convert tokens to string. Stop at END_TOKEN.
        """
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, UNK_TOKEN)
            if word == END_TOKEN:
                break
            if skip_special and word in SPECIAL_TOKENS:
                continue
            words.append(word)
        return " ".join(words)



if __name__ == "__main__":
    import config
    vocab = Vocabulary(threshold=config.VOCAB_THRESHOLD).build_from_coco(config.TRAIN_ANN)
    vocab.save(config.VOCAB_PATH)
