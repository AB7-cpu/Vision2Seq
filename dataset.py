"""
Loading and getting data ready.
"""

import json
import random
from pathlib import Path

import nltk
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import config
from vocabulary import Vocabulary, START_TOKEN, END_TOKEN, PAD_TOKEN, clean_caption


#     Transformations:

def get_transform(split: str) -> transforms.Compose:
    """
    Transforms image for better generalization.
    """
    common_end = [
        transforms.ToTensor(),
        transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ]
    if split == "train":
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(config.IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            *common_end,
        ])
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.IMAGE_SIZE),
        *common_end,
    ])


#     Dataset

class CocoDataset(Dataset):

    def __init__(
        self,
        img_dir    : Path | str,                     # Image directory. (train2017/)
        ann_file   : Path | str,                     # Captions location (train2017.json)
        vocab      : Vocabulary,                     # Prebuilt Vocabulary.
        split      : str   = "train",                # "train" or "val" for deciding the image augmentation.
        max_seq_len: int   = config.MAX_SEQ_LEN,     # Max embedding length.
        subset     : float = 1.0,                    # Percentage of the data to be used.
    ):
        self.img_dir     = Path(img_dir)
        self.vocab       = vocab
        self.split       = split
        self.max_seq_len = max_seq_len
        self.transform   = get_transform(split)


        with open(ann_file) as f:
            data = json.load(f)

        id_to_file: dict[int, str] = {img["id"]: img["file_name"] for img in data["images"]}


        caps_by_id: dict[int, list[str]] = {}
        for ann in data["annotations"]:
            caps_by_id.setdefault(ann["image_id"], []).append(ann["caption"])


        self.samples: list[tuple[str, list[str]]] = [
            (id_to_file[img_id], caps)
            for img_id, caps in caps_by_id.items()
            if img_id in id_to_file
        ]


        if subset < 1.0:
            n = max(1, int(len(self.samples) * subset))
            self.samples = random.sample(self.samples, n)

        print(f"[Dataset] {split.upper():5s} | {len(self.samples):,} images loaded.")



    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        filename, captions = self.samples[idx]

        image = Image.open(self.img_dir / filename).convert("RGB")
        image = self.transform(image)


        caption = random.choice(captions) if self.split == "train" else captions[0]
        tokens  = nltk.tokenize.word_tokenize(clean_caption(caption))

        # Build index sequence: <START> tok1 tok2 … tokN <END>
        indices = (
            [self.vocab[START_TOKEN]]
            + [self.vocab[t] for t in tokens]
            + [self.vocab[END_TOKEN]]
        )

        indices = indices[: self.max_seq_len]
        length  = len(indices)
        indices += [self.vocab[PAD_TOKEN]] * (self.max_seq_len - length)

        return image, torch.tensor(indices, dtype=torch.long), length

    def get_all_captions(self, idx: int) -> list[str]:
        """Return all 5 ground-truth captions for image at `idx` (will be used in BLEU eval)."""
        return self.samples[idx][1]



def collate_fn(batch):

    images, captions, lengths = zip(*batch)
    return (
        torch.stack(images,   dim=0),
        torch.stack(captions, dim=0),
        torch.tensor(lengths, dtype=torch.long),
    )
