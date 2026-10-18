"""
inference.py — Generate captions for a single image or a folder of images.

Supports both greedy decoding and beam-search decoding.
Prints captions to stdout and optionally saves annotated images.

Usage
=====
    # Caption one image
    python inference.py --image path/to/photo.jpg \
                        --checkpoint checkpoints/v1/best_model.pth

    # Caption every image in a folder
    python inference.py --folder path/to/images/ \
                        --checkpoint checkpoints/v1/best_model.pth \
                        --beam_width 5 \
                        --save_dir results/predictions/
"""

import argparse
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

import config
from vocabulary import Vocabulary
from models.v1  import CNNEncoder, LSTMDecoder
from utils      import load_checkpoint


#     Image preprocessing

_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(config.IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
])


def load_image(path: Path | str) -> torch.Tensor:
    
    img = Image.open(path).convert("RGB")
    return _transform(img).unsqueeze(0)


#     Caption generation

@torch.no_grad()
def caption_image(
    image_path  : Path | str,
    encoder     : CNNEncoder,
    decoder     : LSTMDecoder,
    vocab       : Vocabulary,
    device      : torch.device,
    beam_width  : int = 5,
) -> str:
    
    encoder.eval()
    decoder.eval()

    image    = load_image(image_path).to(device)
    features = encoder(image)

    if beam_width == 1:
        tokens = decoder.generate_greedy(features, vocab,
                                         max_length=config.MAX_SEQ_LEN)
    else:
        tokens = decoder.generate_beam(features, vocab,
                                       beam_width=beam_width,
                                       max_length=config.MAX_SEQ_LEN)

    return vocab.decode(tokens)


#     Visualisation

def save_captioned_image(
    image_path : Path | str,
    caption    : str,
    save_path  : Path | str,
):
    """Overlay the generated caption onto the image and save it."""
    img = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(caption, fontsize=13, wrap=True,
                 bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.8))
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


#     CLI

def parse_args():
    p = argparse.ArgumentParser(description="Generate captions with the V1 CNN+LSTM model.")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",  type=str, help="Path to a single image file.")
    group.add_argument("--folder", type=str, help="Directory of images to caption.")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to checkpoint (.pth) produced by train.py")
    p.add_argument("--beam_width", type=int, default=config.BEAM_WIDTH)
    p.add_argument("--save_dir",   type=str, default=None,
                   help="If given, save annotated images here.")
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Inference] Device: {device}")

    #     Load
    vocab = Vocabulary.load(config.VOCAB_PATH)

    encoder = CNNEncoder(embed_dim=config.EMBED_DIM, fine_tune=False).to(device)
    decoder = LSTMDecoder(
        vocab_size = len(vocab),
        embed_dim  = config.EMBED_DIM,
        hidden_dim = config.HIDDEN_DIM,
        num_layers = config.NUM_LAYERS,
        dropout    = 0.0,
    ).to(device)

    state = load_checkpoint(args.checkpoint, device)
    encoder.load_state_dict(state["encoder"])
    decoder.load_state_dict(state["decoder"])


    if args.image:
        image_paths = [Path(args.image)]
    else:
        folder = Path(args.folder)
        image_paths = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        print(f"[Inference] Found {len(image_paths)} images in {folder}")

    #     Generate
    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        caption = caption_image(img_path, encoder, decoder, vocab,
                                device, beam_width=args.beam_width)
        print(f"  {img_path.name:40s} → {caption}")

        if save_dir:
            out_path = save_dir / f"{img_path.stem}_caption.png"
            save_captioned_image(img_path, caption, out_path)


if __name__ == "__main__":
    main()
