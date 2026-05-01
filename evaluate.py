"""
Evaluation script which uses the best_model.pth to generate captions and evaluate the model using the corpus BLEU scores.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm

import config
from dataset    import CocoDataset, collate_fn
from vocabulary import Vocabulary
from models.v1  import CNNEncoder, LSTMDecoder
from utils      import load_checkpoint



@torch.no_grad()
def evaluate(
    encoder    : CNNEncoder,
    decoder    : LSTMDecoder,
    loader     : DataLoader,
    vocab      : Vocabulary,
    dataset    : CocoDataset,
    device     : torch.device,
    beam_width : int = 5,
) -> dict[str, float]:
    """
    Run beam-search on every validation image and compute corpus BLEU scores.

    """
    encoder.eval()
    decoder.eval()

    hypotheses : list[list[str]] = []
    references : list[list[list[str]]] = []

    val_dataset = loader.dataset
    for idx in tqdm(range(len(val_dataset)), desc="[Eval] Generating captions"):
        image, _, _ = val_dataset[idx]
        image       = image.unsqueeze(0).to(device)

        features = encoder(image)
        tokens   = decoder.generate_beam(features, vocab,
                                         beam_width=beam_width,
                                         max_length=config.MAX_SEQ_LEN)
        hyp = [vocab.idx2word.get(t, "<UNK>") for t in tokens]

        # All 5 reference captions, tokenised
        raw_refs = val_dataset.get_all_captions(idx)
        refs     = [ref.lower().split() for ref in raw_refs]

        hypotheses.append(hyp)
        references.append(refs)

    smoother = SmoothingFunction().method1

    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0),             smoothing_function=smoother)
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0),         smoothing_function=smoother)
    bleu3 = corpus_bleu(references, hypotheses, weights=(1/3, 1/3, 1/3, 0),       smoothing_function=smoother)
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)

    return {
        "bleu1": round(bleu1 * 100, 2),
        "bleu2": round(bleu2 * 100, 2),
        "bleu3": round(bleu3 * 100, 2),
        "bleu4": round(bleu4 * 100, 2),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate a V1 captioning model on COCO val.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a .pth checkpoint produced by train.py")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Device: {device}")

    vocab = Vocabulary.load(config.VOCAB_PATH)

    val_dataset = CocoDataset(
        img_dir = config.VAL_IMG_DIR,
        ann_file= config.VAL_ANN,
        vocab   = vocab,
        split   = "val",
        subset  = 1.0,
    )

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

    dummy_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    scores = evaluate(encoder, decoder, dummy_loader, vocab, val_dataset,
                      device, beam_width=config.BEAM_WIDTH)

    print("\n" + "=" * 40)
    print("  Evaluation Results (V1 CNN + LSTM)")
    print("=" * 40)
    for k, v in scores.items():
        print(f"  {k.upper():6s}: {v:.2f}")
    print("=" * 40)

    out_path = config.RESULTS_DIR / "v1_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"[Eval] Results saved → {out_path}")


if __name__ == "__main__":
    main()
