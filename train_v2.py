"""
Training script for V2 (CNN + Attention + LSTM) image captioning.
"""

import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler
from tqdm import tqdm

import config
from dataset    import CocoDataset, collate_fn
from vocabulary import Vocabulary
from models.v2  import AttentionEncoder, AttentionDecoder
from evaluate   import evaluate
from utils      import (AverageMeter, save_checkpoint, load_checkpoint,
                        plot_training_curves, save_history, get_lr, format_time)


#  One epoch of training

def train_one_epoch(
    encoder  : AttentionEncoder,
    decoder  : AttentionDecoder,
    loader   : DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimiser: torch.optim.Optimizer,
    scaler   : GradScaler,
    device   : torch.device,
    epoch    : int,
) -> float:

    encoder.train()
    decoder.train()

    meter = AverageMeter("Train Loss")
    pbar  = tqdm(loader, desc=f"Epoch {epoch:02d} [Train]", leave=False)

    for step, (images, captions, lengths) in enumerate(pbar):
        images   = images.to(device, non_blocking=True)
        captions = captions.to(device, non_blocking=True)

        optimiser.zero_grad()

        with torch.autocast(device_type=device.type):
            features        = encoder(images)
            logits, alphas  = decoder(features, captions)

            targets = captions[:, 1:].reshape(-1)
            logits  = logits.reshape(-1, decoder.vocab_size)
            ce_loss = criterion(logits, targets)

            att_reg = config.V2_ALPHA_C * ((1.0 - alphas.sum(dim=1)) ** 2).mean()

            loss = ce_loss + att_reg

        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()),
            config.GRAD_CLIP,
        )
        scaler.step(optimiser)
        scaler.update()

        meter.update(loss.item(), images.size(0))

        if step % config.LOG_INTERVAL == 0:
            pbar.set_postfix({"loss": f"{meter.avg:.4f}", "lr": f"{get_lr(optimiser):.2e}"})

    return meter.avg


#  One epoch of validation

@torch.no_grad()
def val_loss_epoch(
    encoder  : AttentionEncoder,
    decoder  : AttentionDecoder,
    loader   : DataLoader,
    criterion: nn.CrossEntropyLoss,
    device   : torch.device,
    epoch    : int,
) -> float:

    encoder.eval()
    decoder.eval()

    meter = AverageMeter("Val Loss")
    for images, captions, lengths in tqdm(loader, desc=f"Epoch {epoch:02d} [Val]", leave=False):
        images   = images.to(device)
        captions = captions.to(device)

        with torch.autocast(device_type=device.type):
            features        = encoder(images)
            logits, alphas  = decoder(features, captions)

            targets = captions[:, 1:].reshape(-1)
            logits  = logits.reshape(-1, decoder.vocab_size)
            ce_loss = criterion(logits, targets)
            att_reg = config.V2_ALPHA_C * ((1.0 - alphas.sum(dim=1)) ** 2).mean()
            loss    = ce_loss + att_reg

        meter.update(loss.item(), images.size(0))

    return meter.avg


#  Main

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train V2] Using device: {device}")
    if device.type == "cuda":
        print(f"[Train V2] GPU: {torch.cuda.get_device_name(0)}")

    # Weights & Biases
    if config.USE_WANDB:
        import wandb
        wandb.init(project=config.WANDB_PROJECT_V2)

    # Vocabulary 
    if not config.VOCAB_PATH.exists():
        print("[Train V2] Vocabulary not found. Building now …")
        from vocabulary import Vocabulary as Vocab
        vocab = Vocab(threshold=config.VOCAB_THRESHOLD).build_from_coco(config.TRAIN_ANN)
        vocab.save(config.VOCAB_PATH)
    else:
        vocab = Vocabulary.load(config.VOCAB_PATH)

    # Datasets & Loaders
    if config.SMOKE_TEST:
        print("[Train V2] SMOKE TEST MODE — using val2017 for both train and val.")
        train_dataset = CocoDataset(
            img_dir  = config.VAL_IMG_DIR,
            ann_file = config.VAL_ANN,
            vocab    = vocab,
            split    = "train",
            subset   = 0.5,
        )
        val_dataset = CocoDataset(
            img_dir  = config.VAL_IMG_DIR,
            ann_file = config.VAL_ANN,
            vocab    = vocab,
            split    = "val",
            subset   = 0.5,
        )
    else:
        TOTAL_TRAIN_IMAGES = 118_287
        approx_n = int(TOTAL_TRAIN_IMAGES * config.SUBSET)
        print(f"[Train V2] Using {approx_n:,} training images "
              f"({config.SUBSET * 100:.1f}% of full COCO train).")
        train_dataset = CocoDataset(
            img_dir  = config.TRAIN_IMG_DIR,
            ann_file = config.TRAIN_ANN,
            vocab    = vocab,
            split    = "train",
            subset   = config.SUBSET,
        )
        val_dataset = CocoDataset(
            img_dir  = config.VAL_IMG_DIR,
            ann_file = config.VAL_ANN,
            vocab    = vocab,
            split    = "val",
            subset   = 1.0,
        )

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True,
        collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
        collate_fn=collate_fn,
    )

    # Models
    encoder = AttentionEncoder(
        encoder_dim = config.V2_ENCODER_DIM,
        fine_tune   = config.FINE_TUNE_CNN,
    ).to(device)

    decoder = AttentionDecoder(
        vocab_size    = len(vocab),
        encoder_dim   = config.V2_ENCODER_DIM,
        embed_dim     = config.V2_EMBED_DIM,
        hidden_dim    = config.V2_HIDDEN_DIM,
        attention_dim = config.V2_ATTENTION_DIM,
        dropout       = config.V2_DROPOUT,
    ).to(device)

    # Optimiser
    trainable_enc = encoder.get_trainable_params()
    optimiser = torch.optim.Adam([
        {"params": decoder.parameters(),  "lr": config.V2_LEARNING_RATE},
        {"params": trainable_enc,          "lr": config.V2_CNN_LR},
    ], weight_decay=config.WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=config.V2_NUM_EPOCHS, eta_min=1e-6
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scaler    = GradScaler('cuda' if device.type == 'cuda' else 'cpu')

    # Resume
    start_epoch = 1
    best_bleu4  = 0.0
    history     = {"train_loss": [], "val_loss": [], "bleu4": []}

    if config.V2_RESUME_CHECKPOINT and Path(config.V2_RESUME_CHECKPOINT).exists():
        state       = load_checkpoint(config.V2_RESUME_CHECKPOINT, device)
        encoder.load_state_dict(state["encoder"])
        decoder.load_state_dict(state["decoder"])
        optimiser.load_state_dict(state["optimiser"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        best_bleu4  = state.get("bleu4", 0.0)
        history     = state.get("history", history)
        print(f"[Train V2] Resuming from epoch {start_epoch}  (best BLEU-4: {best_bleu4:.2f})")
    elif config.V2_RESUME_CHECKPOINT:
        print(f"[Train V2] No checkpoint found at '{config.V2_RESUME_CHECKPOINT}' — starting fresh.")

    # Training loop
    print(f"\n[Train V2] Starting training for {config.V2_NUM_EPOCHS} epochs …\n")
    total_start = time.time()

    for epoch in range(start_epoch, config.V2_NUM_EPOCHS + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(encoder, decoder, train_loader, criterion,
                                     optimiser, scaler, device, epoch)
        val_loss   = val_loss_epoch(encoder, decoder, val_loader, criterion, device, epoch)

        scores = evaluate(encoder, decoder, val_loader, vocab, val_dataset,
                          device, beam_width=3)
        bleu4  = scores["bleu4"]

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["bleu4"].append(bleu4)

        elapsed = format_time(time.time() - epoch_start)
        print(
            f"Epoch {epoch:02d}/{config.V2_NUM_EPOCHS} | "
            f"Train Loss {train_loss:.4f} | Val Loss {val_loss:.4f} | "
            f"BLEU-4 {bleu4:.2f} | {elapsed}"
        )

        if config.USE_WANDB:
            import wandb
            wandb.log({"epoch": epoch, "train_loss": train_loss,
                       "val_loss": val_loss, "bleu4": bleu4,
                       "lr": get_lr(optimiser)})

        # Save checkpoint
        is_best = bleu4 > best_bleu4
        if is_best:
            best_bleu4 = bleu4

        if epoch % config.SAVE_INTERVAL == 0 or is_best:
            save_checkpoint(
                state   = {
                    "epoch"    : epoch,
                    "encoder"  : encoder.state_dict(),
                    "decoder"  : decoder.state_dict(),
                    "optimiser": optimiser.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "bleu4"    : bleu4,
                    "history"  : history,
                },
                save_dir = config.V2_CHECKPOINT_DIR,
                filename = f"ckpt_epoch_{epoch:02d}.pth",
                is_best  = is_best,
            )

    # Wrap up
    total_time = format_time(time.time() - total_start)
    print(f"\n[Train V2] Done! Total time: {total_time}  |  Best BLEU-4: {best_bleu4:.2f}")

    save_history(history, config.RESULTS_DIR / "v2_history.json")
    plot_training_curves(history, config.RESULTS_DIR / "v2_curves.png")

    if config.USE_WANDB:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
