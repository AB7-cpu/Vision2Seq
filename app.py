"""
Gradio web demo for the Image Captioning project.
"""

import argparse
import random
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from torchvision import transforms

import config
from vocabulary import Vocabulary
from models.v1  import CNNEncoder        as V1Encoder
from models.v1  import LSTMDecoder       as V1Decoder
from models.v2  import AttentionEncoder  as V2Encoder
from models.v2  import AttentionDecoder  as V2Decoder
from models.v3  import TransformerEncoder as V3Encoder
from models.v3  import TransformerDecoder as V3Decoder
from utils      import load_checkpoint

# Version registry

VERSIONS: dict[str, dict] = {
    "V1 — CNN + LSTM  (Show and Tell)": {
        "key"         : "v1",
        "checkpoint"  : config.CHECKPOINT_DIR / "v1" / "best_model.pth",
        "description" : (
            "Classic encoder-decoder architecture. "
            "ResNet-50 encodes the image into a 256-dim vector that seeds the "
            "LSTM hidden state. The LSTM then generates one word per step using "
            "teacher forcing during training and beam search at inference."
        ),
        "arch_table"  : f"""
| Component | Details |
|---|---|
| **Encoder** | ResNet-50 (pretrained ImageNet-1K V2) |
| **Projection** | Linear(2048 → {config.EMBED_DIM}) + ReLU + Dropout |
| **LSTM init** | h₀ = tanh(W·feat),  c₀ = tanh(W·feat) |
| **LSTM** | {config.NUM_LAYERS} layer · hidden = {config.HIDDEN_DIM} |
| **Training** | Teacher forcing · CrossEntropy · Cosine LR |
| **Inference** | Greedy (beam=1) or Beam Search |
""",
    },
    "V2 — CNN + Attention + LSTM  (Show, Attend and Tell)": {
        "key"         : "v2",
        "checkpoint"  : config.CHECKPOINT_DIR / "v2" / "best_model.pth",
        "description" : (
            "Adds soft spatial attention (Xu et al. 2015). The encoder keeps the "
            "full 7×7 feature grid (49 locations). At each decoding step an additive "
            "attention module learns *where to look*, computing a context vector as a "
            "weighted sum over spatial features. Training uses doubly-stochastic "
            "regularisation to encourage each location to be attended equally often."
        ),
        "arch_table"  : f"""
| Component | Details |
|---|---|
| **Encoder** | ResNet-50 — spatial grid (7 × 7 × 2048 = 49 locations) |
| **Attention** | Additive soft attention · dim = {config.V2_ATTENTION_DIM} |
| **Context** | Weighted sum of spatial features per timestep |
| **LSTM** | LSTMCell · hidden = {config.V2_HIDDEN_DIM} · input = embed + context |
| **Regularisation** | Doubly-stochastic α penalty (λ = {config.V2_ALPHA_C}) |
| **Inference** | Greedy or Beam Search |
""",
    },
    "V3 — CNN + Transformer Decoder": {
        "key"         : "v3",
        "checkpoint"  : config.CHECKPOINT_DIR / "v3" / "best_model.pth",
        "description" : (
            "Fully attention-based architecture. ResNet-50 outputs a 7×7 spatial "
            "grid projected to d_model=512. A Transformer decoder (Pre-LN, 3 layers, "
            "8 heads) generates captions with multi-head self-attention over previous "
            "tokens and cross-attention over all 49 image locations simultaneously — "
            "no recurrence, parallel training."
        ),
        "arch_table"  : f"""
| Component | Details |
|---|---|
| **Encoder** | ResNet-50 → Linear(2048 → {config.V3_D_MODEL}) + LayerNorm |
| **Spatial pos** | Learned embedding · 49 locations |
| **Decoder** | Transformer · {config.V3_N_LAYERS} layers · {config.V3_N_HEADS} heads · d_model = {config.V3_D_MODEL} |
| **FFN** | dim = {config.V3_DIM_FEEDFORWARD} · Pre-LN (norm_first=True) |
| **Token pos** | Learned sequential position encoding |
| **Inference** | Greedy or Beam Search (autoregressive) |
""",
    },
}

# Global model state
_state: dict = {
    "encoder"   : None,
    "decoder"   : None,
    "vocab"     : None,
    "device"    : None,
    "loaded_ver": None,
}

_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(config.IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
])


# Model loading

def load_model(version_label: str) -> str:
    global _state

    ver = VERSIONS[version_label]
    key = ver["key"]
    ckpt_path = ver["checkpoint"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_info = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"

    # Vocab
    if not config.VOCAB_PATH.exists():
        return "❌  Vocabulary not found. Run `python train.py` first to build it."

    # Checkpoint
    if not ckpt_path.exists():
        return (
            f"⚠️  {version_label.split('—')[0].strip()} checkpoint not found at:\n"
            f"    {ckpt_path}\n"
            f"Train this version first, then reload."
        )

    vocab = Vocabulary.load(config.VOCAB_PATH)

    # Build model for the selected version
    if key == "v1":
        encoder = V1Encoder(embed_dim=config.EMBED_DIM, fine_tune=False).to(device)
        decoder = V1Decoder(
            vocab_size = len(vocab),
            embed_dim  = config.EMBED_DIM,
            hidden_dim = config.HIDDEN_DIM,
            num_layers = config.NUM_LAYERS,
            dropout    = 0.0,
        ).to(device)

    elif key == "v2":
        encoder = V2Encoder(
            encoder_dim = config.V2_ENCODER_DIM,
            fine_tune   = False,
        ).to(device)
        decoder = V2Decoder(
            vocab_size    = len(vocab),
            encoder_dim   = config.V2_ENCODER_DIM,
            embed_dim     = config.V2_EMBED_DIM,
            hidden_dim    = config.V2_HIDDEN_DIM,
            attention_dim = config.V2_ATTENTION_DIM,
            dropout       = 0.0,
        ).to(device)

    elif key == "v3":
        encoder = V3Encoder(
            d_model   = config.V3_D_MODEL,
            dropout   = 0.0,
            fine_tune = False,
        ).to(device)
        decoder = V3Decoder(
            vocab_size      = len(vocab),
            d_model         = config.V3_D_MODEL,
            n_heads         = config.V3_N_HEADS,
            n_layers        = config.V3_N_LAYERS,
            dim_feedforward = config.V3_DIM_FEEDFORWARD,
            dropout         = 0.0,
            max_seq_len     = config.MAX_SEQ_LEN,
        ).to(device)

    else:
        return f"⚠️  Unknown version key: {key}"

    state = load_checkpoint(str(ckpt_path), device)
    encoder.load_state_dict(state["encoder"])
    decoder.load_state_dict(state["decoder"])
    encoder.eval()
    decoder.eval()

    _state.update({
        "encoder"   : encoder,
        "decoder"   : decoder,
        "vocab"     : vocab,
        "device"    : device,
        "loaded_ver": key,
    })

    return (
        f"✅  {version_label.split('—')[0].strip()} loaded on {gpu_info}  "
        f"|  Vocab: {len(vocab):,} words"
    )


# Inference

@torch.no_grad()
def generate_caption(image: Image.Image, beam_width: int) -> tuple[str, str]:
    if _state["encoder"] is None:
        return "⚠️  Load a model first using the panel above.", ""
    if image is None:
        return "⚠️  Please upload an image.", ""

    tensor   = _transform(image.convert("RGB")).unsqueeze(0).to(_state["device"])
    features = _state["encoder"](tensor)

    if beam_width == 1:
        tokens = _state["decoder"].generate_greedy(
            features, _state["vocab"], max_length=config.MAX_SEQ_LEN
        )
        method = "Greedy decoding"
    else:
        tokens = _state["decoder"].generate_beam(
            features, _state["vocab"],
            beam_width=beam_width, max_length=config.MAX_SEQ_LEN
        )
        method = f"Beam search  (width = {beam_width})"

    caption = _state["vocab"].decode(tokens)
    caption = caption[0].upper() + caption[1:] if caption else caption

    info = f"🔤  Tokens: {len(tokens)}   |   📐  {method}"
    return caption, info


# Helpers

def version_is_available(label: str) -> bool:
    return VERSIONS[label]["checkpoint"].exists()

def get_arch_table(label: str) -> str:
    return VERSIONS[label]["arch_table"]

def get_description(label: str) -> str:
    return VERSIONS[label]["description"]

def get_status_badge(label: str) -> str:
    if version_is_available(label):
        return "🟢 Checkpoint found"
    key = VERSIONS[label]["key"]
    if key == "v1":
        return "🔴 Not trained yet"
    return "🔵 Coming soon"

def get_example_images(n: int = 6) -> list[list]:
    val_dir = config.VAL_IMG_DIR
    if not val_dir.exists():
        return []
    all_imgs = list(val_dir.glob("*.jpg"))
    sample   = random.sample(all_imgs, min(n, len(all_imgs)))
    return [[str(p), 5] for p in sample]

def on_version_change(label: str):
    return get_arch_table(label), get_description(label), get_status_badge(label)


# Gradio UI

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

body, .gradio-container { font-family: 'Inter', sans-serif !important; }

/* ── Header ── */
#header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    border-radius: 16px;
    padding: 32px 36px;
    margin-bottom: 4px;
    border: 1px solid rgba(255,255,255,0.07);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
#header h1 {
    color: #f1f5f9;
    font-size: 2rem;
    font-weight: 700;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
#header p { color: #94a3b8; font-size: 0.92rem; margin: 0; }

/* ── Version panel ── */
#version_panel {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 20px 24px;
}

/* ── Status ── */
#status_box textarea {
    background: #0d1117 !important;
    color: #58a6ff !important;
    font-family: monospace !important;
    font-size: 0.84rem !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
}

/* ── Caption output ── */
#caption_out textarea {
    font-size: 1.3rem !important;
    font-weight: 600 !important;
    color: #e2e8f0 !important;
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 10px !important;
    line-height: 1.7 !important;
}
#info_out textarea {
    font-size: 0.84rem !important;
    color: #8b949e !important;
    background: #0d1117 !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
}

/* ── Generate button ── */
#gen_btn {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    border: none !important;
    color: white !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    padding: 14px !important;
    box-shadow: 0 4px 20px rgba(102,126,234,0.4) !important;
    transition: all 0.2s ease !important;
}
#gen_btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(102,126,234,0.6) !important;
}

/* ── Load button ── */
#load_btn {
    background: linear-gradient(135deg, #11998e, #38ef7d) !important;
    border: none !important;
    color: #0d1117 !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}

/* ── Badge ── */
#badge_box textarea {
    background: transparent !important;
    border: none !important;
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
    padding: 0 !important;
}

/* ── Arch card ── */
#arch_card {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px 20px;
    color: #94a3b8;
    font-size: 0.88rem;
    line-height: 1.7;
}

input[type='range'] { accent-color: #667eea; }
"""


def build_ui() -> gr.Blocks:
    version_labels = list(VERSIONS.keys())
    default_ver    = version_labels[0]

    with gr.Blocks(
        css   = CSS,
        title = "Image Captioning Demo",
        theme = gr.themes.Base(
            primary_hue  = "indigo",
            neutral_hue  = "slate",
            font         = gr.themes.GoogleFont("Inter"),
        ),
    ) as demo:

        # Header
        gr.HTML("""
        <div id="header">
          <h1>🖼️ Image Captioning &nbsp;·&nbsp;
            <span style="background:linear-gradient(90deg,#667eea,#a78bfa);
                         -webkit-background-clip:text;-webkit-text-fill-color:transparent">
              Deep Learning Portfolio
            </span>
          </h1>
          <p>
            Three progressively complex architectures trained on MS-COCO 2017 &nbsp;|&nbsp;
            Select a version below, load it, then upload any image.
          </p>
        </div>
        """)

        # Version selector panel
        with gr.Group(elem_id="version_panel"):
            gr.Markdown("### ⚙️ Model Version")
            with gr.Row(equal_height=True):
                version_dd = gr.Dropdown(
                    choices  = version_labels,
                    value    = default_ver,
                    label    = "Select Architecture",
                    scale    = 3,
                    interactive = True,
                )
                with gr.Column(scale=1):
                    badge_box = gr.Textbox(
                        value       = get_status_badge(default_ver),
                        label       = "Status",
                        interactive = False,
                        elem_id     = "badge_box",
                    )
                    load_btn = gr.Button("⚡ Load Model", elem_id="load_btn")

            # Description — updates on version change
            desc_md = gr.Markdown(f"*{get_description(default_ver)}*")

            # Status bar
            status_box = gr.Textbox(
                label       = "Load Status",
                value       = "⏳  Select a version and click Load Model.",
                interactive = False,
                elem_id     = "status_box",
            )

        # Inference panel
        gr.Markdown("---")
        with gr.Row(equal_height=False):

            # Left — inputs
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type   = "pil",
                    label  = "📤 Upload Image",
                    height = 340,
                )
                beam_slider = gr.Slider(
                    minimum = 1, maximum = 10, value = 5, step = 1,
                    label   = "Beam Width  (1 = greedy, 2–10 = beam search)",
                )
                gen_btn = gr.Button("✨ Generate Caption", variant="primary",
                                    elem_id="gen_btn")

            # Right — outputs + arch
            with gr.Column(scale=1):
                caption_out = gr.Textbox(
                    label       = "Generated Caption",
                    placeholder = "Upload an image and click Generate…",
                    lines       = 3,
                    elem_id     = "caption_out",
                )
                info_out = gr.Textbox(
                    label       = "Decoding Info",
                    interactive = False,
                    lines       = 1,
                    elem_id     = "info_out",
                )

                gr.Markdown("### 📐 Architecture")
                arch_md = gr.Markdown(get_arch_table(default_ver))

        # Example images
        examples = get_example_images(6)
        if examples:
            gr.Markdown("---\n### 🗂️ Example Images  *(randomly sampled from COCO val2017)*")
            gr.Examples(
                examples       = examples,
                inputs         = [image_input, beam_slider],
                outputs        = [caption_out, info_out],
                fn             = generate_caption,
                cache_examples = False,
                label          = "",
            )

        # Wire events

        # Version dropdown → update badge, description, arch table
        version_dd.change(
            fn      = on_version_change,
            inputs  = [version_dd],
            outputs = [arch_md, desc_md, badge_box],
        )

        # Load button
        load_btn.click(
            fn      = load_model,
            inputs  = [version_dd],
            outputs = [status_box],
        )

        # Generate caption
        gen_btn.click(
            fn      = generate_caption,
            inputs  = [image_input, beam_slider],
            outputs = [caption_out, info_out],
        )

        # Also generate on image upload
        image_input.upload(
            fn      = generate_caption,
            inputs  = [image_input, beam_slider],
            outputs = [caption_out, info_out],
        )

    return demo


# Entry point

def parse_args():
    p = argparse.ArgumentParser(description="Launch the Image Captioning Gradio demo.")
    p.add_argument("--share", action="store_true",
                   help="Create a public Gradio share link.")
    p.add_argument("--port", type=int, default=7860)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    demo = build_ui()
    demo.launch(
        server_port = args.port,
        share       = args.share,
        show_error  = True,
        inbrowser   = True,
    )
