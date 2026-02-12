# Fine-tune Audio Flamingo 3 for Audio Captioning

Fine-tune [NVIDIA's Audio Flamingo 3](https://huggingface.co/nvidia/audio-flamingo-3-hf) (8B) on audio captioning tasks using Hugging Face Transformers. Supports both full fine-tuning and LoRA (parameter-efficient) training.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Deep-unlearning/Finetune-AudioFlamingo3.git
cd Finetune-AudioFlamingo3
```

### 2. Set up the environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Or with [uv](https://github.com/astral-sh/uv):

```bash
uv venv .venv --python 3.10 && source .venv/bin/activate
uv pip install -r requirements.txt
```

## Dataset

Both training scripts use [OpenSound/AudioCaps](https://huggingface.co/datasets/OpenSound/AudioCaps) in **streaming mode** -- no need to download the full 43.9 GB dataset upfront.

- Audio is automatically resampled to 16 kHz (required by the Whisper feature extractor)
- Default: 5,000 training samples + 1,000 eval samples (full fine-tune), 1,000 + 100 (LoRA)
- Captions are formatted as conversations using the prompt `"Describe the audio."`

You can adjust sample counts in the `load_and_prepare_dataset()` function:

```python
train_dataset = dataset.take(5000)       # number of training samples
eval_dataset = dataset.skip(5000).take(1000)  # number of eval samples
```

## Training

### Full fine-tuning

```bash
python train.py
```


### LoRA fine-tuning (parameter-efficient)

```bash
python train_lora.py
```

Both scripts use streaming datasets with `max_steps` instead of epochs, save only model weights (not optimizer state) to conserve disk, and keep at most 2 checkpoints.

## Inference

### With a full fine-tuned model

```bash
python inference.py \
  --model_path ./audio-flamingo-3-hf-finetuned \
  --audio_path /path/to/audio.wav \
  --question "What sounds can you hear in this audio?"
```

### With LoRA adapters

```bash
python inference.py \
  --model_path ./audio-flamingo-3-hf-lora-finetuned \
  --audio_path /path/to/audio.wav \
  --question "Describe the sounds in this audio clip" \
  --use_lora
```

