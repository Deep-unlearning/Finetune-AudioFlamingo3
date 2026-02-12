# Finetune Audio Flamingo for Audio QA with Transformers 🤗

This project fine-tunes NVIDIA's Audio Flamingo 3 model for Audio Question Answering (AQA) tasks using Hugging Face's transformers and datasets libraries.

## Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/Deep-unlearning/Finetune-Audio-Flamingo.git
cd Finetune-Audio-Flamingo
```

### Step 2: Set up the environment

**Option 1: Using UV (recommended)**

Install [uv](https://github.com/astral-sh/uv) and run:

```bash
uv venv .venv --python 3.10 && source .venv/bin/activate
uv pip install -r requirements.txt
```

**Option 2: Using pip**

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset

This repository uses **AudioCaps** by default - a dataset with 51,812 audio clips and captions that's perfect for training audio understanding models.

### AudioCaps Dataset (Default)
- Dataset: `OpenSound/AudioCaps`
- 51,812 audio clips with text descriptions
- ~45k training samples, 2.2k validation, 4.4k test
- 1-10 second audio clips
- **✅ Efficient sampling** - uses slice notation to download only what you need
- **✅ Subsampled by default** - uses 1,000 train + 100 eval samples
- [Dataset Page](https://huggingface.co/datasets/OpenSound/AudioCaps)

### Adjusting Sample Size

The training scripts use streaming to avoid downloading the entire dataset. You can adjust the number of samples by editing the configuration in `train.py` or `train_lora.py`:

```python
NUM_TRAIN_SAMPLES = 1000  # Increase for more training data
NUM_EVAL_SAMPLES = 100    # Increase for better evaluation
```

To use the full dataset, change to non-streaming mode:
```python
dataset = load_dataset(DATASET_NAME, split="train", streaming=False)
dataset = dataset.train_test_split(test_size=0.1, seed=42)
```

The training scripts automatically convert audio captions to QA format using questions like:
- "What sounds can you hear in this audio?"
- "Describe the audio."
- "What is happening in this audio clip?"

### Alternative Datasets

You can easily switch to other audio QA datasets by changing the `DATASET_NAME` variable in `train.py` or `train_lora.py`:

**Option 1: MMSU** (Spoken Language Understanding)
- Dataset: `ddwang2000/MMSU`
- 5,000 multiple-choice QA pairs across 47 tasks
- Covers phonetics, prosody, semantics, and more

**Option 2: Other audio captioning/QA datasets**
- The training scripts support multiple formats
- See "Dataset Format" section below for compatibility

### Dataset Format

The training scripts support multiple dataset formats:

**Format 1: Audio with caption (AudioCaps format - default)**
```python
{
    "audio": Audio(sampling_rate=16000),  # Audio object
    "caption": "A dog barking loudly"  # Automatically converted to QA
}
```

**Format 2: Audio with QA**
```python
{
    "audio": Audio(sampling_rate=16000),  # Audio object
    "question": "What animal sound can you hear?",
    "answer": "A dog barking"
}
```

**Format 3: Audio path (requires local files)**
```python
{
    "audio_url": "path/to/audio.wav",  # or "audio_path"
    "question": "How many different sounds are present?",
    "answer": "C. 2",
    "choice": ["A. 3", "B. 1", "C. 2", "D. 4"]  # Optional
}
```

**Format 4: Conversations (multi-turn)**
```python
{
    "audio": Audio(sampling_rate=16000),
    "conversations": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you hear?"},
                {"type": "audio"}
            ]
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "Birds chirping"}]
        }
    ]
}
```

The training scripts will automatically detect and handle the appropriate format.

## Training

### Standard fine-tuning

```bash
python train.py
```

**Note**: By default, the script uses 1,000 training samples and 100 evaluation samples with streaming. This allows for quick experimentation without downloading the full 43.9 GB dataset.

### LoRA fine-tuning (parameter-efficient)

```bash
python train_lora.py
```

**Recommended**: Start with LoRA fine-tuning as it's faster and requires less memory.

Outputs are saved to the `outputs/` directory by default.

## Inference

### Run Audio QA with your fine-tuned model

```bash
python inference.py \
  --model_path ./outputs/final \
  --audio_path /path/to/audio.wav \
  --question "What animal sound can you hear in this audio?"
```

### Using LoRA adapters

```bash
python inference.py \
  --model_path ./outputs_lora/final \
  --audio_path /path/to/audio.wav \
  --question "Describe the sounds in this audio clip" \
  --use_lora
```

## Model Information

This repository supports fine-tuning of:
- `nvidia/audio-flamingo-3-hf` (8B parameters)
- Specialized for audio question answering and reasoning
- Handles up to 10 minutes of audio input
- Uses AF-Whisper encoder + Qwen2.5-7B LLM backbone

## Use Cases

- Environmental sound understanding and reasoning
- Music analysis and description
- Speech content understanding
- Temporal audio event detection
- Bioacoustic recognition

## License

Please note that Audio Flamingo 3 is released under the NVIDIA OneWay Noncommercial License and is intended for non-commercial research purposes only.
