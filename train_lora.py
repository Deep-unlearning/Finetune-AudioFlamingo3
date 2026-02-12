"""
Fine-tune Audio Flamingo 3 for Audio QA with LoRA (Low-Rank Adaptation) for parameter-efficient training.
"""

import torch
from transformers import (
    AudioFlamingo3ForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Audio


class AudioFlamingo3DataCollator:
    def __init__(self, processor, model_checkpoint):
        self.processor = processor
        self.model_checkpoint = model_checkpoint

    def __call__(self, features):

        conversation = []
        for feature in features:
            sample = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the audio."},
                    {"type": "audio", "audio": feature["audio"]["array"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": feature["caption"]}],
            }]
            conversation.append(sample)

        return self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            output_labels=True,
        )


def load_and_prepare_dataset():
    """Load and prepare streaming dataset for training."""
    dataset_name = "OpenSound/AudioCaps"
    
    print(f"Loading dataset: {dataset_name} (streaming)")
    dataset = load_dataset(dataset_name, split="train", streaming=True)
    
    # Cast audio to 16kHz (required by WhisperFeatureExtractor)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Streaming datasets use .take() and .skip() instead of .select()
    train_dataset = dataset.take(1000)
    eval_dataset = dataset.skip(1000).take(100)
    
    return train_dataset, eval_dataset


def main():
    # Configuration
    model_checkpoint = "nvidia/audio-flamingo-3-hf"
    output_dir = "./audio-flamingo-3-hf-lora-finetuned"
    
    # Set device
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {torch_device}")
    
    # Load processor and model
    print("Loading processor and model...")
    processor = AutoProcessor.from_pretrained(model_checkpoint)

    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        model_checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # LoRA configuration - target the LLM attention layers
    lora_config = LoraConfig(
        r=16,  # LoRA rank
        lora_alpha=32,  # LoRA scaling factor
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load and prepare dataset
    train_dataset, eval_dataset = load_and_prepare_dataset()
    
    # Setup data collator
    data_collator = AudioFlamingo3DataCollator(processor, model_checkpoint)
    
    # Training arguments (use max_steps for streaming datasets)
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        learning_rate=1e-4,
        max_steps=500,  # Must use max_steps with streaming datasets
        bf16=True,
        logging_steps=10,
        eval_steps=100,
        save_steps=250,  # Save less often to avoid filling disk
        save_total_limit=2,  # Keep only the latest checkpoint
        save_only_model=True,  # Skip saving optimizer state
        eval_strategy="steps",
        save_strategy="steps",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,  # Must be 0 for streaming datasets
    )
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    print("Starting LoRA training...")
    trainer.train()
    
    # Save LoRA adapters and processor
    print(f"Saving LoRA adapters to {output_dir}")
    trainer.save_model()
    processor.save_pretrained(output_dir)
    
    # Final evaluation
    if eval_dataset:
        results = trainer.evaluate()
        print(f"Final evaluation results: {results}")
    
    print("LoRA training complete!")
    print(f"To use the model, load the base model and apply these LoRA adapters.")


if __name__ == "__main__":
    main()
