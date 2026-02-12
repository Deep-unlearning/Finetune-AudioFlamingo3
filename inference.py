"""
Run inference with a fine-tuned Audio Flamingo model for Audio QA.
"""

import torch
from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
from peft import PeftModel
import librosa
import argparse


def load_model(model_path, use_lora=False, base_model_id="nvidia/audio-flamingo-3-hf"):
    """Load the model and processor."""
    processor = AutoProcessor.from_pretrained(model_path)

    if use_lora:
        # Load base model and apply LoRA adapters
        print(f"Loading base model: {base_model_id}")
        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print(f"Loading LoRA adapters from: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
    else:
        # Load full fine-tuned model
        print(f"Loading model from: {model_path}")
        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    model.eval()
    return model, processor


def answer_question(model, processor, audio_path, question):
    """Answer a question about an audio file."""
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)

    # Create QA conversation
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "audio"}
            ],
        }
    ]

    # Process inputs
    inputs = processor.apply_chat_template(
        [conversation],
        audio=[audio],
        sampling_rate=sr,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    ).to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=500)

    # Decode
    answer = processor.batch_decode(
        outputs[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )[0]

    return answer


def multi_turn_qa(model, processor, audio_path, questions):
    """Ask multiple questions about an audio file in sequence."""
    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)

    conversation = []
    responses = []

    for question in questions:
        # Add user question
        conversation.append({
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "audio"} if len(conversation) == 0 else {"type": "text", "text": ""}
            ],
        })

        # Process inputs
        inputs = processor.apply_chat_template(
            [conversation],
            audio=[audio] if len(conversation) == 1 else None,
            sampling_rate=sr,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        # Decode
        response = processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]

        # Add assistant response to conversation
        conversation.append({
            "role": "assistant",
            "content": [{"type": "text", "text": response}]
        })

        responses.append(response)

    return responses


def main():
    parser = argparse.ArgumentParser(description="Run Audio QA inference with Audio Flamingo")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model")
    parser.add_argument("--audio_path", type=str, required=True, help="Path to audio file")
    parser.add_argument("--question", type=str, required=True, help="Question to ask about the audio")
    parser.add_argument("--use_lora", action="store_true", help="Load model as LoRA adapters")
    parser.add_argument("--base_model_id", type=str, default="nvidia/audio-flamingo-3-hf",
                        help="Base model ID (for LoRA)")

    args = parser.parse_args()

    # Load model
    model, processor = load_model(args.model_path, args.use_lora, args.base_model_id)

    # Run inference
    print(f"\nQuestion: {args.question}")
    result = answer_question(model, processor, args.audio_path, args.question)
    print(f"Answer: {result}")


if __name__ == "__main__":
    main()
