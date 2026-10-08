import torch
import whisper

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

model = whisper.load_model("turbo", device=device)
result = model.transcribe("Recording.m4a")
print(result["text"])