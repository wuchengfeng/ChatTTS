from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torchaudio
import torch

# Load the pre-trained Wav2Vec2.0 processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

# Load and process the audio file
wav_path = "sample-answer-Eileen.wav"
waveform, sample_rate = torchaudio.load(wav_path)

# Ensure the waveform is single channel
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0, keepdim=True)

# Resample the audio if it's not in 16kHz
if sample_rate != 16000:
    waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)

# Process the audio (convert to the format needed by the model)
inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")

# Get the embeddings from the model (these are contextual embeddings)
with torch.no_grad():
    outputs = model(**inputs)

# The embeddings will be in outputs.last_hidden_state
embeddings = outputs.last_hidden_state

# To get a 768-dimensional embedding, you can take the mean across the time dimension
mean_embedding = torch.mean(embeddings, dim=1)

print(f"Extracted embedding shape: {mean_embedding.shape}")  # Should be [batch_size, 768]

# Save the embedding
torch.save(mean_embedding, "audio_embedding_768.pt")
print("Embedding saved to audio_embedding_768.pt")
