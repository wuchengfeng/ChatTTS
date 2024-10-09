from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np

# Load and preprocess the audio file
fpath = Path("活动-Sample answer copy.m4a")
wav = preprocess_wav(fpath)

# Create an encoder and generate the embedding
encoder = VoiceEncoder()
embed = encoder.embed_utterance(wav)
print(embed)
# 将嵌入向量保存到文件中
with open('saved_speaker_embedding_get.txt', 'w') as f:
    f.write(','.join(map(str, embed.tolist())))
print(f"Speaker embedding saved as saved_speaker_embedding.txt")

