import torch
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')

import ChatTTS
from IPython.display import Audio

chat = ChatTTS.Chat()
chat.load_models()

import torch

params_refine_text = {'prompt':'[break_1][oral_2]'}
# 加载保存的嵌入向量
embedding = torch.load('seed_1518_restored_emb.pt', map_location=torch.device('cpu'))
# embedding = embedding.squeeze()
#print("Loaded embedding shape:", embedding)

# 确认嵌入向量的维度是否正确
assert embedding.shape[0] == 768, "Embedding dimension mismatch!"  # 假设你的目标维度是768
#params_infer_code={'spk_emb':embedding}
params_infer_code = {'spk_emb': embedding, 'temperature':.3, 'top_P':0.7, 'top_K':20}
texts = """I prefer cycling to travelling by bus . Actually, it is often faster to go by bike because you don't get stuck in traffic congestion!"""
wav = chat.infer(texts, \
    params_refine_text=params_refine_text, params_infer_code=params_infer_code)
Audio(wav[0], rate=24_000, autoplay=True)


# 保存音频文件
import numpy as np
from scipy.io.wavfile import write

# 假设wavs[0]是包含音频数据的二维数组
audio_data = np.array(wav[0])

# 确保音频数据是一维数组
audio_data = audio_data.flatten()

# 确保音频数据在-1.0到1.0之间
audio_data = np.clip(audio_data, -1.0, 1.0)

# 将音频数据转换为int16类型
audio_data = (audio_data * 32767).astype(np.int16)

# 设置采样率为24,000 Hz
rate = 24000

# 将音频数据保存为WAV文件
output_file = "output_audio.wav"
write(output_file, rate, audio_data)
print(f"Audio file saved as {output_file}")