from pyannote.audio import Inference
import torch
import torchaudio

# 初始化预训练的说话者嵌入模型
model = Inference("pyannote/embedding", device=torch.device('cpu'), use_auth_token="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")

# 加载并预处理音频
wav_path = "sample-answer-Eileen.wav"
waveform, sample_rate = torchaudio.load(wav_path)

# 确保 waveform 是二维张量，形状为 (channel, time)
if len(waveform.shape) == 1:
    waveform = waveform.unsqueeze(0)  # 如果是一维的，增加一个维度使其变成 (1, time)

# 创建包含 waveform 和 sample_rate 的字典
audio_data = {"waveform": waveform, "sample_rate": sample_rate}

# 提取说话者嵌入向量
embedding = model(audio_data)

# 提取嵌入向量的实际数据
embedding_tensor = embedding.data

# 打印嵌入向量的形状
print("Speaker embedding shape:", embedding_tensor.shape)

# 保存嵌入向量到文件
torch.save(embedding_tensor, 'speaker_embedding.pt')
