import os
import pandas as pd
import numpy as np
from scipy.io.wavfile import write
from tqdm import tqdm  # 用于显示进度条
import torch
from IPython.display import Audio
import ChatTTS
import torch._dynamo
torch._dynamo.config.suppress_errors = True
# 初始化 ChatTTS 模型
chat = ChatTTS.Chat()
chat.load_models()

# 加载保存的嵌入向量
embedding = torch.load('seed_1518_restored_emb.pt')

# 确认嵌入向量的维度是否正确
assert embedding.shape[0] == 768, "Embedding dimension mismatch!"  # 假设你的目标维度是768

# 参数设置
params_refine_text = {'prompt': '[break_2][oral_2][speed_1]'}
params_infer_code = {'spk_emb': embedding, 'temperature': .3, 'top_P': 0.7, 'top_K': 20}

# CSV 文件路径
csv_file = 'part111.csv'  # 请修改为你的 CSV 文件路径

# 读取 CSV 文件
df = pd.read_csv(csv_file, delimiter=',')

# 处理每一行并生成音频
for index, row in tqdm(df.iterrows(), total=len(df)):
    part = row['part']
    number = row['number']
    text = row['text']
    
    # 生成音频
    try:
        wav = chat.infer(text, params_refine_text=params_refine_text, params_infer_code=params_infer_code)

        # 将音频数据转换为 numpy 数组
        audio_data = np.array(wav[0])

        # 确保音频数据是一维数组
        audio_data = audio_data.flatten()

        # 确保音频数据在-1.0到1.0之间
        audio_data = np.clip(audio_data, -1.0, 1.0)

        # 将音频数据转换为 int16 类型
        audio_data = (audio_data * 32767).astype(np.int16)

        # 设置采样率为24,000 Hz
        rate = 24000

        # 保存文件路径
        output_dir = os.path.join('output_audio', part)
        os.makedirs(output_dir, exist_ok=True)  # 创建 part 对应的目录
        
        output_file = os.path.join(output_dir, f"{number}.wav")

        # 将音频数据保存为 WAV 文件
        write(output_file, rate, audio_data)
        print(f"Audio file saved as {output_file}")
        
    except Exception as e:
        print(f"Error processing row {index}: {e}")
