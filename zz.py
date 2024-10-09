import torch
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')

import ChatTTS
from IPython.display import Audio

chat = ChatTTS.Chat()
chat.load_models()

import torch

params_refine_text = {'prompt':'[oral_2][break_3][speed_5]'}
# 加载保存的嵌入向量
embedding = torch.load('seed_1518_restored_emb.pt', map_location=torch.device('cpu'))
# embedding = embedding.squeeze()
#print("Loaded embedding shape:", embedding)

# 确认嵌入向量的维度是否正确
assert embedding.shape[0] == 768, "Embedding dimension mismatch!"  # 假设你的目标维度是768
#params_infer_code={'spk_emb':embedding}
params_infer_code = {'spk_emb': embedding, 'temperature':.3, 'top_P':0.7, 'top_K':20}

wav = chat.infer("I prefer cycling to travelling by bus . Cycling is so much more convenient than taking the bus if you are not travelling too far . Actually, it is often faster to go by bike because I won't get stuck in traffic trafﬁc trafﬁc congestion! Cycling is also better for my health than all other means of transport because it is a form of physical exercise. ", \
    params_refine_text=params_refine_text, params_infer_code=params_infer_code)
Audio(wav[0], rate=24_000, autoplay=True)