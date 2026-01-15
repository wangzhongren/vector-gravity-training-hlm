import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import requests, os, re
from torch.cuda.amp import autocast, GradScaler
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# ==========================================
# 1. 4090 红楼梦专项配置
# ==========================================
device = torch.device("cuda")
HIDDEN_SIZE = 2048    # 中文语义复杂，增大隐藏层，4090 显存管够
NUM_LAYERS = 6
SEQ_LEN = 256         # 增加序列长度，捕捉红楼梦长句逻辑
BATCH_SIZE = 256      # 配合 1536 的维度，128 是个兼顾速度与稳定的平衡点
LR = 0.0002           # 降低学习率，防止大词库下梯度震荡

START_ALPHA = 50.0    # 初始引力（中文初期不宜过大，防止 Loss 爆炸）
END_ALPHA = 1.0
ANNEAL_STEPS = 8000   

# 自动获取红楼梦数据集
def get_hlm_data():
    path = "hlm.txt"
    # if not os.path.exists(path):
    #     print("正在从镜像源获取《红楼梦》全本...")
    #     # 使用一个干净的开源文本源
    #     url = "https://raw.githubusercontent.com/the-paper-guide/hongloumeng/master/hongloumeng.txt"
    #     r = requests.get(url)
    #     r.encoding = 'utf-8'
    #     raw_text = r.text
    #     # 清洗：去除空白、无意义符号，只保留核心文本
    #     clean_text = re.sub(r'\s+', ' ', raw_text)
    #     with open(path, "w", encoding='utf-8') as f:
    #         f.write(clean_text)
    
    with open(path, "r", encoding='utf-8') as f:
        return f.read()

text = get_hlm_data()
chars = sorted(list(set(text)))
vocab_size = len(chars)
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}
print(f"✅ 数据就绪！总字符: {len(text)}, 词库大小: {vocab_size}")

# ==========================================
# 2. 定向矢量对齐模型 (保持 VGT 核心)
# ==========================================
class VectorGravityModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(num_layers)])
        self.lns = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.out_proj = nn.Linear(hidden_size, vocab_size)

    def forward_step(self, x_emb, h_list):
        new_h_list = []
        curr = x_emb
        for i, (layer, ln) in enumerate(zip(self.layers, self.lns)):
            h_next = torch.tanh(ln(h_list[i] + layer(curr)))
            new_h_list.append(h_next)
            curr = h_next
        logits = self.out_proj(new_h_list[-1])
        return logits, new_h_list

# ==========================================
# 3. 引力引擎与训练
# ==========================================
def get_alpha(it):
    if it >= ANNEAL_STEPS: return END_ALPHA
    return START_ALPHA - (it / ANNEAL_STEPS) * (START_ALPHA - END_ALPHA)

def train_hlm_vgt():
    model = VectorGravityModel(vocab_size, HIDDEN_SIZE, NUM_LAYERS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    scaler = GradScaler()

    print(f"� 启动红楼梦 VGT 引擎 | 目标：{vocab_size} 汉字流形结晶")

    for i in range(20001): # 建议多跑一会儿，中文收敛慢
        model.train()
        current_alpha = get_alpha(i)
        optimizer.zero_grad(set_to_none=True)
        
        idx = np.random.randint(0, len(text) - SEQ_LEN - 1, BATCH_SIZE)
        batch_data = torch.tensor([[char_to_ix[c] for c in text[j : j + SEQ_LEN + 1]] for j in idx]).to(device)
        inputs, targets = batch_data[:, :-1], batch_data[:, 1:]

        h_states = [torch.zeros(BATCH_SIZE, 1, HIDDEN_SIZE, device=device) for _ in range(NUM_LAYERS)]
        all_input_embs = model.emb(inputs)
        with torch.no_grad():
            all_target_vecs = model.emb(targets)

        total_loss = 0
        with autocast():
            for t in range(SEQ_LEN):
                logits, h_states = model.forward_step(all_input_embs[:, t:t+1, :], h_states)
                
                # VGT 核心：矢量引力
                target_v = all_target_vecs[:, t:t+1, :]
                geo_loss = 0
                for layer_idx in [2, 5]: # 对中间层和顶层施压
                    geo_loss += F.mse_loss(h_states[layer_idx], target_v) * current_alpha
                
                ce_loss = F.cross_entropy(logits.squeeze(1), targets[:, t])
                total_loss += (ce_loss + geo_loss)

        scaler.scale(total_loss / SEQ_LEN).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if i % 100 == 0:
            avg_loss = total_loss.item() / SEQ_LEN
            # 测试文本改为红楼梦风格的 Prompt
            test_prompt = "宝玉听了，忙笑道："
            sample = generate_sample(model, test_prompt, 40)
            print(f"步数 {i:5d} | Alpha: {current_alpha:.2f} | Loss: {avg_loss:.4f}")
            print(f"� {sample}")
            print("-" * 60)
        if i % 1000 == 0:
            checkpoint = {
                'it': i,
                'model': model.state_dict(),
                'vocab': chars, # 必须带上词库
                'cfg': {'hidden': HIDDEN_SIZE, 'layers': NUM_LAYERS}
            }
            torch.save(checkpoint, f"VGT_HLM_Step_{i}.pt")
            print(f"� 模型已存档：VGT_HLM_Step_{i}.pt")

def generate_sample(model, start_str, length=50):
    model.eval()
    with torch.no_grad():
        h = [torch.zeros(1, 1, HIDDEN_SIZE).to(device) for _ in range(NUM_LAYERS)]
        # 过滤掉不在词库里的字符，防止报错
        safe_start = [c for c in start_str if c in char_to_ix]
        if not safe_start: safe_start = [text[0]]
        ixs = [char_to_ix[c] for c in safe_start]
        
        for idx in ixs[:-1]:
            _, h = model.forward_step(model.emb(torch.tensor([[idx]]).to(device)), h)
        
        curr, res = ixs[-1], "".join(safe_start)
        for _ in range(length):
            logits, h = model.forward_step(model.emb(torch.tensor([[curr]]).to(device)), h)
            probs = F.softmax(logits.squeeze(1) / 0.7, dim=-1) # 降低温度，增加稳定性
            curr = torch.multinomial(probs[0], 1).item()
            res += ix_to_char[curr]
    return res

if __name__ == "__main__":
    train_hlm_vgt()