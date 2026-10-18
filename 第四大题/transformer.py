import torch
import torch.nn as nn

batch_size = 2
seq_len = 4
embed_dim = 8              #一个词用多少个数字来描述
num_heads = 2
num_layers = 2             #encoder堆叠几层

x = torch.randn(batch_size, seq_len, embed_dim)
print("Input shape:", x.shape)

encoder_layer = nn.TransformerEncoderLayer(
    d_model=embed_dim,     #模型维度，必须与输入最后一维一致
    nhead=num_heads,       #注意力头数
    batch_first=True       #输入格式为 [batch, seq, dim]（
)

encoder = nn.TransformerEncoder(   #堆叠
    encoder_layer,
    num_layers=num_layers
)

output = encoder(x)         #前向传播

print("Output shape:", output.shape)

expected_shape = (batch_size,seq_len,embed_dim)
assert output.shape == expected_shape,"输出shape不符合预期！"
print("输出shape符合预期：Transformer Encoder不改变输入 [batch, seq, embed] 形状")

print("\n----- 实验：把num_layers 从2改为4 -----")

num_layers = 4

encoder_layers2 = nn.TransformerEncoderLayer(
    d_model=embed_dim,
    nhead=num_heads,
    batch_first=True
)
encoder2 = nn.TransformerEncoder(
    encoder_layers2,
    num_layers=num_layers
)
output2 = encoder2(x)

print("Input shape :", x.shape)        # torch.Size([2, 4, 8])
print("Output shape:", output2.shape)  # 仍然是 torch.Size([2, 4, 8])
assert output2.shape == expected_shape,"输出shape不符合预期！"
print("结论：修改 num_layers（层数）只改变模型深度/参数量，不改变输出 shape。")
