import torch
import torch.nn as nn

# class matmul(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x1, x2):
#         x = x1 @ x2
#         return x

class Attention(nn.Module):
    def __init__(self, img_size, dim, num_heads=8):
        super(Attention, self).__init__()
        self.pos_embbedding = nn.Parameter(torch.randn(1, img_size ** 2, dim))
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.mat = torch.matmul
        self.proj = nn.Linear(dim, dim)
        self.conv = nn.Conv2d(dim * 2, dim, (1, 1))

    def forward(self, content_feat, component_list):
        B, C, H, W = content_feat.shape
        # style = style_feat.reshape(B, H * W, C)
        component = []
        for i in range(len(component_list)):
            comp = component_list[i].reshape(B, H * W, C)
            component.append(comp)
        content = content_feat.reshape(B, H * W, C)

        # style_token = style + self.pos_embbedding[:, :(H * W)]
        comp_token = []
        for i in range(len(component)):
            token = component[i] + self.pos_embbedding[:, :(H * W)]
            comp_token.append(token)
        content_token = content + self.pos_embbedding[:, :(H * W)]
        # content_token = self.q(content_token)
        query = self.q(content_token).reshape(B, H*W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        keys = []
        values = []
        for i in range(len(comp_token)):
            kv = self.kv(comp_token[i]).reshape(B, H*W, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
            key, value = kv[0], kv[1]
            keys.append(key)
            values.append(value)

        # query = content_token.reshape(B, H*W, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # attention_map = (self.mat(query, key.transpose(-2, -1))) * self.scale
        result = 0
        for i in range(len(keys)):
            attn = (self.mat(query, keys[i].transpose(-2, -1))) * self.scale
            attn = attn.softmax(dim=-1)
            res = self.mat(attn, values[i]).transpose(1, 2).reshape(B, H*W, C)
            result = result + res

        s = result + content_token
        s = self.proj(s)
        s = s.reshape(B, C, H, W)
        feat_c_s = torch.cat((s, content_feat), dim=1)
        feat_c_s = self.conv(feat_c_s)

        return feat_c_s

# model = Attention(img_size=16, dim=512, num_heads=8)
# content = torch.randn((4, 512, 16, 16))
# style = torch.randn((4, 512, 16, 16))
# comp_list = []
# for i in range(15):
#     comp_list.append(style)
# out = model(content, comp_list)
# print(out.shape)
