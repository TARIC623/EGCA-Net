import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# 基础组件 (标准的 YOLOv8 Conv 模块)
# ================================================================
def autopad(k, p=None, d=1):
    """自动计算Padding以保持输出尺寸不变"""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """标准的 Conv-BN-SiLU 模块"""
    default_act = nn.SiLU(inplace=True)

    # 使用关键字参数传递 act，防止错传给 d
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))



# ================================================================
class ESFF(nn.Module):
    """

    利用"均值+标准差"双重统计量描述高频信号特征，
    并通过一个小型 MLP 网络智能学习门控权重。
    融合方式保持轻量级的可学习加权求和。
    """

    def __init__(self, c1, c2=None, k=3, s=1, p=None, g=1, act=True):
        super().__init__()
        self.c1 = c1
        self.c2 = c2 if c2 is not None else c1

        # 1. 区域感知器 (轻量级)
        self.region_mask = nn.Sequential(
            Conv(c1, c1 // 4, 3, 1),
            nn.Conv2d(c1 // 4, 1, 1, bias=True),
            nn.Sigmoid()
        )

        # 2. 频域分离组件 (低通滤波器)
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        # 3. 【核心改进】MLP 智能门控网络 (替代了原有的 mu/sigma 参数)
        # 输入是双重统计量 (均值 + 标准差)，所以输入通道是 2 * c1
        # 使用 1x1 卷积实现通道间的全连接层
        self.mlp_gate = nn.Sequential(
            nn.Conv2d(2 * c1, c1, 1),  # 隐藏层保持 c1 维度以保证拟合能力
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c1, 1),  # 输出层，每个通道一个权重
            nn.Sigmoid()  # 将权重限制在 0~1 之间
        )

        # 4. 基础空域分支
        self.spatial_conv = Conv(c1, c1, k, s, p, g, act=act)

        # 5. 可学习融合权重 (保持轻量化融合)
        self.merge_weights = nn.Parameter(torch.zeros(2, c1, 1, 1))

        # 6. 最终通道调整
        self.final_conv = Conv(c1, self.c2, 1, 1) if c1 != self.c2 else nn.Identity()

    def forward(self, x):
        # x: [B, C_in, H, W]

        # --- 步骤 1: 生成区域掩码并聚焦前景 ---
        mask = self.region_mask(x)
        x_masked = x * mask

        # --- 步骤 2: 高频分量提取 ---
        feat_low = self.avg_pool(x_masked)
        feat_high = x_masked - feat_low

        # --- 步骤 3: 【核心改进】计算双重区域统计量 ---
        # 只统计掩码区域内的高频信号
        feat_high_in_region = feat_high * mask

        # 统计量 A: 标准差 (反映信号波动性/纹理丰富度)
        region_std = torch.std(feat_high_in_region, dim=[2, 3], keepdim=True)

        # 统计量 B: 绝对值均值 (反映信号整体强度/能量)
        # 取绝对值是因为高频信号在0附近波动，直接求均值接近0，无法反映强度
        region_mean = torch.mean(torch.abs(feat_high_in_region), dim=[2, 3], keepdim=True)

        # --- 步骤 4: 【核心改进】MLP 智能门控 ---
        # 拼接两个统计特征, shape: [B, 2*C, 1, 1]
        dual_stats = torch.cat([region_mean, region_std], dim=1)

        # 输入 MLP 生成门控权重, shape: [B, C, 1, 1]
        gate_weight = self.mlp_gate(dual_stats)

        # 清洗高频特征并重构频域分支
        feat_high_clean = feat_high * gate_weight
        feat_freq = feat_low + feat_high_clean

        # --- 步骤 5: 可学习加权求和融合 (保持不变) ---
        feat_spatial = self.spatial_conv(x_masked)
        weights = F.softmax(self.merge_weights, dim=0)
        alpha, beta = weights[0], weights[1]
        fused_feat = alpha * feat_spatial + beta * feat_freq
        out = self.final_conv(fused_feat)

        return out