import torch
import torch.nn as nn
import torch.nn.functional as F


# ================================================================
# 基础依赖组件
# ================================================================
def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """标准卷积块"""
    default_act = nn.SiLU(inplace=True)

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class SimAM(nn.Module):
    """无参空间能量注意力"""

    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):
        b, c, h, w = x.size()
        n = h * w - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        # 计算能量函数，评估空间重要性
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activaton(y)


# ================================================================
# 特制的瓶颈层：SCR_Bottleneck
# (内核：大核DW + 局部平滑 + SimAM，专治碎框)
# ================================================================
class SCR_Bottleneck(nn.Module):
    """
    Spatial Coherence Rectification Bottleneck.
    替代标准 C2f 中的 Bottleneck，用于在网络末端进行特征的空间连贯性矫正。
    缝合机制：[7x7大核DW卷积(全局连接) + 3x3局部平均池化(局部平滑)] -> SimAM(中心固化)
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels

        # 1. 降维
        self.cv1 = Conv(c1, c_, 1, 1)

        # --- 核心矫正机制 (并行双分支) ---
        # 分支 A: 大核深度卷积 (7x7) - 建立大范围空间连接
        # 强制使用 k=7，忽略传入的 k 参数
        self.dw_large = nn.Sequential(
            nn.Conv2d(c_, c_, kernel_size=7, stride=1, padding=3, groups=c_, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True)
        )

        # 分支 B: 局部平滑 (3x3 AvgPool) - 融合临近孤立特征点
        self.local_smooth = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            Conv(c_, c_, 1, 1)  # 接一个1x1调整通道
        )

        # 空间注意力 - 固化聚合后的目标中心
        self.simam = SimAM()
        # --------------------

        # 3. 升维
        self.cv2 = Conv(c_, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        x_in = self.cv1(x)

        # 并行执行
        x_context = self.dw_large(x_in)
        x_smooth = self.local_smooth(x_in)

        # 融合 + 注意力聚焦
        x_fused = self.simam(x_context + x_smooth)

        y = self.cv2(x_fused)
        return x + y if self.add else y


# ================================================================
# 主模块：SCR (Spatial Coherence Rectifier)
# 【重点】：名字叫 SCR，但代码结构是完整的 C2f
# ================================================================
class SCR(nn.Module):
    """
    Spatial Coherence Rectifier (SCR) module.
    Implementation: A C2f-style structure using specialized SCR_Bottlenecks.
    学术名称隐藏了其 C2f 的本质结构，用于在网络末端进行强力特征聚合。
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        # C2f 的标准输入输出卷积
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        # 【核心】：使用特制的 SCR_Bottleneck 堆叠 n 次
        # e=1.0 表示在瓶颈层内部不再进行额外的通道缩放
        self.m = nn.ModuleList(SCR_Bottleneck(self.c, self.c, shortcut, g, e=1.0) for _ in range(n))

    def forward(self, x):
        # C2f 的标准前向传播逻辑（分流、堆叠、聚合）
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
