import torch
import torch.nn as nn
import time
from torchvision import models
from torchvision.models import DenseNet169_Weights
from thop import profile


# ==================== 模型定义部分 (提取自你的 DenseNet169+SE 代码) ====================
class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.in_channels = in_channels

        self.se_pool = nn.AdaptiveAvgPool2d(1)
        self.se_pool1 = nn.AdaptiveMaxPool2d(1)
        self.se_conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.se_conv2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()
        se = self.se_pool(x) + self.se_pool1(x)
        se = self.relu(self.se_conv1(se))
        se_att = self.sigmoid(self.se_conv2(se))

        x = x * se_att
        return x


def densenet169NoECA(num_classes):
    model = models.densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1)
    features = model.features
    conv0 = features[0]
    norm0 = features[1]
    relu0 = features[2]
    pool0 = features[3]
    denseblock1 = features[4]
    transition1 = features[5]
    denseblock2 = features[6]
    transition2 = features[7]
    denseblock3 = features[8]
    transition3 = features[9]
    denseblock4 = features[10]
    norm5 = features[11]

    cbam1 = CBAM(in_channels=128)
    cbam2 = CBAM(in_channels=256)
    cbam3 = CBAM(in_channels=640)

    new_features = nn.Sequential(
        conv0, norm0, relu0, pool0,
        denseblock1, transition1, cbam1,
        denseblock2, transition2, cbam2,
        denseblock3, transition3, cbam3,
        denseblock4, norm5
    )
    model.features = new_features
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    for m in [cbam1, cbam2, cbam3]:
        for sub_m in m.modules():
            if isinstance(sub_m, nn.Conv2d) or isinstance(sub_m, nn.Conv1d):
                nn.init.kaiming_normal_(sub_m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(sub_m, nn.BatchNorm2d):
                nn.init.constant_(sub_m.weight, 1)
                nn.init.constant_(sub_m.bias, 0)
    return model


# ==================== 评估核心模块 ====================
def evaluate_model_performance(model, input_size=(1, 3, 224, 224), device='cuda'):
    print(f"正在使用 {device} 评估模型...")
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(input_size).to(device)

    # 1. 计算 FLOPs 和 Params
    flops, params = profile(model, inputs=(dummy_input,), verbose=False)

    params_M = params / 1e6
    flops_G = flops / 1e9

    # 2. 计算 Latency (延迟)
    print("正在进行推理预热...")
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input)

    print("正在测量推理延迟...")
    iterations = 300
    if device == 'cuda':
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        times = torch.zeros(iterations)

        with torch.no_grad():
            for i in range(iterations):
                starter.record()
                _ = model(dummy_input)
                ender.record()

                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                times[i] = curr_time

        mean_time = times.mean().item()
        std_time = times.std().item()
    else:
        times = []
        with torch.no_grad():
            for _ in range(iterations):
                start_time = time.time()
                _ = model(dummy_input)
                times.append((time.time() - start_time) * 1000)

        mean_time = sum(times) / iterations
        std_time = torch.tensor(times).std().item()

    print("\n" + "=" * 50)
    print("模型性能评估报告 (DenseNet169 + SE)")
    print("=" * 50)
    print(f"输入尺寸     : {input_size}")
    print(f"Params (M)   : {params_M:.3f} M")
    print(f"FLOPs (G)    : {flops_G:.3f} G")
    print(f"Latency (ms) : {mean_time:.3f} ± {std_time:.3f} ms (基于 {iterations} 次迭代取平均)")
    print("=" * 50)


if __name__ == "__main__":
    num_classes = 12
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = densenet169NoECA(num_classes=num_classes)

    evaluate_model_performance(model, input_size=(1, 3, 224, 224), device=device)