import argparse
import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

try:
    import timm
except ImportError:
    timm = None

try:
    from models import GlobalFeatureEnhancement, LocalFeatureEnhancement
except ImportError as e:
    raise ImportError(
        "无法导入 models.py 里的 GlobalFeatureEnhancement / LocalFeatureEnhancement。"
        "请把本脚本和你的 models.py 放在同一目录下再运行。"
    ) from e


class XceptionGFE(nn.Module):
    def __init__(self, backbone: nn.Module, gfe: nn.Module, classifier: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.gfe = gfe
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.forward_features(x)
        x = self.gfe(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def build_model(model_type: str, num_classes: int = 2) -> nn.Module:
    if model_type == "resnet50_baseline":
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    if model_type == "resnet50_gfe":
        backbone = models.resnet50(pretrained=False)
        gfe = GlobalFeatureEnhancement(channels=2048)
        classifier = nn.Linear(backbone.fc.in_features, num_classes)
        return nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            gfe,
            backbone.avgpool,
            nn.Flatten(),
            classifier,
        )

    if model_type == "resnet50_lfe":
        backbone = models.resnet50(pretrained=False)
        lfe = LocalFeatureEnhancement(channels=2048)
        classifier = nn.Linear(backbone.fc.in_features, num_classes)
        return nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            lfe,
            backbone.avgpool,
            nn.Flatten(),
            classifier,
        )

    if model_type == "resnet50_gfe_lfe":
        backbone = models.resnet50(pretrained=False)
        gfe = GlobalFeatureEnhancement(channels=2048)
        lfe = LocalFeatureEnhancement(channels=2048)
        classifier = nn.Linear(backbone.fc.in_features, num_classes)
        return nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
            gfe,
            lfe,
            backbone.avgpool,
            nn.Flatten(),
            classifier,
        )

    if model_type == "xception_baseline":
        if timm is None:
            raise ImportError("需要先安装 timm：pip install timm")
        return timm.create_model("xception", pretrained=False, num_classes=num_classes)

    if model_type == "xception_gfe":
        if timm is None:
            raise ImportError("需要先安装 timm：pip install timm")
        backbone = timm.create_model("xception", pretrained=False)
        backbone.reset_classifier(0)
        gfe = GlobalFeatureEnhancement(channels=2048)
        classifier = nn.Linear(2048, num_classes)
        return XceptionGFE(backbone, gfe, classifier)

    raise ValueError(f"不支持的 model_type: {model_type}")


def load_checkpoint(model: nn.Module, checkpoint_path: str, model_type: str) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 你的 baseline / xception_baseline 保存的是 dict，其余大多是纯 state_dict
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("[警告] missing keys:", missing)
    if unexpected:
        print("[警告] unexpected keys:", unexpected)

    # 对 GFE+LFE 的特殊提醒：你的训练脚本把它误存成了 resnet50_gfe.pth
    if model_type == "resnet50_gfe_lfe" and os.path.basename(checkpoint_path) == "resnet50_gfe.pth":
        print("[提醒] 你传入的是 resnet50_gfe.pth。请确认这是不是被 train_gfe_lfe.py 覆盖后的 GFE+LFE 权重。")

    return model


def compute_binary_metrics(tp: int, tn: int, fp: int, fn: int) -> Dict[str, float]:
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate(model: nn.Module, dataloader: DataLoader, device: torch.device) -> Tuple[Dict[str, float], Dict[str, int]]:
    model.eval()
    tp = tn = fp = fn = 0

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1)

            tp += ((preds == 1) & (labels == 1)).sum().item()
            tn += ((preds == 0) & (labels == 0)).sum().item()
            fp += ((preds == 1) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels == 1)).sum().item()

    metrics = compute_binary_metrics(tp, tn, fp, fn)
    counts = {"TP": tp, "TN": tn, "FP": fp, "FN": fn}
    return metrics, counts


def main():
    parser = argparse.ArgumentParser(description="Deepfake 独立测试集评估脚本")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=[
            "resnet50_baseline",
            "resnet50_gfe",
            "resnet50_lfe",
            "resnet50_gfe_lfe",
            "xception_baseline",
            "xception_gfe",
        ],
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="训练好的权重路径")
    parser.add_argument("--test_dir", type=str, default="dataset_frames/test", help="测试集根目录")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_txt", type=str, default="", help="可选：把结果写入 txt 文件")
    args = parser.parse_args()

    device = torch.device(args.device)

    if not os.path.isdir(args.test_dir):
        raise FileNotFoundError(f"测试集目录不存在: {args.test_dir}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_dataset = datasets.ImageFolder(args.test_dir, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(args.model_type, num_classes=2)
    model = load_checkpoint(model, args.checkpoint, args.model_type)
    model.to(device)

    metrics, counts = evaluate(model, test_loader, device)

    lines = []
    lines.append("=" * 60)
    lines.append(f"Model Type   : {args.model_type}")
    lines.append(f"Checkpoint   : {args.checkpoint}")
    lines.append(f"Test Dir     : {args.test_dir}")
    lines.append(f"Classes      : {test_dataset.classes}")
    lines.append(f"Num Samples  : {len(test_dataset)}")
    lines.append("-" * 60)
    lines.append(f"Accuracy     : {metrics['accuracy'] * 100:.2f}%")
    lines.append(f"Precision    : {metrics['precision'] * 100:.2f}%")
    lines.append(f"Recall       : {metrics['recall'] * 100:.2f}%")
    lines.append(f"F1-score     : {metrics['f1'] * 100:.2f}%")
    lines.append(f"Confusion    : TP={counts['TP']}  TN={counts['TN']}  FP={counts['FP']}  FN={counts['FN']}")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)

    if args.save_txt:
        with open(args.save_txt, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"结果已保存到: {args.save_txt}")


if __name__ == "__main__":
    main()
