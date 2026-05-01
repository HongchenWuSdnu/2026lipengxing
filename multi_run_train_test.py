
import os
import json
import random
import argparse
from statistics import mean, stdev

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

try:
    import timm
except Exception:
    timm = None

from models import GlobalFeatureEnhancement, LocalFeatureEnhancement


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def binary_metrics_from_counts(tp, tn, fp, fn):
    acc = safe_div(tp + tn, tp + tn + fp + fn)

    prec_fake = safe_div(tp, tp + fp)
    rec_fake = safe_div(tp, tp + fn)
    f1_fake = safe_div(2 * prec_fake * rec_fake, prec_fake + rec_fake)

    prec_real = safe_div(tn, tn + fn)
    rec_real = safe_div(tn, tn + fp)
    f1_real = safe_div(2 * prec_real * rec_real, prec_real + rec_real)

    macro_f1 = (f1_fake + f1_real) / 2.0

    return {
        "accuracy": acc,
        "precision_fake": prec_fake,
        "recall_fake": rec_fake,
        "f1_fake": f1_fake,
        "precision_real": prec_real,
        "recall_real": rec_real,
        "f1_real": f1_real,
        "macro_f1": macro_f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


def build_model(model_type: str, num_classes: int = 2):
    if model_type == "resnet50_baseline":
        model = models.resnet50(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    elif model_type == "resnet50_gfe":
        backbone = models.resnet50(pretrained=True)
        gfe = GlobalFeatureEnhancement(channels=2048)
        classifier = nn.Linear(backbone.fc.in_features, num_classes)
        return nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
            gfe, backbone.avgpool, nn.Flatten(), classifier
        )

    elif model_type == "resnet50_lfe":
        backbone = models.resnet50(pretrained=True)
        lfe = LocalFeatureEnhancement(channels=2048)
        classifier = nn.Linear(backbone.fc.in_features, num_classes)
        return nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
            lfe, backbone.avgpool, nn.Flatten(), classifier
        )

    elif model_type == "resnet50_gfe_lfe":
        backbone = models.resnet50(pretrained=True)
        gfe = GlobalFeatureEnhancement(channels=2048)
        lfe = LocalFeatureEnhancement(channels=2048)
        classifier = nn.Linear(backbone.fc.in_features, num_classes)
        return nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
            gfe, lfe, backbone.avgpool, nn.Flatten(), classifier
        )

    elif model_type == "xception_baseline":
        if timm is None:
            raise ImportError("xception 需要先安装 timm: pip install timm")
        return timm.create_model("xception", pretrained=False, num_classes=num_classes)

    elif model_type == "xception_gfe":
        if timm is None:
            raise ImportError("xception 需要先安装 timm: pip install timm")
        backbone = timm.create_model("xception", pretrained=False)
        backbone.reset_classifier(0)
        gfe = GlobalFeatureEnhancement(channels=2048)
        classifier = nn.Linear(2048, num_classes)

        class XceptionGFE(nn.Module):
            def __init__(self, backbone, gfe, classifier):
                super().__init__()
                self.backbone = backbone
                self.gfe = gfe
                self.pool = nn.AdaptiveAvgPool2d((1, 1))
                self.classifier = classifier

            def forward(self, x):
                x = self.backbone.forward_features(x)
                x = self.gfe(x)
                x = self.pool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x

        return XceptionGFE(backbone, gfe, classifier)

    else:
        raise ValueError(f"未知 model_type: {model_type}")


@torch.no_grad()
def evaluate(model, dataloader, device, class_to_idx):
    model.eval()
    fake_idx = class_to_idx["fake"]
    real_idx = class_to_idx["real"]

    tp = tn = fp = fn = 0
    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1)

        for y_true, y_pred in zip(labels.cpu().tolist(), preds.cpu().tolist()):
            if y_true == fake_idx and y_pred == fake_idx:
                tp += 1
            elif y_true == real_idx and y_pred == real_idx:
                tn += 1
            elif y_true == real_idx and y_pred == fake_idx:
                fp += 1
            elif y_true == fake_idx and y_pred == real_idx:
                fn += 1

    return binary_metrics_from_counts(tp, tn, fp, fn)


def train_one_run(model_type, train_dir, test_dir, seed, batch_size, num_epochs, lr, device):
    set_seed(seed)

    transform = get_transforms()
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = build_model(model_type, num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    epoch_train_accs = []

    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total = 0
        running_loss = 0.0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = safe_div(correct, total)
        epoch_train_accs.append(train_acc)
        print(f"[Seed {seed}] Epoch {epoch+1}/{num_epochs} | Loss={running_loss:.4f} | TrainAcc={train_acc*100:.2f}%")

    test_metrics = evaluate(model, test_loader, device, test_dataset.class_to_idx)
    test_metrics["final_train_acc"] = epoch_train_accs[-1]
    test_metrics["best_train_acc"] = max(epoch_train_accs)
    test_metrics["epoch_train_accs"] = epoch_train_accs
    return test_metrics


def fmt_pct(x):
    return f"{x*100:.2f}"


def summarize(all_results):
    keys = [
        "accuracy", "macro_f1", "f1_fake", "f1_real",
        "precision_fake", "recall_fake", "precision_real", "recall_real",
        "final_train_acc", "best_train_acc"
    ]
    summary = {}
    for k in keys:
        vals = [r[k] for r in all_results]
        summary[k] = {
            "mean": mean(vals),
            "std": stdev(vals) if len(vals) > 1 else 0.0
        }
    return summary


def save_report(out_path, model_type, seeds, all_results, summary):
    lines = []
    lines.append("=" * 72)
    lines.append(f"Model Type: {model_type}")
    lines.append(f"Seeds     : {seeds}")
    lines.append("=" * 72)

    for i, r in enumerate(all_results):
        lines.append(f"[Run {i+1}]")
        lines.append(f"Accuracy        : {fmt_pct(r['accuracy'])}%")
        lines.append(f"Macro-F1        : {fmt_pct(r['macro_f1'])}%")
        lines.append(f"Fake F1         : {fmt_pct(r['f1_fake'])}%")
        lines.append(f"Real F1         : {fmt_pct(r['f1_real'])}%")
        lines.append(f"Fake Precision  : {fmt_pct(r['precision_fake'])}%")
        lines.append(f"Fake Recall     : {fmt_pct(r['recall_fake'])}%")
        lines.append(f"Real Precision  : {fmt_pct(r['precision_real'])}%")
        lines.append(f"Real Recall     : {fmt_pct(r['recall_real'])}%")
        lines.append(f"Final Train Acc : {fmt_pct(r['final_train_acc'])}%")
        lines.append(f"Best Train Acc  : {fmt_pct(r['best_train_acc'])}%")
        lines.append(f"Confusion       : TP={r['tp']} TN={r['tn']} FP={r['fp']} FN={r['fn']}")
        lines.append("-" * 72)

    lines.append("[Mean ± Std]")
    for k, v in summary.items():
        lines.append(f"{k:16s}: {fmt_pct(v['mean'])}% ± {fmt_pct(v['std'])}%")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    json_path = out_path.replace(".txt", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_type": model_type,
            "seeds": seeds,
            "runs": all_results,
            "summary": summary
        }, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=[
        "resnet50_baseline", "resnet50_gfe", "resnet50_lfe",
        "resnet50_gfe_lfe", "xception_baseline", "xception_gfe"
    ])
    parser.add_argument("--train_dir", type=str, default="dataset_frames/train")
    parser.add_argument("--test_dir", type=str, default="dataset_frames/test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 42, 2026])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_txt", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    all_results = []

    for seed in args.seeds:
        result = train_one_run(
            model_type=args.model_type,
            train_dir=args.train_dir,
            test_dir=args.test_dir,
            seed=seed,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            lr=args.lr,
            device=device
        )
        all_results.append(result)

    summary = summarize(all_results)
    save_report(args.out_txt, args.model_type, args.seeds, all_results, summary)

    print("\n" + "=" * 72)
    print("Final Mean ± Std")
    print("=" * 72)
    print(f"Accuracy        : {fmt_pct(summary['accuracy']['mean'])}% ± {fmt_pct(summary['accuracy']['std'])}%")
    print(f"Macro-F1        : {fmt_pct(summary['macro_f1']['mean'])}% ± {fmt_pct(summary['macro_f1']['std'])}%")
    print(f"Fake F1         : {fmt_pct(summary['f1_fake']['mean'])}% ± {fmt_pct(summary['f1_fake']['std'])}%")
    print(f"Real F1         : {fmt_pct(summary['f1_real']['mean'])}% ± {fmt_pct(summary['f1_real']['std'])}%")
    print(f"Fake Precision  : {fmt_pct(summary['precision_fake']['mean'])}% ± {fmt_pct(summary['precision_fake']['std'])}%")
    print(f"Fake Recall     : {fmt_pct(summary['recall_fake']['mean'])}% ± {fmt_pct(summary['recall_fake']['std'])}%")
    print(f"Real Precision  : {fmt_pct(summary['precision_real']['mean'])}% ± {fmt_pct(summary['precision_real']['std'])}%")
    print(f"Real Recall     : {fmt_pct(summary['recall_real']['mean'])}% ± {fmt_pct(summary['recall_real']['std'])}%")
    print(f"Final Train Acc : {fmt_pct(summary['final_train_acc']['mean'])}% ± {fmt_pct(summary['final_train_acc']['std'])}%")
    print(f"Best Train Acc  : {fmt_pct(summary['best_train_acc']['mean'])}% ± {fmt_pct(summary['best_train_acc']['std'])}%")

if __name__ == "__main__":
    main()
