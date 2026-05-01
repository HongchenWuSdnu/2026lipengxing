import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from models import GlobalFeatureEnhancement


# ========= 参数 =========
data_dir = "dataset_frames/train"
batch_size = 8
num_epochs = 5
lr = 1e-4
num_classes = 2
# =======================

device = torch.device("cpu")

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 数据集
dataset = datasets.ImageFolder(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

backbone = models.resnet50(pretrained=True)

gfe = GlobalFeatureEnhancement(channels=2048)

classifier = nn.Linear(backbone.fc.in_features, num_classes)

model = nn.Sequential(
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
    classifier
)

model.to(device)


# 损失函数 & 优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# ========= 训练 =========
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = correct / total * 100
    print(f"Epoch [{epoch+1}/{num_epochs}] "
          f"Loss: {running_loss:.4f} "
          f"Acc: {acc:.2f}%")

torch.save(model.state_dict(), "resnet50_gfe.pth")
