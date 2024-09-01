import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.io import read_image
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader,TensorDataset
from ..data_utils import get_acdc,convert_masks
# UNet模型定义
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def up_conv(in_channels, out_channels):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.encoder1 = conv_block(in_channels, 64)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = conv_block(512, 1024)

        self.upconv4 = up_conv(1024, 512)
        self.decoder4 = conv_block(1024, 512)
        self.upconv3 = up_conv(512, 256)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = up_conv(256, 128)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = up_conv(128, 64)
        self.decoder1 = conv_block(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)



# 训练函数
def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


# 验证函数
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            running_loss += loss.item()

    return running_loss / len(loader)


# 计算Dice系数的函数
def dice_coefficient(preds, targets, threshold=0.5):
    preds = (preds > threshold).float()
    smooth = 1e-6
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)


# 主函数
def main():
    # 设置超参数和路径
    image_dir = 'path/to/images'
    mask_dir = 'path/to/masks'
    batch_size = 8
    num_epochs = 25
    learning_rate = 1e-4

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 定义数据转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # 加载数据集
    # Training data
    train_data, _, _ = get_acdc('ACDC/training', input_size=(args.img_size, args.img_size, 1))
    train_dataset = ACDCTrainDataset(train_data[0], train_data[1], args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)

    # Validation data
    val_data, _, _ = get_acdc('ACDC/testing', input_size=(args.img_size, args.img_size, 1))
    val_data[1] = convert_masks(val_data[1])
    val_data[0] = np.transpose(val_data[0], (0, 3, 1, 2))
    val_data[1] = np.transpose(val_data[1], (0, 3, 1, 2))
    val_data[0] = torch.Tensor(val_data[0])
    val_data[1] = torch.Tensor(val_data[1])
    val_dataset = TensorDataset(val_data[0], val_data[1])
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)


    # 初始化模型、损失函数和优化器
    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 开始训练
    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # 保存模型
    torch.save(model.state_dict(), 'unet_acdc.pth')

    # 在验证集上计算平均Dice系数
    model.eval()
    dice_scores = []
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            dice = dice_coefficient(outputs, masks)
            dice_scores.append(dice.item())

    average_dice = np.mean(dice_scores)
    print(f'Average Dice Coefficient on validation set: {average_dice:.4f}')


if __name__ == "__main__":
    main()
