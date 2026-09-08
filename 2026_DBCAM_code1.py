import os
import time
import gc
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights, DenseNet121_Weights, Inception_V3_Weights, ViT_B_16_Weights, DenseNet169_Weights, DenseNet201_Weights
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
plt.rcParams["axes.unicode_minus"] = False
import PIL.Image as Image
from PIL import ImageEnhance
from sklearn.model_selection import StratifiedKFold
import traceback
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 使用设备：{device} | GPU数量：{torch.cuda.device_count()}")

# ======================== 基础配置 ========================
class Config:
    n_folds = 5  #分层5折
    img_size = 224  #图像尺寸
    lr = 1e-4  #初始学习率
    batch_size = 32  # 批次大小
    epochs = 25  # 训练轮数
    label_root_dir = r"E:\ResNet\ResNet\Rabbit Coccidia\Data"  #图像总目录，12个类别文件夹的父目录
    
    progress_interval = 5  #每5个批次打印进度
    img_formats = ['.jpg', '.jpeg', '.png', '.bmp']
    modelname = "Ours"   # 目前改进的模型用Ours调用，采取的是DenseNet169+全局-局部通道注意力融合的结构
    num_classes = 12
    preload_images = True  
    

    minority_threshold = 100  
    minority_aug_times = 5  
    

    lr_scheduler_monitor = "val_auc"  #'val_loss'/'val_auc'，推荐使用val_auc
    lr_scheduler_patience = 3
    lr_scheduler_factor = 0.5
    lr_scheduler_min_lr = 1e-6
    lr_scheduler_threshold = 1e-5    
    result_save_dir = f"313_Rabbit_Coccidia/{modelname}-{n_folds}fold-{minority_aug_times}-{lr_scheduler_monitor}_{batch_size}-{epochs}-{lr}-1"


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

class DBCAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(DBCAM, self).__init__()
        self.in_channels = in_channels
        
        self.se_pool = nn.AdaptiveAvgPool2d(1)
        self.se_pool1 = nn.AdaptiveMaxPool2d(1)
        self.se_conv1 = nn.Conv2d(in_channels, in_channels//reduction, 1, bias=False)
        self.se_conv2 = nn.Conv2d(in_channels//reduction, in_channels, 1, bias=False)
        
        self.eca_pool = nn.AdaptiveAvgPool2d(1)
        self.eca_pool1 = nn.AdaptiveMaxPool2d(1)
        gamma, b = 2, 1
        t = int(abs(math.log2(in_channels) + b) / gamma)
        self.eca_k = t if t % 2 != 0 else t + 1
        self.eca_conv = nn.Conv1d(1, 1, kernel_size=self.eca_k, padding=(self.eca_k-1)//2, bias=False)
        

        self.fusion_conv = nn.Conv2d(2 * in_channels, in_channels, 1, bias=False)  
        self.fusion_bn = nn.BatchNorm2d(in_channels)                              
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()
        se = self.se_pool(x) +self.se_pool1(x)
        se = self.relu(self.se_conv1(se))
        se_att = self.sigmoid(self.se_conv2(se))  

        eca = self.eca_pool(x).view(B, 1, C) + self.eca_pool1(x).view(B, 1, C)
        eca = self.eca_conv(eca).view(B, C, 1, 1)
        eca_att = self.sigmoid(eca)  

        channel_att = self.sigmoid(se_att + eca_att)      
        
        x = x * channel_att
        return x

def densenet169split(num_classes):
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
    
    dbcam1 = DBCAM(in_channels=128)
    dbcam2 = DBCAM(in_channels=256)
    dbcam3 = DBCAM(in_channels=640)
    
    new_features = nn.Sequential(
        conv0, norm0, relu0, pool0,
        denseblock1, transition1, dbcam1,
        denseblock2, transition2, dbcam2,
        denseblock3, transition3, dbcam3,
        denseblock4, norm5
    )
    model.features = new_features
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    
    for m in [dbcam1, dbcam2, dbcam3]:
        for sub_m in m.modules():
            if isinstance(sub_m, nn.Conv2d) or isinstance(sub_m, nn.Conv1d):
                nn.init.kaiming_normal_(sub_m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(sub_m, nn.BatchNorm2d):
                nn.init.constant_(sub_m.weight, 1)
                nn.init.constant_(sub_m.bias, 0)
    return model
    
def get_model(model_name, num_classes):
    model_dict = {
        "ResNet34": lambda: models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1),
        "ResNet50": lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
        "MobileNetV2": lambda: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1),
        "EfficientNet-B0": lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1),
        "MobileNetV3-Large": lambda: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1),
        "ShuffleNetV2": lambda: models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1),
        "EfficientNetV2-S": lambda: models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1),
        "DenseNet121": lambda: models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1),
        "DenseNet169": lambda: models.densenet169(weights=DenseNet169_Weights.IMAGENET1K_V1),
        "ViT": lambda: models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1),
        "Ours": lambda: densenet169split(num_classes=12)
        }

    if model_name not in model_dict:
        raise ValueError(f"❌ 未知模型！可选模型：{list(model_dict.keys())}")
    
    model = model_dict[model_name]()
    

    if model_name.startswith("ResNet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "MobileNetV2":
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "EfficientNet-B0" or model_name == "EfficientNetV2-S":
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "MobileNetV3-Large":
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name == "ShuffleNetV2":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name in ["DenseNet121", "DenseNet169", "DenseNet201"]:
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "ViT":
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    
    return model.to(device)

class MinorityAugmentation:
    def __init__(self, img_size):
        self.img_size = img_size
        self.base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.minority_aug_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.8),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def augment_minority_image(self, img):
        return self.minority_aug_transform(img)
    
    def base_process(self, img):
        return self.base_transform(img)


def plot_multiclass_roc(all_labels, all_probs, save_dir, class_names, fold_idx):
    plt.figure(figsize=(6, 5))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', 
              '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78']
    all_probs = np.array(all_probs)
    
    for i, class_name in enumerate(class_names):
        y_true_binary = (np.array(all_labels) == i).astype(int)
        y_prob_binary = all_probs[:, i]
        fpr, tpr, _ = roc_curve(y_true_binary, y_prob_binary)
        auc = roc_auc_score(y_true_binary, y_prob_binary)
        plt.plot(fpr, tpr, color=colors[i], lw=2, 
                 label=f'{class_name} (AUC = {auc:.4f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=10)
    plt.ylabel('True Positive Rate (TPR)', fontsize=10)
    plt.title(f'Multiclass ROC Curve ({Config.modelname})', fontsize=14)
    plt.legend(loc='lower right', fontsize=8)
    plt.grid(alpha=0.3)
    
    roc_save_path = os.path.join(save_dir, f"multiclass_roc_curve.png")
    plt.tight_layout()
    plt.savefig(roc_save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✅ 多分类ROC曲线保存至：{roc_save_path}")
    
def plot_training_curves(history, save_dir, fold_idx):

    epochs = range(1, len(history['train_loss']) + 1) 
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    

    if Config.lr_scheduler_monitor == 'val_auc':
        # Loss曲线
        ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
        ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss')
        ax1.set_title(f'Loss Curve ({Config.modelname})', fontsize=12)
        ax1.set_xlabel('Epoch', fontsize=10)
        ax1.set_ylabel('Loss', fontsize=10)
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
        ax1.set_xticks(epochs)  
        
        # Accuracy曲线
        ax2.plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Train Acc')
        ax2.plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Val Acc')
        ax2.set_title(f'Accuracy Curve ({Config.modelname})', fontsize=12)
        ax2.set_xlabel('Epoch', fontsize=10)
        ax2.set_ylabel('Accuracy', fontsize=10)
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
        ax2.set_xticks(epochs)  
        
        # AUC曲线
        ax3.plot(epochs, history['val_auc'], 'g-', linewidth=2, label='Val AUC (Monitor)')
        ax3.set_title(f'Validation AUC Curve (Monitor) ({Config.modelname})', fontsize=12)
        ax3.set_xlabel('Epoch', fontsize=10)
        ax3.set_ylabel('AUC', fontsize=10)
        ax3.legend(fontsize=10)
        ax3.grid(alpha=0.3)
        ax3.set_xticks(epochs)  
        
        # 学习率曲线
        ax4.plot(epochs, history['lr'], 'orange', linewidth=2, label='Learning Rate')
        ax4.set_title(f'Learning Rate Curve ({Config.modelname})', fontsize=12)
        ax4.set_xlabel('Epoch', fontsize=10)
        ax4.set_ylabel('LR', fontsize=10)
        ax4.legend(fontsize=10)
        ax4.grid(alpha=0.3)
        ax4.set_xticks(epochs)  
    else:  
        # Loss曲线
        ax1.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train Loss')
        ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val Loss (Monitor)')
        ax1.set_title(f'Loss Curve (Monitor) ({Config.modelname})', fontsize=12)
        ax1.set_xlabel('Epoch', fontsize=10)
        ax1.set_ylabel('Loss', fontsize=10)
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
        ax1.set_xticks(epochs)  
        
        # Accuracy曲线
        ax2.plot(epochs, history['train_acc'], 'b-', linewidth=2, label='Train Acc')
        ax2.plot(epochs, history['val_acc'], 'r-', linewidth=2, label='Val Acc')
        ax2.set_title(f'Accuracy Curve ({Config.modelname})', fontsize=12)
        ax2.set_xlabel('Epoch', fontsize=10)
        ax2.set_ylabel('Accuracy', fontsize=10)
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
        ax2.set_xticks(epochs)  
        
        # AUC曲线
        ax3.plot(epochs, history['val_auc'], 'g-', linewidth=2, label='Val AUC')
        ax3.set_title(f'Validation AUC Curve ({Config.modelname})', fontsize=12)
        ax3.set_xlabel('Epoch', fontsize=10)
        ax3.set_ylabel('AUC', fontsize=10)
        ax3.legend(fontsize=10)
        ax3.grid(alpha=0.3)
        ax3.set_xticks(epochs)  
        
        # 学习率曲线
        ax4.plot(epochs, history['lr'], 'orange', linewidth=2, label='Learning Rate')
        ax4.set_title(f'Learning Rate Curve ({Config.modelname})', fontsize=12)
        ax4.set_xlabel('Epoch', fontsize=10)
        ax4.set_ylabel('LR', fontsize=10)
        ax4.legend(fontsize=10)
        ax4.grid(alpha=0.3)
        ax4.set_xticks(epochs)  
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"training_curves_with_lr_fold{fold_idx+1}.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"✅ 训练曲线（含学习率）保存至：{save_path}")


def preload_all_images(instance_list, img_size, minority_classes=None, is_train=False):
    print(f"\n🔧 开始预加载图像（尺寸：{img_size}x{img_size}）{'[训练集-少数类增强]' if is_train else '[验证/测试集-无增强]'}")
    start_time = time.time()
    
    aug_tool = MinorityAugmentation(img_size)
    preloaded_data = []
    
    for idx, (img_path, mapped_label, img_name) in enumerate(instance_list):
        try:
            img = Image.open(img_path).convert("RGB")
            base_tensor = aug_tool.base_process(img)
            preloaded_data.append((base_tensor, mapped_label, img_name))
            

            if is_train and minority_classes is not None and mapped_label in minority_classes:
                for aug_idx in range(Config.minority_aug_times - 1): 
                    aug_tensor = aug_tool.augment_minority_image(img)
                    aug_img_name = f"{img_name}_aug{aug_idx+1}"
                    preloaded_data.append((aug_tensor, mapped_label, aug_img_name))
            
            if (idx + 1) % 1000 == 0:
                print(f"  已加载 {idx+1}/{len(instance_list)} 张图像（训练集增强后总数：{len(preloaded_data)}）")
        except Exception as e:
            print(f"⚠️  跳过损坏图像：{img_path}，错误：{str(e)}")
            continue
    

    total_memory_gb = len(preloaded_data) * img_size * img_size * 3 * 4 / (1024**3)
    print(f"\n✅ 预加载完成！共加载 {len(preloaded_data)} 张图像 | 内存占用：{total_memory_gb:.2f} GB")
    print(f"⏱️  预加载耗时：{time.time() - start_time:.2f} 秒")
    return preloaded_data


class PreloadedImageDataset(Dataset):
    def __init__(self, preloaded_data):
        self.preloaded_data = preloaded_data
    
    def __len__(self):
        return len(self.preloaded_data)
    
    def __getitem__(self, idx):
        img_tensor, mapped_label, img_name = self.preloaded_data[idx]
        return img_tensor, mapped_label, img_name

def collate_fn(batch):
    imgs, labels, img_names = zip(*batch)
    return torch.stack(imgs), torch.tensor(labels, dtype=torch.long), img_names

def load_and_split_data():
    all_raw_instances = []
    class_folders = sorted(os.listdir(Config.label_root_dir))
    label_map = {folder: idx for idx, folder in enumerate(class_folders)}
    
    for folder_name in class_folders:
        folder_path = os.path.join(Config.label_root_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        label = label_map[folder_name]
        
        for filename in os.listdir(folder_path):
            if (any(filename.lower().endswith(fmt) for fmt in Config.img_formats) and 
                "aug" not in filename.lower()):
                img_path = os.path.join(folder_path, filename)
                all_raw_instances.append((img_path, label, filename))
    

    class_raw_count = {}
    minority_classes = []
    for idx in range(Config.num_classes):
        count = len([inst for inst in all_raw_instances if inst[1] == idx])
        class_raw_count[idx] = count
        if count < Config.minority_threshold:
            minority_classes.append(idx)

    all_labels = [inst[1] for inst in all_raw_instances]  
    skf = StratifiedKFold(n_splits=Config.n_folds, shuffle=True, random_state=seed) 
    fold_splits = []
    

    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(all_raw_instances, all_labels)):
        train_val_inst = [all_raw_instances[i] for i in train_val_idx]
        train_val_labels = [all_labels[i] for i in train_val_idx]
        

        train_idx_sub, val_idx_sub = train_test_split(
            range(len(train_val_inst)),
            train_size=0.8,  
            random_state=seed,
            stratify=train_val_labels  
        )
        

        fold_train = [train_val_inst[i] for i in train_idx_sub]
        fold_val = [train_val_inst[i] for i in val_idx_sub]
        fold_test = [all_raw_instances[i] for i in test_idx] 
        
        fold_splits.append({
            "fold_idx": fold_idx,
            "train": fold_train,
            "val": fold_val,
            "test": fold_test,
            "class_dist": {
                "train": {i: len([x for x in fold_train if x[1]==i]) for i in range(Config.num_classes)},
                "test": {i: len([x for x in fold_test if x[1]==i]) for i in range(Config.num_classes)}
            }
        })
        

        print(f"\n===== 第{fold_idx+1}折（分层抽样）=====")
        print(f"训练集：{len(fold_train)}张 | 验证集：{len(fold_val)}张 | 测试集：{len(fold_test)}张")
        print(f"测试集类别分布：{fold_splits[-1]['class_dist']['test']}")
    
    return fold_splits, class_folders, minority_classes


def train(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch_idx, (imgs, labels, _) in enumerate(train_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        total_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
        total_samples += len(labels)
        
        if (batch_idx + 1) % Config.progress_interval == 0 or (batch_idx + 1) == len(train_loader):
            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples
            print(f"  批次 {batch_idx+1}/{len(train_loader)} | 损失：{avg_loss:.4f} | 准确率：{avg_acc:.4f}")
    
    return total_loss / total_samples, total_correct / total_samples

def validate(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels, _ in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item() * len(labels)
            preds = torch.argmax(outputs, dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += len(labels)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(F.softmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    print(f"===== 验证结束 | 损失：{avg_loss:.4f} | 准确率：{avg_acc:.4f} | AUC：{auc:.4f} =====")
    return avg_loss, avg_acc, auc


def test(model, test_loader, class_names, fold_idx, save_dir):
    model.eval()
    test_results = {"img_name": [], "true_label": [], "pred_class": [], "pred_prob": []}
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels, img_names in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            test_results["img_name"].extend(img_names)
            test_results["true_label"].extend(labels.numpy())
            test_results["pred_class"].extend(preds)
            test_results["pred_prob"].extend([str(p) for p in probs])
            
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    

    test_acc = accuracy_score(all_labels, all_preds)
    test_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    test_f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    test_f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    

    cls_report = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        zero_division=0,
        output_dict=True 
    )

    report_df = pd.DataFrame(cls_report).T
    report_df.to_csv(
        os.path.join(save_dir, f"classification_report_fold{fold_idx+1}.csv"),
        index=True, encoding="utf-8-sig"
    )

    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(all_labels, all_preds)
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - Fold{fold_idx+1} ({Config.modelname})", fontsize=14)
    plt.colorbar()
    plt.xticks(range(Config.num_classes), class_names, rotation=45, ha='right')
    plt.yticks(range(Config.num_classes), class_names)
    for i in range(Config.num_classes):
        for j in range(Config.num_classes):
            plt.text(j, i, cm[i,j], ha="center", va="center", color="white" if cm[i,j]>cm.max()/2 else "black")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"confusion_matrix_fold{fold_idx+1}.png"), dpi=300, bbox_inches='tight')
    plt.close()
    

    plot_multiclass_roc(all_labels, all_probs, save_dir, class_names, fold_idx=fold_idx)
    

    pd.DataFrame(test_results).to_csv(os.path.join(save_dir, f"test_results_fold{fold_idx+1}.csv"), index=False, encoding="utf-8-sig")
    
    return test_acc, test_auc, test_f1_macro, test_f1_weighted


def main():
    try:
        scheduler_mode = "min" if Config.lr_scheduler_monitor == "val_loss" else "max"
        os.makedirs(Config.result_save_dir, exist_ok=True)
        

        fold_splits, class_names, minority_classes = load_and_split_data()
        fold_metrics = []  
        

        for fold in fold_splits:
            fold_idx = fold["fold_idx"]
            fold_save_dir = os.path.join(Config.result_save_dir, f"Fold{fold_idx+1}")
            os.makedirs(fold_save_dir, exist_ok=True)
            
            print(f"\n{'='*80}")
            print(f"===== 分层5折 - 第{fold_idx+1}/{Config.n_folds}折 =====")
            

            train_data = preload_all_images(fold["train"], Config.img_size, minority_classes, is_train=True)
            val_data = preload_all_images(fold["val"], Config.img_size, is_train=False)
            test_data = preload_all_images(fold["test"], Config.img_size, is_train=False)
            

            train_dataset = PreloadedImageDataset(train_data)
            val_dataset = PreloadedImageDataset(val_data)
            test_dataset = PreloadedImageDataset(test_data)
            
            train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)
            test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=2)

            model = get_model(Config.modelname, Config.num_classes)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=Config.lr)
            scheduler = ReduceLROnPlateau(optimizer, mode=scheduler_mode, factor=Config.lr_scheduler_factor, patience=Config.lr_scheduler_patience)
            
            best_monitor_value = float('inf') if scheduler_mode == "min" else 0.0
            best_model_path = os.path.join(fold_save_dir, f"best_model_fold{fold_idx+1}.pth")
            train_history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_auc': [], 'lr': []}
            
            for epoch in range(Config.epochs):
                print(f"\n===== Epoch {epoch+1}/{Config.epochs} =====")
                current_lr = optimizer.param_groups[0]['lr']
                train_history['lr'].append(current_lr)
                

                train_loss, train_acc = train(model, train_loader, criterion, optimizer)
                val_loss, val_acc, val_auc = validate(model, val_loader, criterion)
                
                train_history['train_loss'].append(train_loss)
                train_history['train_acc'].append(train_acc)
                train_history['val_loss'].append(val_loss)
                train_history['val_acc'].append(val_acc)
                train_history['val_auc'].append(val_auc)
                
                current_monitor = val_loss if Config.lr_scheduler_monitor == 'val_loss' else val_auc
                if (scheduler_mode == "min" and current_monitor < best_monitor_value) or (scheduler_mode == "max" and current_monitor > best_monitor_value):
                    best_monitor_value = current_monitor
                    torch.save(model.state_dict(), best_model_path)
                

                scheduler.step(current_monitor)
            

            history_df = pd.DataFrame({
                'Epoch': range(1, len(train_history['train_loss'])+1), 
                'Train_Loss': train_history['train_loss'],
                'Train_Accuracy': train_history['train_acc'],
                'Val_Loss': train_history['val_loss'],
                'Val_Accuracy': train_history['val_acc'],
                'Val_AUC': train_history['val_auc'],
                'Learning_Rate': train_history['lr']
            })
            history_df.to_csv(
                os.path.join(fold_save_dir, f"training_history_fold{fold_idx+1}.csv"),
                index=False, encoding="utf-8-sig"
            )
            

            plot_training_curves(train_history, fold_save_dir, fold_idx=fold_idx)
            
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            test_acc, test_auc, test_f1_macro, test_f1_weighted = test(model, test_loader, class_names, fold_idx, fold_save_dir)
            
            fold_metrics.append({
                "Fold": fold_idx+1,
                "Test_Accuracy": test_acc,
                "Test_AUC": test_auc,
                "F1_Macro": test_f1_macro,
                "F1_Weighted": test_f1_weighted,
                "Best_Monitor_Value": best_monitor_value
            })
            
            del model, optimizer, scheduler
            torch.cuda.empty_cache()
            gc.collect()
        
        metrics_df = pd.DataFrame(fold_metrics)
        summary_df = pd.DataFrame({
            "Mean_Test_Acc": [metrics_df["Test_Accuracy"].mean()],
            "Std_Test_Acc": [metrics_df["Test_Accuracy"].std()],
            "Mean_Test_AUC": [metrics_df["Test_AUC"].mean()],
            "Std_Test_AUC": [metrics_df["Test_AUC"].std()],
            "Mean_F1_Macro": [metrics_df["F1_Macro"].mean()],
            "Std_F1_Macro": [metrics_df["F1_Macro"].std()]
        })
        

        metrics_df.to_csv(os.path.join(Config.result_save_dir, "5fold_stratified_metrics_detail.csv"), index=False, encoding="utf-8-sig")
        summary_df.to_csv(os.path.join(Config.result_save_dir, "5fold_stratified_metrics_summary.csv"), index=False, encoding="utf-8-sig")
        
        print(f"\n===== 分层5折交叉验证完成 =====")
        print(f"平均测试准确率：{summary_df['Mean_Test_Acc'].iloc[0]:.4f} ± {summary_df['Std_Test_Acc'].iloc[0]:.4f}")
        print(f"平均测试AUC：{summary_df['Mean_Test_AUC'].iloc[0]:.4f} ± {summary_df['Std_Test_AUC'].iloc[0]:.4f}")
        
    except Exception as e:
        print(f"❌ 错误：{str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    global minority_classes
    minority_classes = []
    
    start_time = time.time()
    main()
    total_time = (time.time() - start_time) / 60
    print(f"\n总运行时间：{total_time:.2f} 分钟")