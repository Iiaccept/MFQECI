# -*- coding: utf-8 -*-
import os
from tqdm import tqdm
from PIL import Image
import torch
from torchvision import transforms
from imageencoder import ImageEncoder  # 自定义 model 文件

# ================== 参数设置 ==================
device = "cuda" if torch.cuda.is_available() else "cpu"
img_dir = "img"  # 图片文件夹
save_path = "image_features_enhanced1118.pt"
pretrained_path = "ImageMol.pth.tar"  # ImageMol 预训练权重路径
image_size = 224  # 与 ImageMol 一致
# ============================================


# ========== 图像预处理函数 ==========
def load_norm_transform(image_size=224):
    """
    定义图像增强（不含normalize），并单独返回normalize函数
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_transforms = transforms.Compose([
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(degrees=360),
        transforms.ToTensor()
    ])
    return img_transforms, normalize


# ========== 初始化模型 ==========
img_transforms, normalize = load_norm_transform(image_size)
model = ImageEncoder().to(device)

# ========== 加载 ImageMol 权重并自动映射层名 ==========
if os.path.exists(pretrained_path):
    checkpoint = torch.load(pretrained_path, map_location=device)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model_sd = model.state_dict()

    new_state_dict = {}
    for k, v in state_dict.items():
        # 自动映射 embedding_layer -> img_encoder
        if k.startswith("embedding_layer."):
            new_k = k.replace("embedding_layer.", "img_encoder.")
            new_state_dict[new_k] = v
        elif k in model_sd and v.shape == model_sd[k].shape:
            new_state_dict[k] = v

    matched_keys, unmatched_keys = [], []
    for k in model_sd.keys():
        if k in new_state_dict and model_sd[k].shape == new_state_dict[k].shape:
            model_sd[k] = new_state_dict[k]
            matched_keys.append(k)
        else:
            unmatched_keys.append((k, new_state_dict[k].shape if k in new_state_dict else None, model_sd[k].shape))

    model.load_state_dict(model_sd, strict=False)
    print(f"✅ 成功匹配 {len(matched_keys)}/{len(model_sd)} 个参数")
    if unmatched_keys:
        print(f"⚠️ 有 {len(unmatched_keys)} 个参数未匹配，前10个如下：")
        for i, (k, ckps, curs) in enumerate(unmatched_keys[:10]):
            print(f"  [{i + 1}] {k}: checkpoint {ckps} -> model {curs}")
else:
    print(f"❌ 未找到预训练文件: {pretrained_path}")

model.eval()

# ========== 批量提取特征 ==========
image_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

features, file_names = [], []

with torch.no_grad():
    for idx, path in enumerate(tqdm(image_paths, desc="Extracting features")):
        img = Image.open(path).convert("RGB")
        img_tensor = img_transforms(img).unsqueeze(0).to(device)
        img_tensor = normalize(img_tensor.squeeze(0)).unsqueeze(0)  # 手动归一化

        if idx == 0:  # 验证是否生效
            print(f"\n🔍 [{os.path.basename(path)}] 归一化后张量范围:",
                  torch.min(img_tensor).item(), "→", torch.max(img_tensor).item())

        feat = model(img_tensor)  # 输出 [1, 512]
        features.append(feat.cpu())
        file_names.append(os.path.basename(path))

features = torch.cat(features, dim=0)
print("✅ 特征矩阵形状:", features.shape)

# ========== 保存结果 ==========
torch.save({"features": features, "filenames": file_names}, save_path)
print(f"💾 特征已保存到 {save_path}")
