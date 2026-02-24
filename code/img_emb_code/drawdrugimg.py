import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw

# ==============================
# 1. 基础配置
# ==============================
input_file = "drug.xlsx"
sheet_name = "Sheet1"
output_dir = "img"
os.makedirs(output_dir, exist_ok=True)

# ==============================
# 2. 读取数据
# ==============================
df = pd.read_excel(input_file, sheet_name=sheet_name)
df.columns = df.columns.str.lower()

print(f"读取到 {len(df)} 条药物记录。")

# ==============================
# 3. 主循环：生成分子图像
# ==============================
invalid_rows = []
success_count = 0

for idx, row in df.iterrows():
    drug_name = str(row['name'])
    smiles = row['smiles']

    if pd.isna(smiles) or str(smiles).strip() == "":
        invalid_rows.append((idx + 1, drug_name, "空SMILES"))
        continue

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        invalid_rows.append((idx + 1, drug_name, smiles))
        continue

    # ✅ 防止文件名重复：加上序号
    safe_name = "".join(c if c.isalnum() else "_" for c in drug_name)
    filename = os.path.join(output_dir, f"{idx+1:03d}_{safe_name}.png")

    try:
        img = Draw.MolToImage(mol, size=(300, 300))
        img.save(filename)
        success_count += 1
        print(f"[√] 已生成: {filename}")
    except Exception as e:
        invalid_rows.append((idx + 1, drug_name, f"绘图错误: {e}"))

# ==============================
# 4. 汇总
# ==============================
total_count = len(df)
invalid_count = len(invalid_rows)

print("\n====== 生成结果汇总 ======")
print(f"总药物数: {total_count}")
print(f"成功生成图片数: {success_count}")
print(f"无效 SMILES 或错误记录数: {invalid_count}")
print(f"预计 {total_count} 张，实际生成 {len(os.listdir(output_dir))} 张。")

if invalid_count > 0:
    invalid_df = pd.DataFrame(invalid_rows, columns=["行号", "药物名称", "问题/SMILES"])
    invalid_df.to_excel("invalid_smiles.xlsx", index=False)
    print("\n⚠️ 以下药物存在问题，详情已导出至 'invalid_smiles.xlsx'：")
    print(invalid_df)
else:
    print("\n✅ 所有 SMILES 均成功解析并生成图片。")

print("\n所有药物图片生成完毕！")
