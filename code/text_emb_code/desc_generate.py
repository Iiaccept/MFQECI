import csv
import os
import dashscope
from http import HTTPStatus
import argparse
import time


# 设置 DashScope API 密钥
try:
    with open("key.txt", "r") as file:
        dashscope.api_key = file.read().strip()
    print(f"✅ 成功读取 API Key: {dashscope.api_key[:6]}...{dashscope.api_key[-4:]}")
except FileNotFoundError:
    raise FileNotFoundError("❌ 找不到 key.txt，请确保文件存在")
except Exception as e:
    raise RuntimeError(f"❌ 读取 key.txt 失败: {e}")

# 定义模型（推荐使用固定版本号更稳定）
MODEL_ID = "qwen3-max"
# 可选：MODEL_ID = "qwen-max-2024-09-19"

def create_user_message_rna_unified(rna_name, rna_type):
    return (
        f"Generate a single, cohesive, narrative paragraph for the {rna_type} '{rna_name}'. "
        "The response should include the following 10 key pieces of information:\n"
        "1) Genomic location and structural features;\n"
        "2) Primary gene regulation function;\n"
        "3) Key signaling pathways;\n"
        "4) Experimentally validated target genes, proteins, or other RNAs;\n"
        "5) Association with human diseases;\n"
        "6) Cellular phenotypic outcomes upon perturbation;\n"
        "7) Key structural motifs and subcellular localization;\n"
        "8) Expression in normal vs. diseased tissues;\n"
        "9) Potential as a diagnostic, prognostic, or predictive biomarker in clinical settings;\n"
        "10) Feasibility as a therapeutic target or agent.\n"
        "Ensure the summary is precise, evidence-based, suitable for a professional pharmacogenomics or molecular medicine audience, "
        "and integrates all points into a coherent narrative without listing them numerically."
    )


def create_user_message_drug(drug_name):
    return (
        f"Generate a single, comprehensive paragraph for the drug '{drug_name}'. "
        "The response should include the following 10 key pieces of information:\n"
        "1) Chemical structure class and representative scaffold;\n"
        "2) Primary molecular targets;\n"
        "3) Key signaling pathways modulated by the drug;\n"
        "4) Known mechanisms of action at the molecular level;\n"
        "5) Major therapeutic indications;\n"
        "6) Details of its toxicity, with examples;\n"
        "7) List of any known target proteins;\n"
        "8) Indication of this drug, with specific examples of diseases or symptoms;\n"
        "9) Side effects of this drug, with examples;\n"
        "10) Clinical usage of this drug, with examples.\n"
        "Ensure the summary is precise, evidence-based, suitable for a professional pharmacogenomics or computational biology audience, "
        "and integrates all points into a coherent narrative without listing them numerically."
    )

def create_user_message_dis_OMIM(disease_name, omim_id):
    return (
        f"Generate a single, cohesive, narrative paragraph for the disease '{disease_name}' associated with OMIM ID '{omim_id}'. "
        "The response should include 10 key pieces of information as follows:\n"
        "1) associated genes, proteins, or mutations, with at least 3 examples.\n"
        "2) associated signal pathway, including key molecular or cellular components.\n"
        "3) associated drugs commonly used for treatment, with at least 3 examples and their mechanisms of action.\n"
        "4) any linked comorbidities and complications.\n"
        "5) nature of the disease.\n"
        "6) typical clinical symptoms and signs.\n"
        "7) types of the disease.\n"
        "8) inheritance patterns and any known genetic component, with examples.\n"
        "9) diagnostic criteria and testing methods.\n"
        "Ensure the final summary is precise, evidence-based, suitable for a professional medical audience, and condenses all the points above into a coherent narrative."
    )


def create_user_message_dis_MeSH(disease_name, mesh_id):
    return (
        f"Generate a single, cohesive, narrative paragraph for the disease '{disease_name}' in MeSH (Medical Subject Headings) vocabulary have association with MeSH ID '{mesh_id}'. "
        "The response should include 10 key pieces of information as follows:\n"
        "1) associated genes, proteins, or mutations, with at least 3 examples.\n"
        "2) associated signal pathway, including key molecular or cellular components.\n"
        "3) associated drugs commonly used for treatment, with at least 3 examples and their mechanisms of action.\n"
        "4) any linked comorbidities and complications.\n"
        "5) nature of the disease.\n"
        "6) typical clinical symptoms and signs.\n"
        "7) types of the disease.\n"
        "8) inheritance patterns and any known genetic component, with examples.\n"
        "9) diagnostic criteria and testing methods.\n"
        "Ensure the final summary is precise, evidence-based, suitable for a professional medical audience, and condenses all the points above into a coherent narrative."
    )


def safe_get_content(response):
    """
    安全提取 response 中的文本内容，兼容多种格式
    """
    if response is None:
        return "Error: API returned None"

    if response.status_code != HTTPStatus.OK:
        error_msg = getattr(response, 'message', 'Unknown error')
        return f"Error: {error_msg}"

    try:
        # 方法1：优先尝试从 message 结构获取（需 result_format='message'）
        if hasattr(response, 'output') and hasattr(response.output, 'choices'):
            if response.output.choices and len(response.output.choices) > 0:
                content = response.output.choices[0].message.content
                return content.strip() if content else "Error: Empty content"
    except Exception as e:
        print(f"⚠️ 解析 choices 失败: {e}")

    try:
        # 方法2：fallback 到 text 字段
        if hasattr(response, 'output') and hasattr(response.output, 'text'):
            text = response.output.text
            return text.strip() if text else "Error: Empty text"
    except Exception as e:
        print(f"⚠️ 解析 text 失败: {e}")

    return "Error: Failed to extract response content"


def process_drugs(input_path, output_path):
    """处理 drug.csv 文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, newline='', encoding='utf-8') as infile, \
         open(output_path, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=['Name',  'Description'])
        writer.writeheader()

        for row in reader:
            drug_name = row['name']

            messages = [
                {"role": "system", "content": "You are an expert in medical research, genetics, chemistry, and pharmacology."},
                {"role": "user", "content": create_user_message_drug(drug_name)}
            ]

            try:
                # ✅ 关键：添加 result_format='message' 确保返回标准结构
                response = dashscope.Generation.call(
                    model=MODEL_ID,
                    messages=messages,
                    result_format='message',  # 👈 保证输出是 message 格式
                    max_tokens=450
                )
                description = safe_get_content(response)

            except Exception as e:
                print(f"[Drug] 调用异常 {drug_name}: {type(e).__name__}: {e}")
                description = "Error: Exception occurred"
            # 写入结果
            writer.writerow({'Name': drug_name,  'Description': description})
            # ✅ 可选：加延时防限流
            time.sleep(1)

    print(f"✅ 药物处理完成。结果已保存至: {output_path}")


def process_rnas(input_path, output_path):
    """处理 disease.csv 文件"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(input_path, newline='', encoding='utf-8') as infile, \
         open(output_path, 'w', newline='', encoding='utf-8') as outfile:

        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=['Name', 'Type', 'Description'])
        writer.writeheader()

        for row in reader:
            rna_name = row['name']
            rna_type = row['type']
            messages = [
                {"role": "system", "content": "You are an expert in medical research, genetics, and pharmacology."},
                {"role": "user", "content":  create_user_message_rna_unified(rna_name, rna_type)}
            ]

            try:
                # ✅ 同样添加 result_format='message'
                response = dashscope.Generation.call(
                    model=MODEL_ID,
                    messages=messages,
                    result_format='message',
                    max_tokens=450
                )
                description = safe_get_content(response)

            except Exception as e:
                print(f"[Disease] 调用异常 {rna_name}: {type(e).__name__}: {e}")
                description = "Error: Exception occurred"

            writer.writerow({'Name': rna_name, 'Type': rna_type, 'Description': description})
            # ✅ 延时
            time.sleep(1)
    print(f"✅ 疾病处理完成。结果已保存至: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="使用通义千问生成药物与疾病描述")
    parser.add_argument('-dataset', '--dataset', default='dataset1', type=str, help="dataset1 path ")
    args = parser.parse_args()
    # 构建路径
    drug_input = os.path.join(args.dataset, 'drug.csv')
    drug_output = os.path.join(args.dataset, 'drug_LLM.csv')
    disease_input = os.path.join(args.dataset, 'RNA.csv')
    disease_output = os.path.join(args.dataset, 'RNA_LLM.csv')
    # 检查输入文件是否存在
    if not os.path.exists(drug_input):
        print(f"❌ 错误: {drug_input} 不存在，请检查路径")
    else:
        print(f"📄 开始处理药物数据: {drug_input}")
        process_drugs(drug_input, drug_output)

    if not os.path.exists(disease_input):
        print(f"❌ 错误: {disease_input} 不存在，请检查路径")
    else:
        print(f"📄 开始处理疾病数据: {disease_input}")
        process_rnas(disease_input, disease_output)



    print("🎉 所有任务已完成！")
