import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
import torch
import argparse
import dashscope
from dashscope import TextEmbedding
import time
import os
import chardet

# 设置 API Key
try:
    with open("key.txt", "r") as file:
        dashscope.api_key = file.read().strip()
    print(f"✅ 成功读取 API Key: {dashscope.api_key[:6]}...{dashscope.api_key[-4:]}")
except FileNotFoundError:
    raise FileNotFoundError("❌ 找不到 key.txt，请确保文件存在")
except Exception as e:
    raise RuntimeError(f"❌ 读取 key.txt 失败: {e}")


def read_csv_with_encoding_fallback(file_path):
    """
    尝试多种编码读取 CSV，优先使用 gbk（中文常见），再 fallback 到 chardet 和 latin1
    """
    encodings_to_try = ['utf-8', 'gbk', 'gb2312', 'latin1', 'ISO-8859-1']

    # 先用 chardet 检测，插入到尝试列表开头（但不完全信任）
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(50000)  # 多读一点提高准确性
            detected = chardet.detect(raw_data)
            if detected['encoding'] and detected['confidence'] > 0.8:
                encodings_to_try.insert(0, detected['encoding'])
                print(f"🔍 chardet 高置信度检测: {detected['encoding']} ({detected['confidence']:.2f})")
    except Exception as e:
        print(f"⚠️ chardet 检测失败: {e}")

    for enc in encodings_to_try:
        try:
            print(f"📄 尝试使用编码: {enc}")
            df = pd.read_csv(file_path, encoding=enc)
            print(f"✅ 成功使用 {enc} 读取文件！")
            return df
        except (UnicodeDecodeError, UnicodeError) as e:
            print(f"❌ 编码 {enc} 失败: {str(e)[:100]}...")
            continue
        except Exception as e:
            print(f"💥 其他错误 ({enc}): {e}")
            continue

    raise ValueError(f"❌ 所有编码尝试失败，无法读取文件: {file_path}")


def get_embeddings_qwen(text, max_retry=3):
    if not text or str(text).strip() == "":
        print("⚠️  输入文本为空，跳过...")
        return None

    text = str(text).strip()[:8192]
    print(f"📝 正在为文本生成嵌入 (长度: {len(text)}): {text[:50]}...")

    for i in range(max_retry):
        try:
            response = TextEmbedding.call(model='text-embedding-v1', input=text)
            if response.status_code == 200:
                emb_len = len(response.output['embeddings'][0]['embedding'])
                print(f"✅ 嵌入生成成功！维度: {emb_len}")
                return response.output['embeddings'][0]['embedding']
            else:
                print(f"⚠️ 调用失败 [{i + 1}/3]: {response.message}")
                time.sleep(1)
        except Exception as e:
            print(f"💥 异常 [{i + 1}/3]: {e}")
            time.sleep(2)
    print(f"❌ 最终失败: {text[:50]}...")
    return None


def get_bert_token_length(description, tokenizer):
    return len(tokenizer.encode(description, add_special_tokens=True))


def get_biobert_embeddings(text, tokenizer, model):
    chunk_size = 512 - 2
    tokens = tokenizer.encode(text, add_special_tokens=False)
    token_count = len(tokens)
    print(f"📝 BioBERT处理中... 文本分词后长度: {token_count}")

    if token_count == 0:
        return None

    chunked_tokens = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    print(f"📦 分成 {len(chunked_tokens)} 个块进行处理...")

    all_embeddings = []
    for idx, chunk in enumerate(chunked_tokens):
        print(f"   ➡️  处理第 {idx + 1} 块 (长度: {len(chunk)})...")
        chunk = [tokenizer.cls_token_id] + chunk + [tokenizer.sep_token_id]
        chunk_tensor = torch.tensor([chunk])
        with torch.no_grad():
            outputs = model(chunk_tensor)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(embeddings)

    aggregated_embeddings = torch.mean(torch.stack(all_embeddings), dim=0)
    print(f"✅ BioBERT嵌入聚合完成！")
    return aggregated_embeddings


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', '--dataset', default='dataset1', type=str)
    parser.add_argument('-method', '--method', default='GPT', type=str, choices=['GPT', 'BioBERT'])
    args = parser.parse_args()

    print(f"📂 正在加载数据集: {args.dataset}")

    # 使用智能编码回退机制读取 CSV
    df_rna = read_csv_with_encoding_fallback('./dataset1/RNA_LLM.csv')
    df_drug = read_csv_with_encoding_fallback('./dataset1/drug_LLM.csv')


    print(f"📊 数据集统计:")
    print(f"   RNA 数量: {len(df_rna)}")
    print(f"   药物数量: {len(df_drug)}")
    # print(f"   药物数量: {len(df_disease)}")
    if args.method == 'GPT':
        print("\n🚀 开始使用通义千问生成嵌入...\n" + "=" * 50)

        rna_to_embeddings = {}
        for idx, (name, desc) in enumerate(zip(df_rna['name'], df_rna['description']), start=1):
            print(f"\n[{idx}/{len(df_rna)}] 🧬 RNA: {name}")
            emb = get_embeddings_qwen(desc)
            rna_to_embeddings[name] = emb
            time.sleep(0.5)

        output_path_rna = f'./dataset1/LLM_rna_emb.pkl'
        pd.to_pickle(rna_to_embeddings, output_path_rna)
        print(f"\n🎉 ✅ RNA 嵌入已全部生成并保存: {output_path_rna}")

        drug_to_embeddings = {}
        for idx, (name, desc) in enumerate(zip(df_drug['name'], df_drug['description']), start=1):
            print(f"\n[{idx}/{len(df_drug)}] 💊 药物: {name}")
            emb = get_embeddings_qwen(desc)
            drug_to_embeddings[name] = emb
            time.sleep(0.5)

        output_path_drug = f'./dataset1/LLM_drug_emb.pkl'
        pd.to_pickle(drug_to_embeddings, output_path_drug)
        print(f"\n🎉 ✅ 药物嵌入已全部生成并保存: {output_path_drug}")



    elif args.method == 'BioBERT':
        print("\n🧫 开始使用 BioBERT 本地生成嵌入...\n" + "=" * 50)

        try:
            tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
            model = BertModel.from_pretrained('dmis-lab/biobert-v1.1')
            print("✅ BioBERT 模型加载成功！")
        except Exception as e:
            print(f"❌ 加载 BioBERT 失败: {e}")
            exit(1)

        # 修复：将 tokenizer 传入 get_bert_token_length
        df_rna['token_length'] = df_rna['Description'].apply(lambda x: get_bert_token_length(x, tokenizer))
        df_rna['embeddings'] = df_rna['Description'].apply(lambda x: get_biobert_embeddings(x, tokenizer, model))

        os.makedirs(f'feat/{args.dataset}', exist_ok=True)
        output_path_rna = f'feat/{args.dataset}/BERT_rna_emb.pkl'
        pd.to_pickle(df_rna, output_path_rna)
        print(f"✅ RNA 嵌入已保存: {output_path_rna}")

        df_drug['token_length'] = df_drug['Description'].apply(lambda x: get_bert_token_length(x, tokenizer))
        df_drug['embeddings'] = df_drug['Description'].apply(lambda x: get_biobert_embeddings(x, tokenizer, model))
        output_path_drug = f'feat/{args.dataset}/BERT_drug_emb.pkl'
        pd.to_pickle(df_drug, output_path_drug)
        print(f"✅ 药物嵌入已保存: {output_path_drug}")

        print("\n🎉 ✅ 所有 BioBERT 嵌入生成完毕！")