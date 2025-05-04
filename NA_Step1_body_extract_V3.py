# 依赖 / Dependencies / 依存関係：python-docx, pandas, openpyxl, unicodedata, tqdm, spacy, fuzzywuzzy, python-Levenshtein
# 安装 / Install / インストール：pip install python-docx pandas openpyxl tqdm spacy fuzzywuzzy python-Levenshtein
# 模型下载 / Model Download / モデルダウンロード：python -m spacy download en_core_web_sm
# 原作者：杨天乐@关西大学 / Author: Shiame Yeung@Kansai University / 作成者：楊　天楽@関西大学

import os
import re
import unicodedata
from docx import Document
import pandas as pd
from tqdm import tqdm

# ------------------ 用户可配置 ------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = SCRIPT_DIR  # 根目录就是脚本所在位置
OUTPUT_CSV = os.path.join(SCRIPT_DIR, 'keyword_hit.csv')  # 输出为 CSV，与Step2/Step3保持一致
# -----------------------------------------------

# 定义关键词根列表 / Keyword roots list
KEYWORD_ROOTS = [
    'partner','alliance','collaborat','cooper','cooperat','join','merger','acquisiti',
    'outsourc','invest','licens','integrat','coordinat','synergiz','associat',
    'confedera','federa','union','unit','amalgamat','conglomerat','combin',
    'buyout','companion','concur','concert','comply','complement','assist',
    'takeover','accession','procure','suppl','conjoint','support','adjust',
    'adjunct','patronag','subsid','affiliat','endors'
]

# 文本清理函数：去除非法字符 / Clean text
def clean_text(text):
    return ''.join(c for c in text if unicodedata.category(c)[0] != 'C' or c in ('\n','\t'))

# 提取单个 Word 文件中的句子 / Extract sentences
def extract_sentences_from_docx(filepath):
    try:
        doc = Document(filepath)
    except Exception as e:
        print(f"❗ 无法读取文件 {filepath}: {e}")
        return []

    collecting = False
    current_article = ''
    all_articles = []
    for para in doc.paragraphs:
        txt = para.text.strip()
        if not txt: continue
        if txt == 'Body':
            collecting = True
            current_article = ''
            continue
        if txt == 'Notes' and collecting:
            collecting = False
            all_articles.append(current_article.strip())
            continue
        if collecting:
            current_article += ' ' + txt

    sentences = []
    for article in all_articles:
        for sent in re.split(r'\.\s*', article):
            sent = sent.strip()
            if len(sent) < 5: continue
            sent = clean_text(sent)
            if sent:
                sentences.append(sent)
    return sentences

# 遍历并处理 docx 文件，兼容 0/1/2 层文件夹
records = []
docx_files = []
for root, dirs, files in os.walk(ROOT_DIR):
    for fname in files:
        if not fname.endswith('.docx') or fname.startswith('~$'): continue
        path = os.path.join(root, fname)
        rel = os.path.relpath(path, ROOT_DIR).split(os.sep)
        if len(rel) >= 3:
            tier1, tier2, filename = rel[0], rel[1], rel[2]
        elif len(rel) == 2:
            tier1, tier2, filename = rel[0], '', rel[1]
        else:
            tier1, tier2, filename = '', '', rel[0]
        docx_files.append((path, tier1, tier2, filename))

for path, tier1, tier2, filename in tqdm(docx_files, desc="📄 处理 Word 文件", ncols=100):
    for sent in extract_sentences_from_docx(path):
        hits = [k for k in KEYWORD_ROOTS if k in sent.lower()]
        records.append({
            'Tier_1': tier1,
            'Tier_2': tier2,
            'Filename': filename,
            'Sentence': sent,
            'Hit_Count': len(hits),
            'Matched_Keywords': '; '.join(hits)
        })

# 构建 DataFrame 并排序
df = pd.DataFrame(records, columns=['Tier_1','Tier_2','Filename','Sentence','Hit_Count','Matched_Keywords'])
if not df.empty:
    df['Hit_Flag'] = df['Hit_Count'].apply(lambda x: 1 if x>0 else 0)
    df = df.sort_values(['Hit_Flag','Tier_1','Tier_2','Filename'], ascending=[False,True,True,True]).reset_index(drop=True)
    df = df.drop(columns=['Hit_Flag'])

# 输出为 CSV
if not df.empty:
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ 全部处理完成，输出文件：{OUTPUT_CSV}")
else:
    print("⚠️ 没有找到任何 docx 文件或内容为空。")
