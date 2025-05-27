# 依赖 / Dependencies / 依存関係：python-docx, pandas, openpyxl, unicodedata, tqdm, spacy, fuzzywuzzy, python-Levenshtein
# 安装 / Install / インストール：pip install python-docx pandas openpyxl tqdm spacy fuzzywuzzy python-Levenshtein
# 模型下载 / Model Download / モデルダウンロード：python -m spacy download en_core_web_sm
# 原作者：杨天乐@关西大学 / Author: Shiame Yeung@Kansai University / 作成者：楊　天楽@関西大学


import os
import re
import time
import pandas as pd
import spacy
from tqdm import tqdm
from fuzzywuzzy import process

# —— 1. SpaCy NER 只保留必要组件，加快速度 ——
nlp = spacy.load(
    "en_core_web_sm",
    disable=["tagger", "parser", "attribute_ruler", "lemmatizer"]
)

# —— 2. 默认公司列表，用于初始化 NA_company_list.csv ——
DEFAULT_COMPANIES = [
    "IBM","Microsoft","Apple","Google","Amazon","Meta","Facebook",
    "Oracle","SAP","Cisco","Intel","AMD","Nvidia","Dell","HP",
    "Lenovo","Huawei","Samsung","Sony","Panasonic","Fujitsu",
    "Tencent","Alibaba","Baidu","Xiaomi","Didi","Bytedance",
    "Uber","Airbnb","Salesforce","Adobe","Twitter","LinkedIn",
    "Zoom","Slack","Spotify","Netflix","eBay","PayPal",
    "Square","Stripe","OpenAI","TSMC","Qualcomm","LG","SK Hynix"
]

# —— 工具：生成唯一文件名 ——
def get_unique_path(path):
    base, ext = os.path.splitext(path)
    counter = 1
    candidate = path
    while os.path.exists(candidate):
        candidate = f"{base}_{counter}{ext}"
        counter += 1
    return candidate

# —— 3. 合法 Token 判断 ——
def is_valid_token(token):
    token = token.strip()
    if not token or all(c in "-–—・.、。！？／ー" for c in token):
        return False
    # 含数字但不含英文 → 跳过
    if re.search(r"\d", token) and not re.search(r"[A-Za-z]", token):
        return False
    # 连续两个空格 → 跳过
    if "  " in token:
        return False
    return True

# —— 4. 原始企业名提取 ——
def extract_companies(text, company_db, ner_model, fuzzy_threshold=95):
    comps = set()
    # 去掉日期及其后内容
    text_clean = re.sub(r"\s*\d{1,2}/\d{1,2}/\d{2,4}.*$", "", text).strip()

    # spaCy NER 提取
    doc = ner_model(text_clean)
    for ent in doc.ents:
        ent_text = ent.text.strip()
        # 跳过噪音
        if "  " in ent_text or re.search(r"[\d/%+]|[^\x00-\x7F]", ent_text):
            continue
        valid_ent = True
        for w in ent_text.split():
            if not w[0].isalpha() or w in {"The","And","For","With","From","That","This"}:
                valid_ent = False
                break
            if not is_valid_token(w):
                valid_ent = False
                break
        if valid_ent:
            comps.add(ent_text)

    # IBMers 类规则
    for m in re.findall(r"\b([A-Z]{2,})ers\b", text_clean):
        comps.add(m)

    # 严格模糊匹配补充
    STOPWORDS = {"The","And","For","With","From","That","This","Have","Will",
                 "Are","You","Not","But","All","Any","One","Our","Their"}
    tokens = re.findall(r"\b\S+\b", text_clean)
    for pos, token in enumerate(tokens):
        if pos == 0 or token in STOPWORDS:
            continue
        if any(ch in token for ch in "/%+") or "  " in token:
            continue
        if len(token) < 5 or not token[0].isupper() or token.isupper():
            continue
        if re.search(r"\d|[^\x00-\x7F]", token) or not is_valid_token(token):
            continue
        best, score = process.extractOne(token, company_db)
        if score >= fuzzy_threshold:
            comps.add(best)

    return list(comps)

# —— 5. 初始化或加载映射表 ——
def prepare_mapping_tables(script_dir):
    comp_list_path = os.path.join(script_dir, "NA_company_list.csv")
    if not os.path.exists(comp_list_path):
        pd.DataFrame({
            "Standard_Name": DEFAULT_COMPANIES,
            "Aliases": ["" for _ in DEFAULT_COMPANIES]
        }).to_csv(comp_list_path, index=False)

    comp_df = pd.read_csv(comp_list_path)
    # 确保 Aliases 列为字符串
    comp_df['Aliases'] = comp_df['Aliases'].fillna('').astype(str)

    # 计算 banned 列表
    banned = set(
        comp_df.loc[
            comp_df['Aliases'].str.strip().str.lower() == 'banned',
            'Standard_Name'
        ]
    )

    # 构建标准名与别名映射
    std_set = set()
    alias_map = {}
    for _, r in comp_df.iterrows():
        std = r['Standard_Name'].strip()
        if std in banned:
            continue
        std_set.add(std)
        for a in [x.strip() for x in r['Aliases'].split(',') if x.strip() and x.strip().lower() != 'banned']:
            alias_map[a] = std

    # 始终新建 mapping
    map_path = os.path.join(script_dir, "NA_mapping.csv")
    if os.path.exists(map_path):
        backup = get_unique_path(map_path)
        os.rename(map_path, backup)
    # 新增 Sentence 列，并放在最前
    mapping_df = pd.DataFrame(columns=["Sentence", "NonStandard", "Standard"])
    mapping_df.to_csv(map_path, index=False)

    return std_set, alias_map, mapping_df, comp_list_path, map_path, banned

# —— 6. 主流程 ——
def process_all_files():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    std_set, alias_map, mapping_df, comp_list_path, map_path, banned = prepare_mapping_tables(script_dir)

    # unmatched 从 set 改为 dict，用于记录首次出现的句子
    unmatched = {}

    # 筛选原始文件
    all_files = [
        f for f in os.listdir(script_dir)
        if (f.lower().endswith('.xlsx') or f.lower().endswith('.csv'))
        and f.lower() not in ('na_company_list.csv','na_mapping.csv')
        and not f.lower().endswith(('_recognized.csv','_log.csv'))
    ]

    for fname in tqdm(all_files, desc="📁 文件总进度", unit="文件"):
        t0 = time.time()
        in_path = os.path.join(script_dir, fname)
        base = os.path.splitext(fname)[0] + "_recognized"
        out_csv = get_unique_path(os.path.join(script_dir, base + ".csv"))
        log_csv = get_unique_path(os.path.join(script_dir, base + "_log.csv"))

        print(f"> 处理文件：{fname}")
        if fname.lower().endswith('.xlsx'):
            df = pd.read_excel(in_path, engine='openpyxl')
        else:
            df = pd.read_csv(in_path)

        if not all(c in df.columns for c in ['Hit_Count','Sentence','Matched_Keywords']):
            print("  ⚠️ 缺少关键列，跳过。")
            continue

        hit_df = df[df['Hit_Count']>0].reset_index(drop=True)
        print(f"  - 共 {len(hit_df)} 条目标句子，开始识别…")
        t1 = time.time()

        cache, extracted, logs = {}, {}, []
        for idx, row in tqdm(hit_df.iterrows(), total=len(hit_df), desc="  🔍 处理中", unit="行"):
            sent = str(row['Sentence'])
            raw = cache.get(sent) or extract_companies(sent, list(std_set)+list(alias_map.keys()), nlp)
            cache[sent] = raw

            std_list = []
            for nm in raw:
                if nm in banned:
                    continue
                if nm in std_set:
                    std_list.append(nm)
                elif nm in alias_map:
                    std_list.append(alias_map[nm])
                else:
                    # 记录首次出现该非标准名时对应的原文句子
                    if nm not in unmatched:
                        unmatched[nm] = sent
                    std_list.append(nm)

            if std_list:
                extracted[idx] = std_list
                logs.append({
                    'Index': idx,
                    'Sentence': sent,
                    'Extracted_Companies': ', '.join(std_list)
                })

        t2 = time.time()

        if not extracted:
            print("  ℹ️ 无识别结果，跳过写入。")
            continue

        # 构造 Company_n 列并保存
        max_n = max(len(v) for v in extracted.values())
        comp_cols = {f'Company_{i+1}':['']*len(df) for i in range(max_n)}
        for idx, names in extracted.items():
            for i, nm in enumerate(names):
                comp_cols[f'Company_{i+1}'][idx] = nm

        out_df = pd.concat([df, pd.DataFrame(comp_cols)], axis=1)
        print("  💾 保存结果和日志…")
        out_df.to_csv(out_csv, index=False)
        pd.DataFrame(logs).to_csv(log_csv, index=False)
        print(f"  ✅ 导出：{out_csv}")
        print(f"  📄 日志：{log_csv}")

        t3 = time.time()
        print(f"  ⏱ 总耗时: {t3-t0:.1f}s，抽取耗时: {t2-t1:.1f}s")

    # 更新 mapping
    existing = set(mapping_df['NonStandard'])
    new = [nm for nm in sorted(unmatched) if nm not in existing]
    if new:
        add_df = pd.DataFrame({
            'Sentence': [unmatched[nm] for nm in new],
            'NonStandard': new,
            'Standard': [''] * len(new)
        })[
            ["Sentence", "NonStandard", "Standard"]
        ]
        mapping_df = pd.concat([mapping_df, add_df], ignore_index=True)
        mapping_df.to_csv(map_path, index=False)
        print(f"🔄 更新映射文件：{map_path}（新增 {len(new)} 条待人工标注）")
    else:
        print("ℹ️ 无新增待标注企业名。")

if __name__ == "__main__":
    process_all_files()
