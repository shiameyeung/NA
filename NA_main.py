# 原作者：杨天乐@关西大学 / Author: Shiame Yeung@Kansai University / 作成者：楊　天楽@関西大学
#!/usr/bin/env python3
# coding: utf-8
"""
na_pipeline.py  ——  单文件版（Step‑1 对齐 + 扩展公司识别）
2025‑07‑08  rev‑C
"""

import os, re, sys, unicodedata, string
from pathlib import Path
from typing import List, Dict, Set

from datetime import datetime
import random

import pandas as pd
from tqdm import tqdm
from sqlalchemy import create_engine, text
from rapidfuzz import fuzz, process

try:
    from docx import Document
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
except Exception:
    print("❌ 缺少依赖：pip install python-docx spacy && python -m spacy download en_core_web_sm"); sys.exit(1)

# ---------------- 常量 ----------------
STOPWORDS = {"the","and","for","with","from","that","this","have","will","are","you","not","but","all","any","one","our","their"}
KEYWORD_ROOTS = [
    'partner','alliance','collaborat','cooper','cooperat','join','merger','acquisiti',
    'outsourc','invest','licens','integrat','coordinat','synergiz','associat',
    'confedera','federa','union','unit','amalgamat','conglomerat','combin',
    'buyout','companion','concur','concert','comply','complement','assist',
    'takeover','accession','procure','suppl','conjoint','support','adjust',
    'adjunct','patronag','subsid','affiliat','endors'
]
# ---------------- Bad-Rate 规则 ----------------
ORG_SUFFIX  = re.compile(
    r'\b(Inc\.?|Corp\.?|Corporation|Ltd\.?|LLC|PLC|AG|NV|SA|GmbH|S\.p\.A|Co\.?|Company|'
    r'Group|Holdings?|Partners?|Capital|Ventures?|Bank|Trust|Software|'
    r'Technolog(?:y|ies)|Pharma(?:ceuticals)?|Systems?|Services?|'
    r'Industr(?:y|ies)|Foundation|Laborator(?:y|ies)|'
    r'University|College|Institute|School|Hospital|Center|Centre)\b',
    re.I)

TIME_QTY    = re.compile(
    r'\b(year|month|week|day|decade|centur(?:y|ies)|quarter|q[1-4]|'
    r'ago|last|next|few|couple|several|dozen|half|around|approximately)s?\b',
    re.I)
ART_LOWER   = re.compile(r'^\s*(a|an|about|approximately|the|this|that|those)\s+[a-z]')
GENERIC_END = re.compile(
    r'\b(plan|plans?|programs?|systems?|platforms?|services?|solutions?|operations?|'
    r'agreements?|strategies?|reports?|statements?)$', re.I)

def _lower_ratio(text: str) -> float:
    w = text.split()
    return sum(t[0].islower() for t in w) / len(w) if w else 0

def calc_bad_rate(text: str) -> int:
    """0–100：越高越可能是 bad"""
    if ORG_SUFFIX.search(text):
        return 0
    score = 0
    if TIME_QTY.search(text):              score += 40
    if ART_LOWER.match(text):              score += 25
    if ' of the ' in text.lower():         score += 20
    if GENERIC_END.search(text):           score += 15
    if len(text.split()) <= 2:             score += 20
    if _lower_ratio(text) > 0.30:          score += 20
    return min(score, 100)
# ---------------- 全局变量 ----------------
BASE_DIR = Path(__file__).resolve().parent
MAX_COMP_COLS = 50
SENTENCE_RECORDS: List[Dict] = []

# ---------------- 共用 ----------------



# ------- 新版本：首次输入后写 .db_key，后续自动读取 -------
def ask_mysql_url() -> str:
    key_file = Path(__file__).with_name(".db_key")   # 脚本同目录 .db_key
    if key_file.exists():
        key = key_file.read_text().strip()
    else:
        key = input("请输入秘钥/キーを入力してください：user:pass@host\n>>>>>> ").strip()
        key_file.write_text(key)                     # 缓存下次用
    return f"mysql+pymysql://{key}.mysql.rds.aliyuncs.com:3306/na_data?charset=utf8mb4"

def choose() -> str:
    # ── 1. 选项框 ───────────────────────────────────────────
    print(r"""
┌──────────────────────────────────────────────┐
│  ①  初次运行（Step-1 ➜ Step-2） / 初回実行      │
│  ②  已有 mapping（Step-3） / mapping 適用のみ  │
│  作者：楊　天楽＠関西大学　伊佐田研究室             │
└──────────────────────────────────────────────┘
""")
    c = input("请输入 1 或 2 / 1 か 2 を入力してください: ").strip()

    # ── 2. 校验 ───────────────────────────────────────────
    if c not in {"1", "2"}:
        print(r"""
┌──────────────────────────┐
│  ❌ 无效选择 / 無効な選択です  │
└──────────────────────────┘
""")
        sys.exit(1)

    return c

def dedup_company_cols(df: pd.DataFrame) -> pd.DataFrame:
    comp_cols = [c for c in df.columns if c.startswith("company_")]
    for ridx in df.index:
        seen: Set[str] = set()
        for col in comp_cols:
            val = str(df.at[ridx, col]).strip()
            if val in seen:
                df.at[ridx, col] = ""
            else:
                seen.add(val)
    return df

# ---------------- Step‑1 ----------------

def _normalize(text: str) -> str:
    t = re.sub(r"\s+", " ", text.lower().strip())
    t = re.sub(r"^[\-:\"']+|[\-:\"']+$", "", t)
    t = re.sub(r"[,.;/()]+", "", t)
    return t.strip()

def clean_text(t: str) -> str:
    return ''.join(c for c in t if unicodedata.category(c)[0] != 'C' or c in ('\n', '\t'))

def extract_sentences(path: Path) -> List[str]:
    doc = Document(path)
    collecting, current, articles = False, "", []
    for p in doc.paragraphs:
        txt = p.text.strip()
        if not txt: continue
        tag = txt.lower()
        if tag == "body": collecting, current = True, ""; continue
        if tag in ("notes", "classification") and collecting:
            collecting = False; articles.append(current.strip()); continue
        if collecting: current += " " + txt
    sents = []
    for art in articles:
        for s in re.split(r"\.\s*", art):
            s = s.strip();
            if len(s) >= 5: sents.append(s)
    return sents

def extract_index_titles(paragraphs):
    paras_text = [p.text.strip() for p in paragraphs]
    m = re.search(r'Documents?\s*\(\s*(\d+)\s*\)', '\n'.join(paras_text), re.I)
    if not m: return []
    total = int(m.group(1)); pat = re.compile(r'^(\d+)\.\s+(.*)$'); seen, titles = set(), []
    for line in paras_text:
        m2 = pat.match(line)
        if m2:
            raw = m2.group(2).strip(); norm = _normalize(raw)
            if norm in seen: continue
            seen.add(norm); titles.append((int(m2.group(1)), raw, norm))
            if len(titles) >= total: break
    return sorted(titles, key=lambda x: x[0])

def extract_sentences_by_titles(filepath: str) -> List[Dict]:
    doc = Document(filepath); paras = doc.paragraphs
    index_titles = extract_index_titles(paras); recs = []
    if index_titles:
        paras_norm = [_normalize(p.text) for p in paras]
        for _, title_raw, title_norm in index_titles:
            try: match_idx = next(i for i, n in enumerate(paras_norm) if n == title_norm)
            except StopIteration: continue
            pub_idx = match_idx + 1; publisher = paras[pub_idx].text.strip() if pub_idx < len(paras) else ""
            body_start = next((i+1 for i in range(match_idx+1,len(paras)) if paras[i].text.strip().lower()=="body"), None)
            if body_start is None: body_start = pub_idx + 1
            body_end = next((i for i in range(body_start, len(paras)) if paras[i].text.strip().lower() in ("notes","classification")), len(paras))
            article = " ".join(clean_text(paras[i].text) for i in range(body_start, body_end))
            for sent in [s.strip() for s in re.split(r"\.\s*", article) if len(s.strip())>=5]:
                hits = [k for k in KEYWORD_ROOTS if k in sent.lower()]
                recs.append({"Title":title_raw,"Publisher":publisher,"Country":"","Sentence":sent,"Hit_Count":len(hits),"Matched_Keywords":"; ".join(hits)})
        if recs: return recs
    # 无索引
    for sent in extract_sentences(Path(filepath)):
        hits=[k for k in KEYWORD_ROOTS if k in sent.lower()]
        recs.append({"Title":"","Publisher":"","Country":"","Sentence":sent,"Hit_Count":len(hits),"Matched_Keywords":"; ".join(hits)})
    return recs

def step1():
    print("\n▶ Step-1：提取 Word 句子 / Word 文から文抽出中…")
    all_recs: List[Dict] = []

    # 1) 收集所有 .docx 路径
    docx_files = []
    for root, _, files in os.walk(BASE_DIR):
        for fname in files:
            if not fname.endswith(".docx") or fname.startswith("~$"):
                continue
            full = Path(root) / fname
            rel = full.relative_to(BASE_DIR).parts
            tier1 = rel[0] if len(rel) >= 1 else ""
            tier2 = rel[1] if len(rel) >= 2 else ""
            docx_files.append((str(full), tier1, tier2, fname))

    # 2) 逐文件提取句子
    for fp, t1, t2, fname in tqdm(docx_files, desc="📄 处理 Word 文件"):
        for r in extract_sentences_by_titles(fp):
            if not r["Title"]:
                r["Title"] = Path(fname).stem
            r.update({"Tier_1": t1, "Tier_2": t2, "Filename": fname})
            all_recs.append(r)

    global SENTENCE_RECORDS
    SENTENCE_RECORDS = all_recs
    print(f"✔ Step-1 完成 / 完了：共 {len(all_recs)} 条记录 / 件（已缓存 / キャッシュ済み）")

# ----------------—— Step‑2 ——----------------

def is_valid_token(token: str) -> bool:
    token = token.strip()
    if "@" in token or token.startswith("http"):    # ① 含邮箱 / URL 特征
        return False
    if not token or all(c in "-–—・.、。！？／ー" for c in token):
        return False
    if re.search(r"\d", token) and not re.search(r"[A-Za-z]", token):
        return False
    if "  " in token:
        return False
    return True


# —— 4. 原始企业名提取 ——
# —— 4. 原始企业名提取 ——
def extract_companies(text: str,
                      company_db: List[str],
                      ner_model,
                      fuzzy_threshold: int = 95) -> List[str]:
    """
    · **仅负责“把句子里可能是公司名的片段全部抓出来”**，
      不做任何 ban/映射/去重处理——这些留给后续数据库比对阶段完成。
    · 识别逻辑完全沿用单体版（spaCy NER + “IBMers”正则 + 严格模糊匹配）。
    """
    comps: Set[str] = set()

    # 1) 去掉日期（排除 ‘xx/xx/xxxx 之后整段’ 的噪音）
    text_clean = re.sub(r"\s*\d{1,2}/\d{1,2}/\d{2,4}.*$", "", text).strip()
    # --- 新增清洗 ---
    # 1) 去掉 ® ™ ©
    text_clean = re.sub(r"[®™©]", "", text_clean)
    # 2) 去掉简写商标括号，如 “Weight Doctors(R)”
    text_clean = re.sub(r"\(\s*[A-Z]{1,3}\s*\)", "", text_clean)
    # 3) 整句里带邮箱的直接剪掉邮箱
    text_clean = re.sub(r"\b\S+@\S+\b", "", text_clean)

    # 2) spaCy NER
    doc = ner_model(text_clean)
    for ent in doc.ents:
        ent_text = ent.text.strip()

        # —— 基础噪音过滤（和单体版一致）
        if "  " in ent_text or re.search(r"[\d/%+]|[^\x00-\x7F]", ent_text):
            continue
        valid_ent = True
        for w in ent_text.split():
            if (not w[0].isalpha()
                or w in {"The","And","For","With","From","That","This"}
                or not is_valid_token(w)):
                valid_ent = False
                break
        if valid_ent:
            comps.add(ent_text)

    # 3) “IBMers” 一类写法
    for m in re.findall(r"\b([A-Z]{2,})ers\b", text_clean):
        comps.add(m)

    # 4) 仅用于“确认是已知公司”，但依旧返回原词
    STOPWORDS = {"The","And","For","With","From","That","This","Have","Will",
                "Are","You","Not","But","All","Any","One","Our","Their"}

    tokens = re.findall(r"\b\S+\b", text_clean)
    for pos, token in enumerate(tokens):
        # —— 噪音与格式过滤（同原先逻辑） ——
        if (pos == 0 or token in STOPWORDS
            or any(ch in token for ch in "/%+") or "  " in token
            or len(token) < 5 or not token[0].isupper() or token.isupper()
            or re.search(r"\d|[^\x00-\x7F]", token)
            or not is_valid_token(token)):
            continue

        # 若数据库里存在“完全同名（大小写不同视同）”的条目，就保留；否则忽略
        if any(token.lower() == db.lower() for db in company_db):
            comps.add(token)

    return list(comps)


def step2(mysql_url: str):
    print("\\n▶ Step-2：公司识别 + ban 过滤 / 企業名認識＋BAN フィルタリング…")
    # 单独导出 canonical 表（engine_tmp）
    engine_tmp = create_engine(mysql_url)            # ← 新建
    df_canon = pd.read_sql("SELECT id, canonical_name FROM company_canonical", engine_tmp)
    df_canon.to_csv(BASE_DIR / "canonical_list.csv", index=False, encoding="utf-8-sig")
    print(f"  · canonical_list.csv 已写 / 保存 {len(df_canon)} 行 / 行")
    # ---- 连接数据库 ----
    engine = create_engine(mysql_url)
    with engine.begin() as conn:
        ban_set = {r[0] for r in conn.execute(text("SELECT alias FROM ban_list"))}
        rows = conn.execute(text("""
            SELECT a.alias, c.canonical_name FROM company_alias a
            JOIN company_canonical c ON a.canonical_id = c.id
        """))
        alias_map = {alias: canon for alias, canon in rows}
        canon_set = {r[0] for r in conn.execute(text("SELECT canonical_name FROM company_canonical"))}
        # ↓↓↓ 新增：名字→ID 的字典，用于 Advice 对应的 ID
        rows2 = conn.execute(text(
            "SELECT id, canonical_name FROM company_canonical"
        ))
        canon_name2id = {name: cid for cid, name in rows2}      # ← 新增
    print(f"  · ban_list {len(ban_set)} 条 / 件，alias_map {len(alias_map)} 条 / 件，canon_set {len(canon_set)} 条 / 件")

    df = pd.DataFrame(SENTENCE_RECORDS)
    df_hit = df[df["Hit_Count"].astype(int) >= 1].reset_index(drop=True)
    if df_hit.empty:
        print("❌ Step-1 未提取到句子，无法继续 Step-2 / Step-1 で文が取得できず、Step-2 を続行できません"); return

    company_db = list(canon_set) + list(alias_map.keys())   # canonical + alias
    comp_cols: List[List[str]] = []
    for sent in tqdm(df_hit["Sentence"].tolist(), desc="公司识别"):
        names_raw = extract_companies(sent, company_db, nlp)
        uniq: List[str] = []
        for alias in names_raw:
            if alias in uniq:     
                continue
            uniq.append(alias)                         # 保留句面原词，不做任何替换
        comp_cols.append(uniq[:MAX_COMP_COLS])

    for i in range(MAX_COMP_COLS):
        df_hit[f"company_{i+1}"] = [lst[i] if i < len(lst) else "" for lst in comp_cols]
        
# === ③ 先按数据库规则处理每行 company_n 列 ===
    ban_lower     = {b.lower() for b in ban_set}
    canon_lower   = {c.lower() for c in canon_set}
    alias_lower   = {a.lower(): canon for a, canon in alias_map.items()}
    canon_lower2orig = {c.lower(): c for c in canon_set}
    # —— 标准化：去掉所有非字母数字，再小写
    def _norm_key(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9]", "", s).lower()

    comp_cols = [f"company_{i+1}" for i in range(MAX_COMP_COLS)]

    for ridx in df_hit.index:
        orig_names = [df_hit.at[ridx, c].strip() for c in comp_cols if df_hit.at[ridx, c].strip()]
        new_names  = []
        for nm in orig_names:
            nm_l = nm.lower()
            # ① ban → 丢弃
            if nm_l in ban_lower:
                continue
            # ② 已是标准名 → 保留原样
            if nm_l in canon_lower:
                new_names.append(canon_lower2orig[nm_l])
                continue
            # ③ 别名 → 替换为对应 canonical
            if nm_l in alias_lower:
                new_names.append(alias_lower[nm_l])
                continue
            # ④ 未知 → 原样
            new_names.append(nm)

        # ⑤ 顺位左移 + “同根” 去重
        cleaned = []
        seen_keys = set()
        for nm in sorted(new_names, key=len, reverse=True):           # 先长后短
            key = _norm_key(nm)
            # 1) 与已选任何名称 key 前缀 / 后缀 相同 → 视为重复
            if any(key in k or k in key for k in seen_keys):
                continue
            cleaned.append(nm)
            seen_keys.add(key)
        # ⑥ 写回行（不足补空，用 .at）
        for i, col in enumerate(comp_cols):
            df_hit.at[ridx, col] = cleaned[i] if i < len(cleaned) else ""


    # === ④ 继续原流程写 result.csv（下方原代码保持不变） ===

    # ---- 组装精简版 result.csv ----
    meta_cols = ["Tier_1", "Tier_2", "Filename",
                 "Title", "Publisher", "Sentence",
                 "Hit_Count", "Matched_Keywords"]

    df_final = (df_hit[meta_cols +
                [c for c in df_hit.columns if c.startswith("company_")]]
                .fillna(""))
    df_final = dedup_company_cols(df_final)

    df_final.to_csv(BASE_DIR / "result.csv",
                    index=False, encoding="utf-8-sig")
    print(f"   · result.csv 已写 / 保存，共 {len(df_final)} 条记录 / 行")

       # ---- 生成 mapping_todo.csv ----
        # ---- 生成 mapping_todo.csv ----
    # 1) 为了能查到 canonical 的 id，先做一个 name→id 的字典
    canon_name2id = {row.canonical_name: row.id for row in df_canon.itertuples()}

    todo_rows: List[Dict] = []
    for _, row in df_final.iterrows():
        for alias in (
            row[c].strip()
            for c in df_final.columns if c.startswith("company_")
            if row[c].strip()
        ):
            # 已在三张表里出现过的 alias 不再进入 todo
            if alias in ban_set or alias in alias_map or alias in canon_set:
                continue

            # ---------- 计算 Advice 和 Adviced_ID ----------
            if canon_set:
                # 必须传入 query + choices 两个参数
                advice, score, _ = process.extractOne(
                    alias,                    # query
                    list(canon_set),          # choices
                    scorer=fuzz.WRatio
                )
                if score < 85:               # 相似度阈值
                    advice = ""
            else:
                advice = ""

            adviced_id = canon_name2id.get(advice, "")

            # ---------- 写入 todo_rows ----------
            todo_rows.append({
                "Sentence":       row["Sentence"],
                "Alias":          alias,
                "Bad_Rate":       calc_bad_rate(alias),
                "Advice":         advice,
                "Adviced_ID":     adviced_id,
                "Canonical_Name": "",
                "Std_Result":     ""
            })

    # ① 组装 DataFrame
    
    for r in todo_rows:
        r["Alias_lower"] = r["Alias"].lower()
    todo_df = (pd.DataFrame(todo_rows)
            .drop_duplicates("Alias_lower")
            .drop(columns="Alias_lower"))

    # ② 分组排序：0=High(≥60) → 1=Mid(30-59) → 2=Low(<30)
    todo_df["__grp"] = todo_df["Bad_Rate"].apply(
        lambda x: 0 if x >= 60 else (1 if x >= 30 else 2)
    )
    todo_df = (todo_df
               .sort_values(["__grp", "Sentence"], ascending=[True, True])
               .drop(columns="__grp"))

    # ③ 固定列顺序（可选）
    todo_df = todo_df[[
        "Sentence", "Alias", "Bad_Rate",
        "Advice", "Adviced_ID",          # ← 新增
        "Canonical_Name", "Std_Result"
    ]]

    # ④ 显示成百分比后写文件（排序已完成，安全）
    todo_df["Bad_Rate"] = todo_df["Bad_Rate"].astype(int).astype(str) + "%"

    todo_df.to_csv(BASE_DIR / "mapping_todo.csv",
                   index=False, encoding="utf-8-sig")
    print(f" mapping_todo.csv 生成 / 作成 {len(todo_df)} 条记录 / 行")
    print("✔ Step-2 完成 / 完了，请编辑 mapping_todo.csv 后运行 Step-3 / mapping_todo.csv を編集してから Step-3 を実行してください\n")
    print("""
┌──────────────────────────────────────────────────────────────┐
│  📄  mapping_todo.csv 简易填写指南 / mapping_todo.csv 簡易入力ガイド │
├──────────────────────────────────────────────────────────────┤
│ 1) 空白 → 跳过 / スキップ                                   │
│ 2) 0  → 加入 ban_list / ban_list に登録                     │
│ 3) 数字 n → 视为 canonical_id = n / 数字 n は ID として処理 │
│ 4) 其他文本 → 新或已有标准名 / それ以外の文字列 = 標準名     │
├──────────────────────────────────────────────────────────────┤
│ 保存后运行 Step-3 / 保存して Step-3 を実行してください        │
│      • 输入 2  → 运行 Step-3（写入数据库）                    │
│      • 输入 e  → 退出程序                                     │
│                                                              │
│  ※ 英語ガイド                                                │
│      • 2 を入力 → Step-3 実行                                 │
│      • e を入力 → 終了       
└──────────────────────────────────────────────────────────────┘
""")

# ================ Step-3 ==============

def step3(mysql_url: str):
    """
    Step-3 标准化 + 写库（与旧 NA_step3_standardize.py 等价）
    - Canonical_Name == ''  → Std_Result = 'No input'
    - Canonical_Name == '0' → 写 ban_list,  Std_Result = 'Banned'
    - 其它:
        • 若已存在 alias → Std_Result = 'Exists'
        • 否则插入/补全 canonical & alias, Std_Result = 'Added'
    同时把最新映射应用回 result.csv
    """
    # 本轮批次号：YYYYMMDD + 8位随机数
    process_id = datetime.now().strftime("%Y%m%d") + f"{random.randint(0, 99999999):08d}"
    res_f  = BASE_DIR / "result.csv"
    todo_f = BASE_DIR / "mapping_todo.csv"
    if not (res_f.exists() and todo_f.exists()):
        print("❌ 缺少 result.csv 或 mapping_todo.csv / result.csv または mapping_todo.csv が見つかりません"); sys.exit(1)

    # 读取
    df_res  = pd.read_csv(res_f,  dtype=str).fillna("")
    df_map  = pd.read_csv(todo_f, dtype=str).fillna("")
    
    if "Process_ID" not in df_map.columns:
        df_map["Process_ID"] = ""

    engine = create_engine(mysql_url)
    with engine.begin() as conn:

        # 1) 拉取三表到内存
        ban_set = {r[0] for r in conn.execute(text(
            "SELECT alias FROM ban_list"
        ))}
        canon_map = {r[0]: r[1] for r in conn.execute(text("SELECT id, canonical_name FROM company_canonical"))}  # id→name
        canon_rev = {v: k for k, v in canon_map.items()}  # name→id
        alias_map = {r[0]: r[1] for r in conn.execute(text("""
            SELECT a.alias, c.canonical_name
            FROM company_alias a
            JOIN company_canonical c ON a.canonical_id = c.id
        """))}

        # 2) 逐行处理 mapping
        for idx, row in df_map.iterrows():
            alias_raw = row["Alias"].strip()
            canon_input = row["Canonical_Name"].strip()
            
            did_write = False

            if not canon_input:                       # —— 空白
                df_map.at[idx, "Std_Result"] = "No input"
                continue

           # === ① Ban（输入 0） ===
            if canon_input == "0":
                if alias_raw not in ban_set:          # 只在第一次才写库＋打批次号
                    conn.execute(text(
                        "INSERT INTO ban_list(alias, process_id) "
                        "VALUES (:a, :pid)"
                    ), {"a": alias_raw, "pid": process_id})
                    ban_set.add(alias_raw)            # 别忘了同步到本地集合
                    df_map.at[idx, "Process_ID"] = f"'{process_id}"
                # 已存在就什么都不更改批次号
                df_map.at[idx, "Std_Result"] = "Banned"
                continue

            # === ② 用户输入数字 → 视为 canonical_id ===
            if canon_input.isdigit():
                cid = int(canon_input)
                if cid not in canon_map:              # id 不存在
                    df_map.at[idx, "Std_Result"] = "Bad ID"
                    continue
                canon_name = canon_map[cid]           # ← id → name
                canon_id   = cid
            else:
                canon_name = canon_input              # 用户直接写的文本
                # 若文本不存在于 canonical 表 → 新建
                if canon_name not in canon_rev:
                    res = conn.execute(text(
                        "INSERT INTO company_canonical(canonical_name, process_id) VALUES (:c, :pid)"
                    ), {"c": canon_name, "pid": process_id})
                    did_write = True
                    
                    canon_id = res.lastrowid
                    canon_rev[canon_name] = canon_id
                    df_map.at[idx, "Process_ID"] = f"'{process_id}"
                else:
                    canon_id = canon_rev[canon_name]

            # === ③ 写 alias（如已存在则跳过） ===
            if alias_raw in alias_map:
                df_map.at[idx, "Std_Result"] = "Exists"
                continue

            conn.execute(text(
                "INSERT IGNORE INTO company_alias(alias, canonical_id, process_id) VALUES (:a, :cid, :pid)"
            ), {"a": alias_raw, "cid": canon_id, "pid": process_id})
            did_write = True
            alias_map[alias_raw] = canon_name
            df_map.at[idx, "Std_Result"] = "Added"
            if did_write:
                df_map.at[idx, "Process_ID"]  = f"'{process_id}"

    # 3) 应用最新映射到 result.csv
    for col in [c for c in df_res.columns if c.startswith("company_")]:
        df_res[col] = df_res[col].apply(lambda x: alias_map.get(x, x))

    df_res = dedup_company_cols(df_res)
    # df_map["Process_ID"] = "'" + process_id   # ← 前面加单引号，Excel 会当文本
    df_res.to_csv(res_f, index=False, encoding="utf-8-sig")
    df_map.to_csv(todo_f, index=False, encoding="utf-8-sig")

    print(f"✔ Step-3 完成 / 完了：{len(df_map)} 条映射 / 件を処理，result.csv 已更新 / 更新完了")
    print(f"📌 本批次/今回の Process ID : {process_id}")
# ================ 主入口 ==============

def main():
    mysql_url = ask_mysql_url()
    try:
        create_engine(mysql_url).connect().close()
        print("✅ 数据库连通 / データベース接続成功")
    except Exception as e:
        print(f"❌ 数据库连接失败 / データベース接続失敗: {e}"); sys.exit(1)

    choice = choose()

    if choice == "1":
        step1()
        step2(mysql_url)

        # —— 新增：跑完 Step-2 后等待用户指令 ——
        while True:
            nxt = input("👉 输入 2 继续 Step-3，或输入 e 退出 / 2 でStep-3を続行, e で終了: ").strip().lower()
            if nxt == "2":
                step3(mysql_url)
                break
            elif nxt == "e":
                print("已退出 / 終了しました。")
                return
            else:
                print("无效输入 / 無効な入力です，请重新输入 / もう一度入力してください。")

    else:   # choice == "2"
        step3(mysql_url)


if __name__ == "__main__":
    main()
