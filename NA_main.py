#!/usr/bin/env python3
# coding: utf-8
"""
na_pipeline.py  ——  单文件版（Step‑1 对齐 + 扩展公司识别）
2025‑07‑08  rev‑C
"""

import os, re, sys, unicodedata, string
from pathlib import Path
from typing import List, Dict, Set

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
        key = input("请输入秘钥：user:pass@host\n>>>>>> ").strip()
        key_file.write_text(key)                     # 缓存下次用
    return f"mysql+pymysql://{key}.mysql.rds.aliyuncs.com:3306/na_data?charset=utf8mb4"

def choose() -> str:
    print("\n1) 初次运行（Step-1 ➜ Step-2）\n2) 适用 mapping（Step-3）")
    c = input("输入 1 或 2：").strip();
    if c not in {"1", "2"}:
        print("❌ 无效选择"); sys.exit(1)
    return c

def dedup_company_cols(df: pd.DataFrame) -> pd.DataFrame:
    comp_cols = [c for c in df.columns if c.startswith("company_")]
    for _, row in df.iterrows():
        seen: Set[str] = set()
        for col in comp_cols:
            val = str(row[col]).strip()
            row[col] = "" if val in seen else val
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
    print("\n▶ Step-1: 提取 docx 句子 …")
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
    print(f"✔ Step-1 完成：共 {len(all_recs)} 条记录（已缓存）")
# ---------------- Step‑2 核心函数修订 ----------------

def is_valid_token(tok:str)->bool:
    tok=tok.strip()
    if not tok or all(c in "-–—・.、。！？／ー" for c in tok): return False
    if re.search(r"\d",tok) and not re.search(r"[A-Za-z]",tok): return False
    return True

def extract_companies(text:str, company_db:List[str], ner, thresh:int=95)->List[str]:
    comps:set[str]=set(); txt=re.sub(r"\s*\d{1,2}/\d{1,2}/\d{2,4}.*$","",text).strip()
    # 1) spaCy ORG
    for e in ner(txt).ents:
        if e.label_=="ORG" and is_valid_token(e.text): comps.add(e.text.strip())
    # 2) token fuzzy
    for pos,raw in enumerate(re.findall(r"\b\S+\b",txt)):
        if pos==0 or raw.lower() in STOPWORDS: continue
        tok=raw.split("@")[ -1 ].split("/")[-1].strip(".,()")
        if len(tok)<3 or not tok[0].isupper() or not is_valid_token(tok): continue
        best,score,_=process.extractOne(tok,company_db,scorer=fuzz.token_set_ratio)
        if score>=thresh: comps.add(best)
    # 3) ≤3-word Caps phrase
    for m in re.finditer(r"(?:[A-Z][\w&’\-]+(?:\s+[A-Z][\w&’\-]+){0,2})",txt):
        phrase=m.group(0).strip()
        if phrase.isupper() and len(phrase)<3: continue
        comps.add(phrase)
    return list(comps)

# ---- 其余 step2, step3, main 与上一版一致 ----


# ---------------- Step‑2 核心函数修订 ----------------

def is_valid_token(tok:str)->bool:
    tok=tok.strip()
    if not tok or all(c in "-–—・.、。！？／ー" for c in tok): return False
    if re.search(r"\d",tok) and not re.search(r"[A-Za-z]",tok): return False
    return True

def extract_companies(text:str, company_db:List[str], ner, thresh:int=95)->List[str]:
    comps:set[str]=set(); txt=re.sub(r"\s*\d{1,2}/\d{1,2}/\d{2,4}.*$","",text).strip()
    # 1) spaCy ORG
    for e in ner(txt).ents:
        if e.label_=="ORG" and is_valid_token(e.text): comps.add(e.text.strip())
    # 2) token fuzzy
    for pos,raw in enumerate(re.findall(r"\b\S+\b",txt)):
        if pos==0 or raw.lower() in STOPWORDS: continue
        tok=raw.split("@")[ -1 ].split("/")[-1].strip(".,()")
        if len(tok)<3 or not tok[0].isupper() or not is_valid_token(tok): continue
        best,score,_=process.extractOne(tok,company_db,scorer=fuzz.token_set_ratio)
        if score>=thresh: comps.add(best)
    # 3) ≤3-word Caps phrase
    for m in re.finditer(r"(?:[A-Z][\w&’\-]+(?:\s+[A-Z][\w&’\-]+){0,2})",txt):
        phrase=m.group(0).strip()
        if phrase.isupper() and len(phrase)<3: continue
        comps.add(phrase)
    return list(comps)
# ----------------—— Step‑2 ——----------------

def is_valid_token(token: str) -> bool:
    token = token.strip()
    if not token or all(c in "-–—・.、。！？／ー" for c in token):
        return False
    if re.search(r"\d", token) and not re.search(r"[A-Za-z]", token):
        return False
    if "  " in token:
        return False
    return True


def extract_companies(text: str, company_db: List[str], ner_model, fuzzy_threshold: int = 95) -> List[str]:
    comps: Set[str] = set()
    text_clean = re.sub(r"\s*\d{1,2}/\d{1,2}/\d{2,4}.*$", "", text).strip()

    # 1) spaCy ORG 实体
    doc = ner_model(text_clean)
    for ent in doc.ents:
        if ent.label_ != "ORG":
            continue
        ent_text = ent.text.strip()
        if "  " in ent_text or re.search(r"[\d/%+]", ent_text):
            continue
        valid_ent = True
        for w in ent_text.split():
            if not w[0].isalpha() or w in STOPWORDS or not is_valid_token(w):
                valid_ent = False; break
        if valid_ent:
            comps.add(ent_text)

    # 2) token‑级模糊匹配兜底
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
        best, score, _ = process.extractOne(token, company_db, scorer=fuzz.token_set_ratio)
        if score >= fuzzy_threshold:
            comps.add(best)
    return list(comps)


def step2(mysql_url: str):
    print("\n▶ Step-2: 公司识别 + ban 过滤 …")
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
    print(f"   · ban_list {len(ban_set)} 条，alias_map {len(alias_map)} 条，canon_set {len(canon_set)} 条")

    df = pd.DataFrame(SENTENCE_RECORDS)
    df_hit = df[df["Hit_Count"].astype(int) >= 1].reset_index(drop=True)
    if df_hit.empty:
        print("❌ Step-1 没提取到任何句子，无法继续 Step-2"); return

    company_db = list(canon_set)
    comp_cols: List[List[str]] = []
    for sent in tqdm(df_hit["Sentence"].tolist(), desc="公司识别"):
        names_raw = extract_companies(sent, company_db, nlp, 95)
        uniq: List[str] = []
        for alias in names_raw:
            mapped = alias_map.get(alias, alias)
            if mapped in ban_set or mapped in uniq:
                continue
            uniq.append(mapped)
        comp_cols.append(uniq[:MAX_COMP_COLS])

    for i in range(MAX_COMP_COLS):
        df_hit[f"company_{i+1}"] = [lst[i] if i < len(lst) else "" for lst in comp_cols]

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
    print(f"   · result.csv 已写，共 {len(df_final)} 条记录")

    # ---- 生成 mapping_todo.csv ----
    todo_rows: List[Dict] = []
    for _, row in df_final.iterrows():          # 只遍历最终结果
        for alias in (row[c] for c in df_final.columns
                      if c.startswith("company_")):
            alias = alias.strip()
            if (not alias or alias in alias_map
                    or alias in canon_set or alias in ban_set):
                continue
            if any(fuzz.token_set_ratio(alias, canon) >= 85
                   for canon in canon_set):
                continue
            todo_rows.append({
                "Sentence":       row["Sentence"],
                "Alias":          alias,
                "Canonical_Name": "",
                "Std_Result":     ""
            })

    todo_df = (pd.DataFrame(todo_rows)
               .drop_duplicates("Alias")
               .sort_values("Alias"))
    todo_df.to_csv(BASE_DIR / "mapping_todo.csv",
                   index=False, encoding="utf-8-sig")
    print(f"   · mapping_todo.csv 生成 {len(todo_df)} 条记录")
    print("✔ Step-2 完成，请编辑 mapping_todo.csv 后运行 Step-3")

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
    res_f  = BASE_DIR / "result.csv"
    todo_f = BASE_DIR / "mapping_todo.csv"
    if not (res_f.exists() and todo_f.exists()):
        print("❌ 缺少 result.csv 或 mapping_todo.csv"); sys.exit(1)

    # 读取
    df_res  = pd.read_csv(res_f,  dtype=str).fillna("")
    df_map  = pd.read_csv(todo_f, dtype=str).fillna("")

    engine = create_engine(mysql_url)
    with engine.begin() as conn:

        # 1) 拉取三表到内存
        ban_set   = {r[0] for r in conn.execute(text("SELECT alias FROM ban_list"))}
        canon_map = {r[0]: r[1] for r in conn.execute(text("SELECT id, canonical_name FROM company_canonical"))}  # id→name
        canon_rev = {v: k for k, v in canon_map.items()}  # name→id
        alias_map = {r[0]: r[1] for r in conn.execute(text("""
            SELECT a.alias, c.canonical_name
            FROM company_alias a
            JOIN company_canonical c ON a.canonical_id = c.id
        """))}

        # 2) 逐行处理 mapping
        for idx, row in df_map.iterrows():
            alias = row["Alias"].strip()
            canon = row["Canonical_Name"].strip()

            if not canon:
                df_map.at[idx, "Std_Result"] = "No input"
                continue

            # 2-A Ban
            if canon == "0":
                if alias not in ban_set:
                    conn.execute(text("INSERT IGNORE INTO ban_list(alias) VALUES (:a)"), {"a": alias})
                    ban_set.add(alias)
                df_map.at[idx, "Std_Result"] = "Banned"
                continue

            # 2-B 已有映射
            if alias in alias_map:
                df_map.at[idx, "Std_Result"] = "Exists"
                continue

            # 2-C canonical 不存在 → 新建
            if canon not in canon_rev:
                res = conn.execute(
                    text("INSERT INTO company_canonical(canonical_name) VALUES (:c)"),
                    {"c": canon}
                )
                canon_id = res.lastrowid
                canon_rev[canon] = canon_id
            else:
                canon_id = canon_rev[canon]

            # 2-D 写 alias
            conn.execute(
                text("INSERT INTO company_alias(alias, canonical_id) VALUES (:a, :cid)"),
                {"a": alias, "cid": canon_id}
            )
            alias_map[alias] = canon
            df_map.at[idx, "Std_Result"] = "Added"

    # 3) 应用最新映射到 result.csv
    for col in [c for c in df_res.columns if c.startswith("company_")]:
        df_res[col] = df_res[col].apply(lambda x: alias_map.get(x, x))

    df_res = dedup_company_cols(df_res)
    df_res.to_csv(res_f, index=False, encoding="utf-8-sig")
    df_map.to_csv(todo_f, index=False, encoding="utf-8-sig")

    print(f"✔ Step-3 完成：{len(df_map)} 条映射已处理，result.csv 更新完毕")
# ================ 主入口 ==============

def main():
    mysql_url = ask_mysql_url()
    try:
        create_engine(mysql_url).connect().close()
        print("✅ 数据库连通")
    except Exception as e:
        print(f"❌ 数据库连接失败: {e}"); sys.exit(1)

    if choose() == "1":
        step1()
        step2(mysql_url)
    else:
        step3(mysql_url)
    print("\n🎉 流程完成")


if __name__ == "__main__":
    main()
