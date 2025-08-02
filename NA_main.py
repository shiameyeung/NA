# 原作者：杨天乐@关西大学 / Author: Shiame Yeung@Kansai University / 作成者：楊　天楽@関西大学
#!/usr/bin/env python3
# coding: utf-8
"""
na_pipeline.py  ——  单文件版（Step‑1 对齐 + 扩展公司识别）
2025‑07‑08  rev‑C
"""

def cute_box(cn: str, jp: str, icon: str = "🌸") -> None:
    """
    多行也能对齐的可爱中/日双语框
    cn: 中文提示（可以多行，用 '\\n' 分隔）
    jp: 日文提示（可以多行）
    icon: 每行开头和结尾的小表情
    """
    # 把中/日各自的多行拆开，拼成统一列表
    lines = []
    for segment in (cn, jp):
        for ln in segment.split("\n"):
            ln = ln.strip()
            # 用 "icon + 空格 + 文本 + 空格 + icon" 构造每一行
            lines.append(f"{icon} {ln} {icon}")

    # 找到最长那行，做为框宽
    width = max(len(ln) for ln in lines)
    border = "─" * width

    # 打印上边框
    print(f"╭{border}╮")
    # 打印每一行，右侧填充空格到 width
    for ln in lines:
        print("│" + ln.ljust(width) + "│")
    # 打印下边框
    print(f"╰{border}╯")

import sys, subprocess, os

def ensure_env() -> None:
    """
    ❶ 判断 Python 版本，给出“老依赖 / 新依赖”两套清单  
    ❷ 先升级 pip / setuptools / wheel，再确保 packaging 与 requests 存在  
    ❸ 安装或升级其余依赖；缺什么自动补什么  
    ❹ 若本轮确实安装过东西，则 cute_box 提示后 sys.exit(0)，
       让用户重新执行主脚本；否则直接返回继续跑
    """
    import sys, subprocess

    # ---------- 0. 小工具 ----------
    def pip_install(pkgs: list[str]):
        """统一 pip 安装入口（带 -U 升级）"""
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-U", *pkgs],
            stdout=subprocess.DEVNULL  # 保持输出简洁，可按需去掉
        )

    # ---------- 1. 先升级 pip / setuptools / wheel ----------
    cute_box(
        "正在升级 pip / setuptools / wheel …",
        "pip / setuptools / wheel をアップグレード中…",
        "🔧"
    )
    pip_install(["pip", "setuptools", "wheel"])

    # ---------- 2. 确保 packaging & requests 存在 ----------
    for base_pkg, jp_name in [("packaging", "packaging"), ("requests", "requests")]:
        try:
            __import__(base_pkg)
        except ImportError:
            cute_box(
                f"缺少 {base_pkg}，正在安装…",
                f"{jp_name} がありません。インストール中…",
                "📦"
            )
            pip_install([base_pkg])

    # 之后要用 packaging 里的版本比较
    from importlib.metadata import version, PackageNotFoundError
    from packaging.specifiers import SpecifierSet
    from packaging.requirements import Requirement

    # ---------- 3. 根据 Python 版本决定依赖 ----------
    py_major, py_minor = sys.version_info[:2]
    is_new_py = (py_major, py_minor) >= (3, 13)

    if is_new_py:
        core_pkgs = ["numpy>=2.0.0", "spacy>=3.8.7", "thinc>=8.3.6", "blis>=1.0.0"]
        torch_spec = "torch==2.6.0"          # PyTorch 目前对 3.13 仅此版本
    else:
        core_pkgs = ["numpy<2.0.0", "spacy<3.8.0", "thinc<8.3.0", "blis<0.8.0"]
        torch_spec = "torch>=2,<2.3"

    common_pkgs = [
        "pandas", "tqdm", "sqlalchemy", "pymysql",
        "rapidfuzz", "python-docx", "sentence-transformers",
        torch_spec,
    ]

    wanted = core_pkgs + common_pkgs

    # ---------- 4. 判断哪些包需要安装 / 升级 ----------
    def need_install(spec: str) -> bool:
        req = Requirement(spec)
        try:
            cur_ver = version(req.name)
        except PackageNotFoundError:
            return True
        return cur_ver not in SpecifierSet(str(req.specifier))

    to_install = [spec for spec in wanted if need_install(spec)]

    did_install = False
    if to_install:
        cute_box(
            "安装 / 升级以下依赖：\n" + "\n".join(to_install),
            "次の依存関係をインストール / アップグレードします：\n" + "\n".join(to_install),
            "📦"
        )
        try:
            pip_install(to_install)
            did_install = True
        except subprocess.CalledProcessError:
            cute_box(
                "❌ 自动安装失败，请手动根据日志解决依赖后重试",
                "❌ 自動インストール失敗。ログを確認し、手動で依存関係を解決してください",
                "⚠️"
            )
            sys.exit(1)

    # ---------- 5. 确保 spaCy 英文模型 ----------
    try:
        import spacy
        spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        cute_box(
            "下载 spaCy 模型 en_core_web_sm …",
            "spaCy モデル en_core_web_sm をダウンロード中…",
            "🔄"
        )
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        did_install = True

    # ---------- 6. 结束语 ----------
    cute_box(
        "依赖检查完毕，脚本可以运行！",
        "依存関係チェック完了。スクリプトを実行できます！",
        "🎉"
    )
    if did_install:
        cute_box(
            "首次/刚升级完，请重新运行主脚本。",
            "初回実行またはアップグレード直後です。もう一度メインスクリプトを実行してください。",
            "🔁"
        )
        sys.exit(0)

# —————— 在脚本一启动就先确保环境 ——————
ensure_env()

import os, re, sys, unicodedata, string
from pathlib import Path
from typing import List, Dict, Set

from datetime import datetime
import random

import itertools

import pandas as pd
from tqdm import tqdm
from sqlalchemy import create_engine, text
from rapidfuzz import fuzz, process
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

try:
    from docx import Document
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    model_emb = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
except Exception:
    cute_box(
      "缺少依赖：请运行 pip install python-docx spacy",
      "依存関係が足りません：pip install python-docx spacy を実行してね",
      "⚠️"
    )
    sys.exit(1)

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

# ── 金融报表 / 业绩公告类 ─────────────────────────────
FIN_REPORT = re.compile(
    r'\b(results?|earnings?|revenues?|turnover|profit(?:s)?|loss(?:es)?|guidance|forecast|'
    r'financial statements?|balance sheets?|cash flows?|income statements?)\b',
    re.I)

# ── 分季 / 分半期 / 分年描述 ─────────────────────────
ORDINAL_PERIOD = re.compile(
    r'\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth)\b.*?\b(quarter|half|year)\b',
    re.I)

# ── 典型“公告/报告/更新”触发词（多见于 ban_list） ─────────────
ANNOUNCE_VERB = re.compile(
    r'\b(reports?|announces?|updates?|revises?|publishes?|files?|issues?|unveils?)\b',
    re.I)
# ===== 在常量区（TIME_QTY 之后）新增几条通用 regex =====
GENERIC_NOUN = re.compile(
    r'\b(services?|solutions?|systems?|platforms?|programs?|projects?|'
    r'statements?|reports?|targets?|technologies?|operations?|activities|'
    r'strategies?|plans?)\b', re.I)

MONTH_NAME = re.compile(
    r'\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|'
    r'jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|'
    r'dec(?:ember)?)\b', re.I)

NEW_GENERIC_TIME = re.compile(
    r'\b(?:end|beginning|middle|start|first|second|third|fourth|prior|previous|'
    r'current|next)\s+(?:of\s+)?(?:the\s+)?(?:year|quarter|month|week)s?\b',
    re.I)

#  纯大写 2–4 位缩写（PBM / ESG …）
ALLCAP_SHORT = re.compile(r'^[A-Z]{2,4}$')

#  %、百万/十亿、美元符号之类
NUMERIC = re.compile(r'[%\$]\s*\d|\d[\d,\.]+\s*(?:million|billion|thousand)', re.I)


ALL_UPPER  = re.compile(r'^[A-Z]{2,}$')
ALL_LOWER  = re.compile(r'^[a-z]{4,}$')

SHORT_TOKEN = re.compile(r'^[A-Za-z]{1,4}$')
ART_LOWER   = re.compile(r'^\s*(a|an|about|approximately|the|this|that|those)\s+[a-z]')
GENERIC_END = re.compile(
    r'\b(plan|plans?|programs?|systems?|platforms?|services?|solutions?|operations?|'
    r'agreements?|strategies?|reports?|statements?)$', re.I)

def _lower_ratio(text: str) -> float:
    w = text.split()
    return sum(t[0].islower() for t in w) / len(w) if w else 0

def calc_Bad_Score(text: str) -> int:
    """
    越高越可能是 bad（需要人工判斷或直接 ban）
    调整点：
      ① 先按“好特征”清零——例如合法公司后缀。
    """
    # === ① 明显好特征：直接判 0 ===
    if ORG_SUFFIX.search(text):           # Inc., Ltd. 等
        return 0

    score = 0

    # === ② 时间 & 数量类 ===
    if TIME_QTY.search(text) or MONTH_NAME.search(text):
        score += 40                       # 季度/月份/年，几乎一定是假公司

    # === ③ 句式 & 组合词 ===
    if ' of the ' in text.lower():        # “…of the…” 典型报告语
        score += 20
    if GENERIC_END.search(text):
        score += 15
    if GENERIC_NOUN.search(text):         # 新增：泛称名词
        score += 15

    # === ④ 大小写 & 长度 ===
    words = text.split()
    if len(words) <= 2:
        score += 20
    if _lower_ratio(text) > 0.30:
        score += 15                       # 原来是 20，稍微放宽
    if ALL_UPPER.match(text) or ALL_LOWER.match(text):
        score += 15
    if any(SHORT_TOKEN.match(w) for w in words):
        score += 10                       # 像“LLC”“LP”这种很短的 token
        
    if FIN_REPORT.search(text):        score += 30        
    if ORDINAL_PERIOD.search(text):    score += 25        
    if ANNOUNCE_VERB.search(text):     score += 20    
    if NEW_GENERIC_TIME.search(text):      score += 40  # time 相关更狠
    if ALLCAP_SHORT.match(text):           score += 50  # 纯缩写
    if NUMERIC.search(text):               score += 25  # 含数值/金额    


    return score
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
    cute_box(
        "① 初次运行（Step-1 ➜ Step-2）\n② mapping适用/邻接（Step-3/4）\n作者：杨 天乐 支持：李 宗昊 李 佳璇 @関西大学　伊佐田研究室",
        "① 初回実行（Step-1 ➜ Step-2）\n② mapping適用/隣接（Step-3/4）\n作成者：楊 天楽　協力：李 宗昊 李 佳璇 @関西大学　伊佐田研究室",
        "📋"
    )
    c = input("请输入 1 或 2 / 1 か 2 を入力してください: ").strip()

    # ── 2. 校验 ───────────────────────────────────────────
    if c not in {"1", "2"}:
        cute_box(
        "无效选择，请输入 1 或 2！",
        "無効な選択です。1 か 2 を入力してね！",
        "🔄"
        )
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
    cute_box(
        "Step-1：提取 Word 句子 中…",
        "Step-1：文抽出中…",
        "📄"
    )
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
    cute_box(
        f"Step-1 完成，共 {len(all_recs)} 条记录",
        f"Step-1 完了しました：全{len(all_recs)}件",
        "✅"
    )

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
    cute_box(
        "Step-2：公司识别＋BAN 过滤 中…",
        "Step-2：企業名認識＋BAN フィルタ中…",
        "🏷️"
    )
    # 单独导出 canonical 表（engine_tmp）
    engine_tmp = create_engine(mysql_url)            # ← 新建
    df_canon = pd.read_sql("SELECT id, canonical_name FROM company_canonical", engine_tmp)
    df_canon.to_csv(BASE_DIR / "canonical_list.csv", index=False, encoding="utf-8-sig")
    cute_box(
        f"已写出 canonical_list.csv，共 {len(df_canon)} 行",
        f"canonical_list.csv を保存しました：{len(df_canon)} 行",
        "🗂️"
    )
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
        # —— 预编码全部 canonical，一次搞定 ——
        canon_names = list(canon_set)
        canon_vecs  = model_emb.encode(canon_names, batch_size=64, normalize_embeddings=True)
        # ↓↓↓ 新增：名字→ID 的字典，用于 Advice 对应的 ID
        rows2 = conn.execute(text(
            "SELECT id, canonical_name FROM company_canonical"
        ))
        canon_name2id = {name: cid for cid, name in rows2}      # ← 新增
    
    cute_box(
    f"ban_list={len(ban_set)}，alias_map={len(alias_map)}，canon_set={len(canon_set)}",
    f"ban_list：{len(ban_set)}件／alias_map：{len(alias_map)}件／canon_set：{len(canon_set)}件",
    "🔍"
    )

    df = pd.DataFrame(SENTENCE_RECORDS)
    df_hit = df[df["Hit_Count"].astype(int) >= 1].reset_index(drop=True)
    if df_hit.empty:
        cute_box(
        "Step-1 没提取到任何句子，请先跑 Step-1！",
        "Step-1 で文が取得できませんでした。まず Step-1 を実行してね",
        "🚫"
        )
        return

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
    cute_box(
        f"已生成 result.csv，共 {len(df_final)} 条记录",
        f"result.csv を生成しました：全{len(df_final)}件",
        "📑"
    )
    
    # ---- 生成 result_mapping_todo.csv （空表安全 + 统计）----
    # 1) name→id 字典（Advice 需要）
    canon_name2id = {row.canonical_name: row.id for row in df_canon.itertuples()}

    todo_rows: List[Dict] = []

    # 统计：哪些被过滤/跳过
    ban_hits = alias_hits = canon_hits = 0
    rows_skipped_not_enough_companies = 0  # 同行公司总数 < 2 的行

    comp_cols = [c for c in df_final.columns if c.startswith("company_")]

    for _, row in df_final.iterrows():
        # 取出该行所有非空企业名（已做过ban/alias/canonical一次清洗与同根去重）
        names = [row[c].strip() for c in comp_cols if row[c].strip()]

        # ✅ 行级门槛：无论是否已知，只要同行公司总数 < 2，就整行跳过
        if len(names) < 2:
            rows_skipped_not_enough_companies += 1
            continue

        # 只把“未知别名”写入 todo（已知的不需要人工映射）
        unknowns: List[str] = []
        for alias in names:
            alias_l = alias.lower()
            if alias_l in ban_lower:
                ban_hits += 1
                continue
            if alias_l in alias_lower:
                alias_hits += 1
                continue
            if alias_l in canon_lower:
                canon_hits += 1
                continue
            unknowns.append(alias)

        # 该行达到“有2+家公司”的门槛，即使 unknowns 只有1个或更多，都进入 todo
        for alias in unknowns:
            # --- 首词命中 canonical ---
            first_tok = re.split(r'[\s\-]+', alias, maxsplit=1)[0].lower()
            if first_tok in canon_lower:
                advice     = canon_lower2orig[first_tok]
                adviced_id = canon_name2id.get(advice, "")
            else:
                # --- n-gram 完全匹配 ---
                advice = adviced_id = ""
                words = alias.split()
                L = len(words)
                for size in range(L, 0, -1):
                    for i in range(0, L - size + 1):
                        phrase = " ".join(words[i:i+size])
                        key = phrase.lower()
                        if key in canon_lower:
                            advice     = canon_lower2orig[key]
                            adviced_id = canon_name2id.get(advice, "")
                            break
                    if advice:
                        break

            # --- 语义相似度（可选） ---
            if not advice and canon_vecs.size > 0:
                alias_vec  = model_emb.encode([alias], normalize_embeddings=True)[0]
                sims       = np.dot(canon_vecs, alias_vec)
                best_idx   = int(np.argmax(sims))
                best_score = float(sims[best_idx])
                if best_score >= 0.80:
                    advice     = canon_names[best_idx]
                    adviced_id = canon_name2id.get(advice, "")

            # 写入 todo
            todo_rows.append({
                "Sentence":        row["Sentence"],
                "Alias":           alias,
                "Bad_Score":       calc_Bad_Score(alias),
                "Advice":          advice or "",
                "Adviced_ID":      adviced_id or "",
                "Canonical_Name":  "",
                "Std_Result":      ""
            })

    # 2) 组装 DataFrame（空表安全）
    todo_cols = [
        "Sentence", "Alias", "Bad_Score",
        "Advice", "Adviced_ID",
        "Canonical_Name", "Std_Result"
    ]

    if not todo_rows:
        # —— 没有新的别名需要映射：写出只有表头的空表，并友好提示
        todo_df = pd.DataFrame(columns=todo_cols)
        todo_df.to_csv(BASE_DIR / "result_mapping_todo.csv",
                       index=False, encoding="utf-8-sig")

        cute_box(
            "本批没有产生新的别名需要映射；已被规则识别/过滤，或因“仅 1 个疑似企业名”规则而跳过。\n"
            f"ban 命中：{ban_hits}，已有 alias：{alias_hits}，已有 canonical：{canon_hits}，同行公司不足跳过：{rows_skipped_not_enough_companies}",
            "今回のバッチでは新しい別名はありません。既存データに一致／除外、または「候補が1件のみ」規則でスキップされました。\n"
            f"ban 一致：{ban_hits}／既存エイリアス：{alias_hits}／既存カノニカル：{canon_hits}／同一行の企業数不足スキップ：{rows_skipped_not_enough_companies}",
            "ℹ️"
        )
    else:
        # —— 正常去重（按别名小写）
        todo_df = pd.DataFrame(todo_rows)
        todo_df["__alias_l"] = todo_df["Alias"].str.lower()
        todo_df = todo_df.drop_duplicates("__alias_l").drop(columns="__alias_l")

        # 分组排序
        todo_df["__grp"] = todo_df["Bad_Score"].apply(lambda x: 0 if x >= 50 else (1 if x >= 10 else 2))
        todo_df = (todo_df
                   .sort_values(["__grp", "Sentence"], ascending=[True, True])
                   .drop(columns="__grp"))

        # 固定列顺序
        for col in todo_cols:
            if col not in todo_df.columns:
                todo_df[col] = ""   # 兜底，保证列齐全
        todo_df = todo_df[todo_cols]

        # 写文件
        todo_df["Bad_Score"] = todo_df["Bad_Score"].astype(int).astype(str)
        todo_df['Sentence'] = todo_df['Sentence'].apply(
            lambda s: "'" + s if isinstance(s, str) and s.startswith('=') else s
        )
        todo_df.to_csv(BASE_DIR / "result_mapping_todo.csv",
                       index=False, encoding="utf-8-sig")

        cute_box(
            f"已生成 result_mapping_todo.csv，共 {len(todo_df)} 条待处理别名。\n"
            f"（ban 命中：{ban_hits}，已有 alias：{alias_hits}，已有 canonical：{canon_hits}，单一候选跳过：{single_suspect_skipped}）",
            f"result_mapping_todo.csv を作成：{len(todo_df)} 件の候補。\n"
            f"（ban 一致：{ban_hits}／既存エイリアス：{alias_hits}／既存カノニカル：{canon_hits}／単一候補スキップ：{single_suspect_skipped}）",
            "📝"
        )
        
    cute_box(
    "Step-2 完成！请编辑 result_mapping_todo.csv 然后运行 Step-3",
    "Step-2 完了！result_mapping_todo.csv を編集してから Step-3 を実行してね",
    "✅"
    )
    cute_box(
        "result_mapping_todo.csv 快速填写指南：\n"
        "1) 空白→跳过\n"
        "2) 0→加 ban_list\n"
        "3) n→视为 canonical_id\n"
        "4) 其他→新/已有标准名",
        "result_mapping_todo.csv 簡易入力ガイド：\n"
        "1) ブランク→スキップ\n"
        "2) 0→ban_list登録\n"
        "3) n→canonical_id と見なす\n"
        "4) その他→新規/既存標準名",
        "📋"
    )

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
    todo_f = BASE_DIR / "result_mapping_todo.csv"
    if not (res_f.exists() and todo_f.exists()):
        cute_box(
            "找不到 result.csv 或 result_mapping_todo.csv，请先生成它们",
            "result.csv または result_mapping_todo.csv が見つかりません。先に作成してね",
            "❗"
        )
        sys.exit(1)

    # 读取现有文件
    df_res = pd.read_csv(res_f,  dtype=str).fillna("")
    df_map = pd.read_csv(todo_f, dtype=str).fillna("")
    if "Process_ID" not in df_map.columns:
        df_map["Process_ID"] = ""

    engine = create_engine(mysql_url)
    with engine.begin() as conn:
        # 拉取三表到内存
        ban_set    = {r[0] for r in conn.execute(text("SELECT alias FROM ban_list"))}
        canon_map  = {r[0]: r[1] for r in conn.execute(text("SELECT id, canonical_name FROM company_canonical"))}
        alias_map  = {r[0]: r[1] for r in conn.execute(text(
            "SELECT a.alias, c.canonical_name FROM company_alias a "
            "JOIN company_canonical c ON a.canonical_id=c.id"
        ))}
        # 无视大小写的镜像
        ban_lower       = {b.lower() for b in ban_set}
        alias_lower_map = {a.lower(): c for a, c in alias_map.items()}
        canon_lower2id  = {name.lower(): cid for cid, name in canon_map.items()}

    # —— 处理 todo 映射 ——  
    for idx, row in df_map.iterrows():
        alias_raw   = row["Alias"].strip()
        alias_raw_l = alias_raw.lower()
        canon_input = row["Canonical_Name"].strip()
        if not canon_input:
            df_map.at[idx, "Std_Result"] = "No input"
            continue

        # ① Ban（输入 0）
        if canon_input == "0":
            if alias_raw_l not in ban_lower:
                with engine.begin() as conn:
                    conn.execute(text(
                        "INSERT INTO ban_list(alias, process_id) VALUES (:a, :pid)"
                    ), {"a": alias_raw, "pid": process_id})
                ban_lower.add(alias_raw_l)
            df_map.at[idx, "Std_Result"]   = "Banned"
            df_map.at[idx, "Process_ID"] = f"'{process_id}"
            continue

        # ② 数字 → 视为 existing canonical_id
        if canon_input.isdigit():
            cid = int(canon_input)
            if cid not in canon_map:
                df_map.at[idx, "Std_Result"] = "Bad ID"
                continue
            canon_name = canon_map[cid]
        else:
            # 新 canonical
            ci_l = canon_input.lower()
            if ci_l not in canon_lower2id:
                with engine.begin() as conn:
                    res = conn.execute(text(
                        "INSERT INTO company_canonical(canonical_name, process_id) VALUES (:c, :pid)"
                    ), {"c": canon_input, "pid": process_id})
                new_id = res.lastrowid
                canon_map[new_id]        = canon_input
                canon_lower2id[ci_l] = new_id
                df_map.at[idx, "Process_ID"] = f"'{process_id}"
                canon_name = canon_input
            else:
                canon_name = canon_map[canon_lower2id[ci_l]]
        # ③ 写 alias（忽略大小写已存在）
        if alias_raw_l in alias_lower_map or alias_raw_l in canon_lower2id:
            df_map.at[idx, "Std_Result"] = "Exists"
            continue
        with engine.begin() as conn:
            conn.execute(text(
                "INSERT IGNORE INTO company_alias(alias, canonical_id, process_id) "
                "VALUES (:a, :cid, :pid)"
            ), {"a": alias_raw, "cid": canon_lower2id[canon_name.lower()], "pid": process_id})
        alias_lower_map[alias_raw_l] = canon_name
        df_map.at[idx, "Std_Result"]   = "Added"
        df_map.at[idx, "Process_ID"] = f"'{process_id}"

    # 先写回 todo，再做回写 result.csv
    df_map.to_csv(todo_f, index=False, encoding="utf-8-sig")

    # ====== 将最新映射应用回 result.csv ======
    # 重新拉取 ban/alias/canonical 准备映射
    with engine.begin() as conn2:
        ban_set2    = {r[0] for r in conn2.execute(text("SELECT alias FROM ban_list"))}
        rows2       = conn2.execute(text(
            "SELECT a.alias, c.canonical_name FROM company_alias a "
            "JOIN company_canonical c ON a.canonical_id=c.id"
        ))
        alias_map2  = {a: c for a, c in rows2}
        canon_set2  = {r[0] for r in conn2.execute(text("SELECT canonical_name FROM company_canonical"))}

    ban_lower2        = {b.lower() for b in ban_set2}
    alias_lower_map2  = {a.lower(): c for a, c in alias_map2.items()}
    canon_lower2orig2 = {c.lower(): c for c in canon_set2}

    comp_cols = [c for c in df_res.columns if c.startswith("company_")]

    def _norm_key(s: str) -> str:
        return re.sub(r"[^A-Za-z0-9]", "", str(s)).lower()

    changed_cells = 0
    for ridx in df_res.index:
        # 读出原值
        orig = df_res.loc[ridx, comp_cols].astype(str).tolist()
        vals_in = [v.strip() for v in orig if v.strip()]
        vals_out = []
        for nm in vals_in:
            key = nm.lower()
            if key in ban_lower2:
                continue
            if key in alias_lower_map2:
                nm = alias_lower_map2[key]
                changed_cells += 1
                key = nm.lower()
            elif key in canon_lower2orig2:
                corrected = canon_lower2orig2[key]
                if corrected != nm:
                    changed_cells += 1
                nm = corrected
            vals_out.append(nm)
        # 同根去重 + 左移
        cleaned, seen = [], set()
        for nm in sorted(vals_out, key=len, reverse=True):
            k = _norm_key(nm)
            if any(k in kk or kk in k for kk in seen):
                continue
            cleaned.append(nm)
            seen.add(k)
        # 回写
        for i, col in enumerate(comp_cols):
            new_val = cleaned[i] if i < len(cleaned) else ""
            if str(df_res.at[ridx, col]) != new_val:
                changed_cells += 1
            df_res.at[ridx, col] = new_val

    # 再清一次列内重复
    df_res = dedup_company_cols(df_res)

    cute_box(
        f"已将最新映射应用到 result.csv（变更单元格约 {changed_cells} 个）",
        f"最新のマッピングを result.csv に適用しました（変更セル数 約 {changed_cells}）",
        "🛠️"
    )

    # 最后保存
    df_res.to_csv(res_f, index=False, encoding="utf-8-sig")

    cute_box(
        f"Step-3 完成，处理 {len(df_map)} 条映射，result.csv 已更新",
        f"Step-3 完了：{len(df_map)}件 処理済み，result.csv 更新完了",
        "🚀"
    )
    cute_box(
        f"本批次 Process ID：{process_id}",
        f"今回の Process ID：{process_id}",
        "📌"
    )
               
            
def step4():
    import pandas as _pd

    # 1) 读 CSV
    df = _pd.read_csv(BASE_DIR / "result.csv", dtype=str).fillna("")

    # 2) 准备输出行：注意这里给每一条都加上 value=1
    rows = []
    for _, r in tqdm(df.iterrows(), desc="生成邻接表", total=len(df)):
        comps = [r[f"company_{i}"] 
                 for i in range(1, MAX_COMP_COLS+1) 
                 if r[f"company_{i}"].strip()]
        for a, b in itertools.permutations(comps, 2):
            rows.append({
                "company_a": a,
                "company_b": b,
                "value": 1,
            })

    # 3) 构建完整 DataFrame
    out = _pd.DataFrame(rows)

    # 4) 写 adjacency list （只保留 a/b 两列）
    out[['company_a','company_b']].to_csv(
        BASE_DIR / "result_adjacency_list.csv",
        index=False, encoding="utf-8-sig"
    )
    cute_box(
        "Step4 已生成邻接表：result_adjacency_list.csv",
        "Step4 隣接リストを生成しました：result_adjacency_list.csv",
        "📋"
    )

    # ——— 生成带行列标题的 Pivot Table ———
    pivot = out.pivot_table(
        index="company_a",      # 行标签
        columns="company_b",    # 列标签
        values="value",         # 聚合字段
        aggfunc="sum",          # 把所有 value=1 累加
        fill_value=""           # 0 或 NaN 都显示空白
    )

    # 5) 导出带行/列标题的矩阵
    pivot.to_csv(
        BASE_DIR / "pivot_table.csv",
        encoding="utf-8-sig"
    )
    cute_box(
        "Step4 已生成透视表：pivot_table.csv",
        "Step4 ピボットテーブルを生成しました：pivot_table.csv",
        "📊"
    )
def main():
    mysql_url = ask_mysql_url()
    try:
        create_engine(mysql_url).connect().close()
        cute_box(
            "数据库连接成功！",
            "データベース接続 成功！",
            "🔗"
        )
    except Exception as e:
        cute_box(
            f"数据库连接失败：{e}",
            f"データベース接続 失敗：{e}",
            "❌"
        )
        sys.exit(1)

    choice = choose()

    if choice == "1":
        step1()
        step2(mysql_url)

        # —— 新增：跑完 Step-2 后等待用户指令 ——
        while True:
            nxt = input("👉 输入 2 继续 Step-3/4，或输入 e 退出 / 2 でStep-3/4を続行, e で終了: ").strip().lower()
            if nxt == "2":
                step3(mysql_url)
                step4()
                cute_box(
                "Step-3/4 全部完成！",
                "Step-3/4 全て完了しました！",
                "🎉"
                )
                break
            elif nxt == "e":
                cute_box(
                "已退出，拜拜～",
                "終了しました、またね！",
                "👋"
                )
                return
            else:
                cute_box(
                "无效输入，请再试一次！",
                "無効な入力です。もう一度入力してね！",
                "🔄"
                )

    else:   # choice == "2"
        step3(mysql_url)
        step4()
        cute_box(
        "Step-3/4 全部完成！",
        "Step-3/4 全て完了しました！",
         "🎉"
        )


if __name__ == "__main__":
    main()
