# 🇨🇳 中文  
# 📘 数据处理脚本使用手册  

## 📌 功能概述  

本脚本集合（Step1～Step3）主要实现以下三项核心功能：

1. 从 Word 文档中提取正文，并按句拆分  
2. 自动识别句子中的公司名称  
3. 根据人工填写的映射表对公司名进行标准化处理  

支持的目录结构包括：  
- 最多两层子文件夹（如 年/月/文件.docx）  
- 仅一层文件夹  
- 无文件夹结构（文件直接放置）  
输出文件格式统一为 `.csv`。

---

## 🛠️ 环境准备  

### ✅ Python 环境  

建议使用 **Python 3.8 及以上版本**，推荐使用 [Visual Studio Code](https://code.visualstudio.com/) 管理和编辑开发环境。

### 📦 依赖库安装  

请在终端运行以下命令安装依赖：

```bash
pip install python-docx pandas openpyxl tqdm spacy fuzzywuzzy python-Levenshtein
python -m spacy download en_core_web_sm



⸻

📁 文件结构示例

项目文件夹/
├── NA_Step1_body_extract_V3.py         # Step1：关键词句子提取
├── NA_step2_company_recognizing_V9.py  # Step2：公司名抽取与映射表生成
├── NA_step3_standardize_V4.py          # Step3：标准化与清洗处理
├── *.docx                               # Word 原始文件（用于 Step1）
├── *.csv / *.xlsx                       # 数据文件（用于 Step2 和 Step3）



⸻

🧪 使用说明

Step 1：关键词句子提取

▶ 使用方法：

python NA_Step1_body_extract_V3.py

📋 处理逻辑：
	•	遍历所有 .docx 文件，提取 Body 和 Notes 之间的正文
	•	将正文按句子分割，匹配关键词根（如 partner, invest 等）
	•	输出 keyword_hit.csv，包含匹配到关键词的句子与文件结构信息：

字段名	说明
Tier_1	第一级文件夹（如年份）
Tier_2	第二级文件夹（如月份）
Filename	Word文件名
Sentence	句子内容
Hit_Count	匹配关键词数量
Matched_Keywords	命中的关键词根



⸻

Step 2：公司名识别与映射表生成

▶ 使用方法：

python NA_step2_company_recognizing_V9.py

📋 处理逻辑：
	•	使用 spaCy 模型及模糊匹配（fuzzywuzzy）识别公司名
	•	自动创建或更新以下文件：
	•	NA_company_list.csv：标准公司名与别名映射表
	•	NA_mapping.csv：识别后尚无法标准化的公司名清单（待人工填写）
	•	每个原始输入文件会输出两个新文件：
	•	*_recognized.csv：原句 + 提取的公司名（列如 Company_1, Company_2…）
	•	*_log.csv：记录每条被提取公司的原句与结果

⸻

✍️ Step 2 后需人工操作：填写映射表 NA_mapping.csv

为什么需要人工标注？

Step2 后生成的 NA_mapping.csv 中包含模型无法确定标准名的公司名，需要人工确认后 Step3 才能完成标准化。

文件结构说明：

列名	内容
NonStandard	被识别出的公司名（原文）
Standard	对应的标准公司名
Result	Step3 中处理结果备注

填写规则：

情况	操作
能对应某标准公司名	填写标准名（与 NA_company_list.csv 中保持一致）
明显不是公司名 / 识别错误	填写 0，表示禁止识别（banned）
留空	不允许，Step3 会提示 "Cannot be empty" 并跳过该项



⸻

Step 3：公司名标准化处理

▶ 使用方法：

python NA_step3_standardize_V4.py

📋 处理逻辑：
	•	根据 NA_mapping.csv 填写结果更新 NA_company_list.csv
	•	将 *_recognized.csv 中的公司名替换为标准公司名
	•	删除标记为 banned 的公司名
	•	更新后的 NA_mapping.csv 会添加 Result 列，标注处理状态

⸻

❓ 常见问题

问题提示	解决办法
AttributeError: Can only use .str accessor...	确保 Aliases 列没有缺失值，使用 .fillna('') 处理
UnicodeDecodeError	检查 CSV 编码，建议统一保存为 UTF-8
文件被覆盖	系统自动生成唯一文件名，如 _recognized_1.csv 等



⸻

💡 使用建议
	•	Step2 每次运行都会新建 NA_mapping.csv，请及时备份旧版
	•	Step3 前务必确认 NA_mapping.csv 全部填写完毕
	•	文件较多时可批量分批处理，减少内存占用
	•	可定期维护 NA_company_list.csv 以提升自动匹配效率

⸻

如有问题请联系原作者：杨天乐 / Shiame Yeung


# 日本語
# 📘 データ処理スクリプト使用ガイド

## ᾚ9 機能概要

本スクリプトセット (Step1\~Step3)は、下記の三つの基本機能を実装します:

1. 文書から本文を抽出し、文に分割
2. 文中の会社名を自動識別
3. 人力手入力によるマッピング情報に基づく標準化

サブフォルダ構成（最大 2 階層、または親フォルダのみでも反応）に対応します。結果は `.csv` として出力されます。

---

## 🛠️ 環境準備

### ✔️ Python 環境

**Python 3.8 以上** を推奨します。開発環境として [Visual Studio Code](https://code.visualstudio.com/) の使用を推奨します。

### ✔️ 依存ライブラリのインストール

ターミナルのコマンドラインにて、以下を実行してください：

```bash
pip install pandas openpyxl spacy fuzzywuzzy tqdm python-Levenshtein
python -m spacy download en_core_web_sm
```

---

## 📂 ファイル構成

処理対象ディレクトリには以下のファイルが含まれていること：

```
├── Word ファイル各端 (0~2階層 対応)
├── NA_Step1_body_extract_V3.py
├── NA_step2_company_recognizing_V9.py
├── NA_step3_standardize_V4.py
```

---

## 🚀 実行フロー

### ▫ Step 1: キーワード文抽出

**実行:**

```bash
python NA_Step1_body_extract_V3.py
```

**処理内容:**

* `.docx` 文書を全部探索
* `Body`から`Notes`までを抽出
* 文分割 & キーワードマッチ
* 結果は `keyword_hit.csv` に出力

### ▫ Step 2: 会社名識別 + 初期標準化

**実行:**

```bash
python NA_step2_company_recognizing_V9.py
```

**処理内容:**

* `spaCy`と`fuzzywuzzy`の組合せで会社名を抽出
* データベース: `NA_company_list.csv`を作成/更新
* 人力検証用の`NA_mapping.csv`を作成
* `banned`は認識されないように除外
* 結果: `_recognized.csv`, `_log.csv`, `NA_mapping.csv`

---

### 📁 Step2 実行後：NA\_mapping.csv の執筆

#### ‼️ 何故必要？

Step2 完了後、未確定な会社名の列表 (`NA_mapping.csv`)が自動生成されます。この列を人力で執筆することで、Step3 の正確な処理が可能となります。

#### 📋 NA\_mapping.csv の構成

| NonStandard | Standard |
| ----------- | -------- |
| Apple Inc.  | Apple    |
| Abcdef      | 0        |

#### ✍️ "Standard"列の執筆規則:

* 確定された会社名: `NA_company_list.csv` と一致する標準名を記述
* 誤識の場合: `0`と記入
* 空白 NG: 空のままの場合 Step3 は "Cannot be empty" と表示

---

### ▫ Step 3: 会社名の標準化

**実行:**

```bash
python NA_step3_standardize_V4.py
```

**処理内容:**

* 執筆済 `NA_mapping.csv`をもとに、`NA_company_list.csv`を更新
* `banned` 標記がある名前を削除
* 標準化した会社名に置換
* `NA_mapping.csv`に `Result` 列が追加され、処理結果が記録

---

## 📆 よくある質問

| 問題                   | 対処方法                  |
| -------------------- | --------------------- |
| `str accessor` エラー   | `.fillna('')` で空値を埋める |
| `UnicodeDecodeError` | 文字コードをUTF-8に統一        |
| ファイル互換               | 自動で重複を回避 `_1.csv` 等   |

---

## 🔖 前提知識

### ✅ Step1:文字抽出

* Word ドキュメントから Body\~Notes の本文を抽出
* 文の分割、キーワード確認
* 結果: `keyword_hit.csv`

### ✅ Step2:会社名認識

* spaCy + fuzzywuzzy を用い、文章中から会社名を抽出
* 非標準名のログを自動記録
* 結果: `_recognized.csv`, `NA_mapping.csv`

### ✅ Step3:標準化編集

* Step2 後の人力検証を元に、会社名を編集
* banned 削除 + 標準名に置換
* 結果: 更新済 `_recognized.csv`, `Result`列

---

## 📊 役立つポイント

* 継続的に正確度の高い会社名データベースを構築
* ついてに認識精度が高まり、人力検証の手間も減る
* ファイル名の重複も自動で避けられる

---

## 📧 サポート

ご不明点や助言がありましたら，下記の作成者までお気軽にお問い合わせください:

**原作者 / Author / 作成者：楊天楽 (Shiame Yeung)**

