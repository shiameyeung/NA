中文
数据处理脚本用户指南
文本抽取与公司名标准化自动流程

🧩 功能概述

本脚本集（Step1~Step3）实现以下三大功能：
	1.	从 Word 文件中抽取正文与句子
	2.	识别句中公司名称并初步标准化
	3.	根据人工填写的映射文件，完成公司名的清洗、标准化与剔除

支持无文件夹结构、单层文件夹、最多两层文件夹的通用目录结构，输出为 .csv 文件。


🛠️ 环境准备

✅ Python 环境

建议使用 Python 3.8 或以上版本。推荐使用 Visual Studio Code 作为脚本运行与环境管理工具。

✅ 安装依赖

请在终端执行以下命令：

pip install pandas openpyxl spacy fuzzywuzzy tqdm python-Levenshtein
python -m spacy download en_core_web_sm



📂 文件结构要求

处理目录中应包含以下内容：

├── Word 文件目录（支持 0~2 层子目录）
├── NA_Step1_body_extract_V3.py
├── NA_step2_company_recognizing_V9.py
├── NA_step3_standardize_V4.py




🚀 脚本使用步骤

🔹 Step 1：关键词句子提取（NA_Step1_body_extract_V3.py）

运行方式：

python NA_Step1_body_extract_V3.py

处理逻辑：
	•	自动遍历所有 .docx 文件
	•	提取从 Body 到 Notes 之间的正文内容
	•	进行分句 + 关键词根匹配（如 partner, merger 等）
	•	输出结果为 keyword_hit.csv，包含路径层级、命中关键词、命中句等信息



🔹 Step 2：公司名识别与初步标准化（NA_step2_company_recognizing_V9.py）

运行方式：

python NA_step2_company_recognizing_V9.py

处理逻辑：
	•	读取上一步输出的 keyword_hit.csv
	•	使用 spaCy 模型 + fuzzywuzzy 模糊匹配识别公司名
	•	自动过滤标记为 banned 的公司
	•	输出：
	•	_recognized.csv：句子 + 识别出的公司名
	•	_log.csv：处理过程记录
	•	NA_mapping.csv：自动生成待人工确认的非标准企业名
	•	NA_company_list.csv：维护当前标准名及别名的数据库


✍️ Step2 执行后，如何填写 NA_mapping.csv？

执行完 Step2 后，会生成 NA_mapping.csv 文件（列：NonStandard, Standard）。请根据以下规则人工填写：

情况	应该填写的 Standard
该名称是合法公司名	正确填写标准公司名（与 NA_company_list.csv 中一致）
该名称是错误识别	填写 0，表示“banned”
不确定是否为公司名	建议暂时填写 0（可后续修正）

注意：不可留空！ 留空行会在 Step3 中被标记为 Cannot be empty 而跳过。



✨ 标注建议
	•	请确保大小写、空格与 NA_company_list.csv 中保持一致
	•	中文/日文公司建议填写其英文通用名称（如无则填 0）
	•	不要直接编辑 _recognized.csv，所有标准化操作都由 Step3 处理



🔹 Step 3：标准化公司名并更新识别结果（NA_step3_standardize_V4.py）

运行方式：

python NA_step3_standardize_V4.py

处理逻辑：
	•	读取用户填写的 NA_mapping.csv
	•	更新公司标准名数据库 NA_company_list.csv
	•	清除识别结果中的 banned 公司名
	•	替换为用户填写的标准名称
	•	输出标准化后的 _recognized.csv
	•	NA_mapping.csv 中新增 Result 列，标注处理状态（如 Done, Cannot be empty 等）



🧩 常见问题与解决方案

问题	说明与解决办法
❗ Can only use .str accessor with string values!	检查 NA_company_list.csv 中是否存在空值，使用 .fillna('') 预处理
❗ UnicodeDecodeError	某些 CSV 文件编码格式异常，建议统一使用 UTF-8 保存
❗ 文件名被覆盖	脚本会自动生成 _recognized_1.csv, _2.csv 等防止覆盖



📝 处理逻辑总结

✅ Step1（文本句子提取）
	•	提取 Word 文件中的 Body 段落
	•	进行句子切分与关键词匹配
	•	输出所有命中句子的结构化结果 CSV

✅ Step2（公司名抽取与初步标准化）
	•	自动识别句中的企业名
	•	标注 banned、构建映射表与标准化表
	•	输出日志与待人工确认表格

✅ Step3（标准化公司名）
	•	根据用户标注标准化公司名
	•	删除 banned 企业名
	•	输出最终清洗过的公司名识别结果


🎯 工作机制优势
	•	✅ 完整支持从文本抽取到公司名标准化的自动流程
	•	✅ 可迭代式数据库构建，匹配越多越智能
	•	✅ 智能处理文件命名、避免覆盖
	•	✅ 所有处理过程透明、可追踪




 


📬 技术支持

如有任何疑问或改进建议，欢迎联系：

原作者 / Author / 作成者：杨天乐（Shiame Yeung）

日本語
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

