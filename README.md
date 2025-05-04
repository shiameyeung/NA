# 日本語（中文见后文）
# 📘 データ処理スクリプト使用ガイド

## 機能概要

今、実現した大体内容：
step1: ニュース文書から本文を抽出し、文に分割

step2: 文中の会社名を自動識別（データベースであったら、参照する）
	残り生データを人力手入力
		実際の会社名：標準企業名を入力
		会社名に該当しない：banする

step3: 人力手入力によるマッピング情報（標準会社名と擬似な会社名）に基づく生データを標準化、データベースにも保存

その過程で
一度入力したマッピング情報は自動的でデータベースファイルに保存（後続はサーバーに保存し共有も考慮する）
データベースは毎回で参照されながら、充実することが期待できる。

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
* banned 削除 、標準名があったら標準名に置換
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

# 中文（日本語は後ろで）
# 📘 数据处理脚本用户指南（文本抽取与公司名标准化自动流程）

## 🧩 功能概述

目前已实现的主要功能：

Step1： 从新闻稿文档中提取正文内容并进行分句处理。

Step2： 自动识别句子中的公司名称（如果已存在于数据库中，则直接引用）。
  对于剩余无法识别的名称，需由人工进行确认和填写：
  - 如果是实际存在的公司名称：请填写对应的标准企业名称
  - 如果不是公司名称：请标记为禁止（ban）

Step3： 根据人工填写的映射信息（标准公司名与类似名称的对应关系），将原始数据进行标准化，并将新信息同步保存到企业名称数据库中。

在整个过程中：
每次用户填写的映射信息都会自动保存进数据库文件；
后续运行将会自动读取该数据库进行比对和识别；
企业名数据库将随着使用逐步完善，并可在未来考虑上传至服务器共享使用。

支持无文件夹结构、单层文件夹、最多两层文件夹的通用目录结构，输出为 .csv 文件。

---

## 🛠️ 环境准备

### ✅ Python 环境

建议使用 Python 3.8 或以上版本。推荐使用 Visual Studio Code 作为脚本运行与环境管理工具。

### ✅ 安装依赖

请在终端执行以下命令：

```bash
pip install pandas openpyxl spacy fuzzywuzzy tqdm python-Levenshtein
python -m spacy download en_core_web_sm
```

---

## 📂 文件结构要求

处理目录中应包含以下内容：

```
├── Word 文件目录（支持 0~2 层子目录）
├── NA_Step1_body_extract_V3.py
├── NA_step2_company_recognizing_V9.py
├── NA_step3_standardize_V4.py
```

---

## 🚀 脚本使用步骤

### 🔹 Step 1：关键词句子提取（NA\_Step1\_body\_extract\_V3.py）

**运行方式：**

```bash
python NA_Step1_body_extract_V3.py
```

**处理逻辑：**

* 自动遍历所有 .docx 文件
* 提取从 Body 到 Notes 之间的正文内容
* 进行分句 + 关键词根匹配（如 partner, merger 等）
* 输出结果为 keyword\_hit.csv，包含路径层级、命中关键词、命中句等信息

---

### 🔹 Step 2：公司名识别与初步标准化（NA\_step2\_company\_recognizing\_V9.py）

**运行方式：**

```bash
python NA_step2_company_recognizing_V9.py
```

**处理逻辑：**

* 读取上一步输出的 keyword\_hit.csv
* 使用 spaCy 模型 + fuzzywuzzy 模糊匹配识别公司名
* 自动过滤标记为 banned 的公司
* 输出：

  * \_recognized.csv：句子 + 识别出的公司名
  * \_log.csv：处理过程记录
  * NA\_mapping.csv：自动生成待人工确认的非标准企业名
  * NA\_company\_list.csv：维护当前标准名及别名的数据库

---

### ✍️ Step2 执行后，如何填写 NA\_mapping.csv？

执行完 Step2 后，会生成 NA\_mapping.csv 文件（列：NonStandard, Standard）。请根据以下规则人工填写：

| 情况        | 应该填写的 Standard                         |
| --------- | -------------------------------------- |
| 该名称是合法公司名 | 正确填写标准公司名（与 NA\_company\_list.csv 中一致） |
| 该名称是错误识别  | 填写 0，表示“banned”                        |
| 不确定是否为公司名 | 建议暂时填写 0（可后续修正）                        |

注意：不可留空！ 留空行会在 Step3 中被标记为 Cannot be empty 而跳过。

---

### ✨ 标注建议

* 请确保大小写、空格与 NA\_company\_list.csv 中保持一致
* 中文/日文公司建议填写其英文通用名称（如无则填 0）
* 不要直接编辑 \_recognized.csv，所有标准化操作都由 Step3 处理

---

### 🔹 Step 3：标准化公司名并更新识别结果（NA\_step3\_standardize\_V4.py）

**运行方式：**

```bash
python NA_step3_standardize_V4.py
```

**处理逻辑：**

* 读取用户填写的 NA\_mapping.csv
* 更新公司标准名数据库 NA\_company\_list.csv
* 清除识别结果中的 banned 公司名
* 替换为用户填写的标准名称
* 输出标准化后的 \_recognized.csv
* NA\_mapping.csv 中新增 Result 列，标注处理状态（如 Done, Cannot be empty 等）

---

## 🧩 常见问题与解决方案

| 问题                                               | 说明与解决办法                                             |
| ------------------------------------------------ | --------------------------------------------------- |
| ⚠ Can only use .str accessor with string values! | 检查 NA\_company\_list.csv 中是否存在空值，使用 .fillna('') 预处理 |
| ⚠ UnicodeDecodeError                             | 某些 CSV 文件编码格式异常，建议统一使用 UTF-8 保存                     |
| ⚠ 文件名被覆盖                                         | 脚本会自动生成 \_recognized\_1.csv, \_2.csv 等防止覆盖          |

---

## 📝 处理逻辑总结

### ✅ Step1（文本句子提取）

* 提取 Word 文件中的 Body 段落
* 进行句子切分与关键词匹配
* 输出所有命中句子的结构化结果 CSV

### ✅ Step2（公司名抽取与初步标准化）

* 自动识别句中的企业名
* 标注 banned、构建映射表与标准化表
* 输出日志与待人工确认表格

### ✅ Step3（标准化公司名）

* 根据用户标注标准化公司名
* 删除 banned 企业名
* 输出最终清洗过的公司名识别结果

---

## 🌟 工作机制优势

* ✅ 完整支持从文本抽取到公司名标准化的自动流程

* ✅ 可进化式数据库构建，匹配越多越智能

* ✅ 智能处理文件命名、避免覆盖

* ✅ 所有处理过程透明、可追踪

📬 技术支持

如有任何疑问或改进建议，欢迎联系：

### **原作者 / Author / 作成者：楊　天楽（Shiame Yeung）**



