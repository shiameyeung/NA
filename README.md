数据处理脚本用户指南


环境准备

Python 环境
建议使用 Python 3.8 或更新版本。
推荐使用 Microsoft Vision Studio Code管理 Python 环境。

必需库
使用前，请在终端或命令行中运行以下命令安装依赖库：
pip install pandas openpyxl spacy fuzzywuzzy tqdm python-Levenshtein
python -m spacy download en_core_web_sm

文件结构
处理目录内应包含：
原始数据文件（.xlsx 或 .csv 格式）
处理脚本（共三个）：
NA_Step1_body_extract.py
NA_step2_company_recognizing_V9.py
NA_step3_standardize_V4.py

步骤说明与使用方法
Step 1: 关键词句子提取
使用方法
在命令行中执行：
python NA_Step1_body_extract.py
处理逻辑
遍历目录内所有 .xlsx 和 .csv 文件。
根据预定义的关键词，从文件中提取包含关键词的句子，并生成新的文件。
输出文件为 _hit_count.xlsx。
Step 2: 企业名识别与初步标准化
使用方法
在命令行中执行：
python NA_step2_company_recognizing_V9.py
处理逻辑
使用 SpaCy 和模糊匹配（fuzzy matching）方法识别并提取企业名。
创建或更新 NA_company_list.csv 和 NA_mapping.csv 文件。
_recognized.csv 文件保存识别的企业名列表，_log.csv 文件记录处理详情。
banned 标签标记不应被识别的企业名称。

step2 后的人工标注说明（填写 NA_mapping.csv）

📌 为什么需要人工标注？

在执行完 step2 (NA_step2_company_recognizing) 之后，会自动生成名为 NA_mapping.csv 的映射文件，其中包括脚本无法自动确定标准企业名的记录。
您需要人工填写这些记录，以便后续 step3 (NA_step3_standardize) 能够正确地标准化公司名称。
📂 映射文件结构介绍 (NA_mapping.csv)
映射文件共有 3个列：

🖊️ 如何填写“Standard”列？
您需要逐行检查并填写 Standard 列，规则如下：
1.	如果该非标准企业名能够明确对应到某个标准企业名：
请准确填写对应的标准企业名（与NA_company_list.csv中的标准名保持一致）
2.	如果该非标准企业名不是企业名或明显是错误识别：
请在 Standard 列内填写数字：0
3.	不得留空：
所有 Standard 列内均需要填写内容（留空则 step3 会标注为 "Cannot be empty" 并跳过处理）

📑 注意事项与填写建议
	•	注意大小写与拼写：
填入的标准企业名应与已有的标准企业名列表（NA_company_list.csv）中对应名称保持完全一致，包括大小写、拼写、空格。
	•	不确定的企业名：
如遇无法确定是否属于企业名，建议先填写0（标记为 banned），之后如需重新识别，可以修改NA_company_list.csv中的标记后再次运行 step2。
	•	中文、日语或非英文企业名的处理：
如果有出现中文、日语或其他语言的企业名，请填写其官方英文名为标准名。如果不存在通用的英文名，建议填写数字0，标记为 banned。

Step 3: 企业名标准化
使用方法
用户首先需人工完善 NA_mapping.csv 后执行：
python NA_step3_standardize_V4.py
处理逻辑
根据用户人工标注的 NA_mapping.csv 更新 NA_company_list.csv。
标记为banned的企业名从识别结果中移除。
标准化识别的企业名，更新 _recognized.csv 文件。
更新后的NA_mapping.csv会记录处理结果。
常见问题与解决方案
问题1: "Can only use .str accessor with string values!"
解决方案：
确保NA_company_list.csv中Aliases列无空值，可使用.fillna('')填充。
问题2: "UnicodeDecodeError"
解决方案：
检查CSV文件编码，建议统一使用UTF-8编码，或使用Excel另存为UTF-8 CSV。
问题3: 文件名冲突或覆盖
解决方案：
脚本会自动创建不重复的文件名，如_recognized_1.csv，无需手动操作。
注意事项
每次运行Step 2，NA_mapping.csv将重新创建，请注意备份。
Step 3前，务必人工校对并完善NA_mapping.csv。
性能优化与建议
处理大量文件时，建议逐批处理，避免单次处理文件过多，导致内存占用过高。
定期检查并清理已处理文件，以保持目录整洁，提升处理效率。

Step1 (NA_Step1_body_extract_V3.py)
此脚本遍历指定目录（最多支持两层文件夹结构或单层及无文件夹结构），读取所有Word文件（.docx），从文档中提取位于Body和Notes之间的正文内容，并进行分句处理。然后对每个句子进行关键词根匹配，统计关键词出现次数，生成包含文件路径、句子、命中关键词和命中数的CSV文件，供后续步骤使用。

Step2 (NA_step2_company_recognizing_V9.py)
此脚本读取Step1输出的CSV文件，利用自然语言处理工具（spaCy）及模糊匹配（fuzzywuzzy）自动识别文本中的公司名。同时维护和自动更新公司名称标准化映射表（NA_mapping.csv），供用户人工确认和修正。输出带有识别公司名结果的CSV文件。

Step3 (NA_step3_standardize_V4.py)
最后一步，此脚本读取用户人工完善后的公司名称映射表（NA_mapping.csv），更新公司标准名列表（NA_company_list.csv），并据此对Step2输出的识别公司名进行标准化。标准化过程中，会删除标记为“banned”的公司名，同时将识别到的非标准公司名统一替换为用户确认后的标准公司名。

目前覆盖的功能从文本抽出、社名抽出、社名标准化（匹配、剔除）
美中不足的是，这套体系需要加入人工操作。尤其前期匹配数据库空荡荡的情况。
我引入了一个可迭代的数据库，后期匹配数据库全了，大部分都能自动匹配，就不需要人工操作了。


如有其他问题，请联系Shiame以获得更多技术支持。


