# 依赖 / Dependencies / 依存関係：python-docx, pandas, openpyxl, unicodedata, tqdm, spacy, fuzzywuzzy, python-Levenshtein
# 安装 / Install / インストール：pip install python-docx pandas openpyxl tqdm spacy fuzzywuzzy python-Levenshtein
# 模型下载 / Model Download / モデルダウンロード：python -m spacy download en_core_web_sm
# 原作者：杨天乐@关西大学 / Author: Shiame Yeung@Kansai University / 作成者：楊　天楽@関西大学

import os
import re
import time
import pandas as pd
from tqdm import tqdm

"""
NA_step3_standardize.py

1. Read NA_mapping.csv
2. Compare each row’s Standard value against NA_company_list.csv Standard_Name
3. If not matched and Standard == '0', add new row: Standard_Name=NonStandard, Aliases=banned
4. If not matched and Standard != '0', add new row: Standard_Name=Standard, Aliases=NonStandard
5. If matched, append NonStandard to existing Aliases
6. Standardize all *_recognized.csv files:
   6.1 Remove banned companies and shift remaining
   6.2 Replace aliases with standard names

- Progress bars for mapping and file standardization
- Adds a "Result" column in NA_mapping.csv with English notes
"""


def load_csv_with_replace(path):
    """
    Try reading CSV with utf-8 replacing errors, fallback to latin1.
    """
    try:
        with open(path, encoding='utf-8', errors='replace') as f:
            return pd.read_csv(f, dtype=str).fillna('')
    except Exception:
        return pd.read_csv(path, dtype=str, encoding='latin1').fillna('')


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    map_path = os.path.join(script_dir, 'NA_mapping.csv')
    company_list_path = os.path.join(script_dir, 'NA_company_list.csv')

    # 1. Load mapping and initialize Result column
    mapping_df = load_csv_with_replace(map_path)
    mapping_df['Result'] = ''

    # 2. Load company list
    if os.path.exists(company_list_path):
        comp_df = load_csv_with_replace(company_list_path)
    else:
        raise FileNotFoundError(f"{company_list_path} not found")

    std_names = set(comp_df['Standard_Name'])

    # 3. Process mapping entries
    for idx, row in tqdm(mapping_df.iterrows(), total=len(mapping_df), desc='🔄 Processing mapping', unit='row'):
        nonstd = row['NonStandard'].strip()
        std = row['Standard'].strip()
        try:
            if std == '':
                mapping_df.at[idx, 'Result'] = 'No input'
            elif std in std_names:
                comp_idx = comp_df[comp_df['Standard_Name'] == std].index[0]
                aliases = comp_df.at[comp_idx, 'Aliases']
                alias_list = [a.strip() for a in aliases.split(',') if a.strip()]
                if nonstd and nonstd not in alias_list:
                    alias_list.append(nonstd)
                    comp_df.at[comp_idx, 'Aliases'] = ', '.join(alias_list)
                mapping_df.at[idx, 'Result'] = 'Appended'
            else:
                if std == '0':
                    new_row = {'Standard_Name': nonstd, 'Aliases': 'banned'}
                else:
                    new_row = {'Standard_Name': std, 'Aliases': nonstd}
                comp_df = pd.concat([comp_df, pd.DataFrame([new_row])], ignore_index=True)
                std_names.add(new_row['Standard_Name'])
                mapping_df.at[idx, 'Result'] = 'Added'
        except Exception as e:
            mapping_df.at[idx, 'Result'] = f'Error: {e}'

    # Save updated company list and mapping
    comp_df.to_csv(company_list_path, index=False, encoding='utf-8')
    mapping_df.to_csv(map_path, index=False, encoding='utf-8')
    print(f"✔ Updated company list: {company_list_path}")
    print(f"✔ Updated mapping file: {map_path}")

    # Build alias map and banned set
    banned = set(comp_df.loc[comp_df['Aliases'].astype(str).str.strip().str.lower() == 'banned', 'Standard_Name'])
    alias_map = {}
    for _, r in comp_df.iterrows():
        std = r['Standard_Name']
        for a in [x.strip() for x in str(r['Aliases']).split(',') if x.strip() and x.strip().lower() != 'banned']:
            alias_map[a] = std
    for std in std_names - banned:
        alias_map[std] = std

    # 4. Standardize recognized files
    for fname in tqdm(os.listdir(script_dir), desc='📄 Standardizing files', unit='file'):
        if not fname.endswith('_recognized.csv'):
            continue
        path = os.path.join(script_dir, fname)
        df = load_csv_with_replace(path)
        comp_cols = [c for c in df.columns if c.startswith('Company_')]
        for idx, row in df.iterrows():
            comps = []
            for c in comp_cols:
                val = row[c].strip()
                if not val or val.lower() == 'nan' or val in banned:
                    continue
                comps.append(alias_map.get(val, val))
            for i, c in enumerate(comp_cols):
                df.at[idx, c] = comps[i] if i < len(comps) else ''
        df.to_csv(path, index=False, encoding='utf-8')
        print(f"Standardized {fname}")

if __name__ == '__main__':
    main()
