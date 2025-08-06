# 初回
## NA_launcher.pyをダウンロードする。
https://github.com/shiameyeung/NA/blob/main/NA_launcher.py

## （Windowsのみ、macOSは不要）以下リンクからダウンロード・インストール  
https://aka.ms/vs/17/release/vc_redist.x64.exe

## Pythonを用意  
Python 3.11.9  
**Windows：**  
https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe  
**macOS：**  
https://www.python.org/ftp/python/3.11.9/python-3.11.9-macos11.pkg

> インストール時は「Add to PATH」に必ずチェックを入れてください

## 環境設定１：プロジェクトフォルダ作成

**macOS（ターミナル）**  
```bash
mkdir -p ~/NA_project && cd ~/NA_project
```
**Windows（PowerShell）**  
```powershell
mkdir %USERPROFILE%\NA_project
cd %USERPROFILE%\NA_project
```

## 環境設定２：仮想環境の作成
```bash
python -m venv NA_env
```

## 環境設定３：仮想環境を有効化

- **macOS：**
  ```bash
  source NA_env/bin/activate
  ```
- **Windows：**
  ```powershell
  %USERPROFILE%\NA_project> NA_env\Scripts\activate
  ```

## 環境設定４：必要パッケージのインストール
```bash
pip install -U pip setuptools wheel
```

## ランチャー起動手順
1. ターミナル（またはPowerShell）で `python `（後ろにスペース）と入力  
2. `NA_launcher.py` をドラッグ＆ドロップし、Enterキーを押す  
3. 完了するまで待ちます

---

# 毎回の実行手順

1. `NA_launcher.py` と対象の `.DOCX` ファイルを同じフォルダに置く  
2. 仮想環境を有効化  
   - macOS: `source NA_env/bin/activate`  
   - Windows: `.\NA_env\Scripts\Activate.ps1`  
3. ターミナルで `python `（後ろにスペース）と入力  
4. `NA_launcher.py` をドラッグ＆ドロップしてEnter  
5. **初回のみ**：キーコードを貼り付けてEnter  
6. 「1」と入力してEnter  
7. 完了後、`NA_mapping_todo.csv` を開き、F列（canonical_name）を記入  
   - **参考**  
     - Bad_rate：値が高いほど、偽企業名の可能性高い  
     - Advice：既存の正規企業名とマッチした候補  
     - Advice ID：正規企業名のID
     - canonical_name表で現存の正規企業名一覧を参考
   - **入力ルール**  
     - 偽企業名 → `0`  
     - 既存の正規企業名 → ID（canonical_name表またはAdvice ID）  
     - 新規の正規企業名 → 英数字  
   - **数分ごとに必ず保存！**

8. 保存してCSVを閉じる  
9. ターミナルに戻り  
   - プログラムがまだ動いていれば「2」と入力→Enter  
   - すでに終了していれば、再度 `python ` → ドラッグ＆ドロップ → 「2」と入力→Enter

---

# 大完成！結果を見ましょう 🎉

---

## 📧 サポート

ご不明点やご意見は以下までお問い合わせください：  
作者：楊 天楽 (Shiame Yeung)  
協力：李 宗昊、李 佳璇
