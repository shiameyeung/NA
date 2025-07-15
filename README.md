```markdown
# 初回セットアップ

## 1. NA_launcher.py をダウンロード

---

### 2. （Windowsのみ、macOSは不要）  
以下リンクからダウンロード＆インストール  
[vc_redist.x64.exe (Microsoft Visual C++ 再頒布可能パッケージ)](https://aka.ms/vs/17/release/vc_redist.x64.exe)

---

### 3. Python を用意（**推奨バージョン: 3.11.9**）

- **Windows:**  
  [python-3.11.9-amd64.exe](https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe)
- **macOS:**  
  [python-3.11.9-macos11.pkg](https://www.python.org/ftp/python/3.11.9/python-3.11.9-macos11.pkg)

> **インストール時は必ず「Add to PATH」にチェックを入れてください**

---

## 4. 環境設定

### 4.1 プロジェクトフォルダ作成  
**macOS（ターミナル）:**  
```bash
mkdir -p ~/NA_project && cd ~/NA_project
```
**Windows（PowerShell）:**  
```powershell
mkdir $env:USERPROFILE\NA_project; cd $env:USERPROFILE\NA_project
```

---

### 4.2 仮想環境の作成  
```bash
python -m venv NA_env
```

---

### 4.3 仮想環境の有効化

**macOS:**  
```bash
source NA_env/bin/activate
```
**Windows:**  
```powershell
.\NA_env\Scripts\Activate.ps1
```

---

### 4.4 パッケージ管理ツールのアップグレード  
```bash
pip install -U pip setuptools wheel
```

---

### 4.5 "python "（後ろにスペース）と入力

---

### 4.6  
**NA_launcher.py** を  
- macOSのターミナル  
- WindowsのPowerShell  
へドラッグ&ドロップし、**エンターキー**を押す

---

### 4.7  
最終の環境設定完了までお待ちください

---

# 毎回の使い方

---

1. 事前にダウンロードした `NA_launcher.py` を、処理したい `.DOCX` ファイルが入ったフォルダと同じ場所に移動。

2. **仮想環境の有効化**
   - macOS:
     ```bash
     source NA_env/bin/activate
     ```
   - Windows:
     ```powershell
     .\NA_env\Scripts\Activate.ps1
     ```

3. `"python "`（後ろにスペース）と入力

4. `NA_launcher.py` をターミナルやPowerShellにドラッグ&ドロップ、**エンターキー**を押す

5. **初回のみ**  
   事前にもらったキーコードをペースト、**エンターキー**を押す

6. **選択肢「1」を選ぶ**（`1`を入力、**エンターキー**）

7. 実行完了後、`NA_mapping_todo.csv` を開き、F列（canonical_name）を必要に応じて記入

    - 参考:  
        - Bad_rate が高いほど、偽企業名の確率が高い  
        - Advice は既存の正規企業名とマッチしたもの  
        - Advice ID はその正規企業名のID  
        - canonical_name表: 現有の正規企業名一覧

    - 入力ルール:  
        - 偽企業名          → `0`  
        - 既存の正規企業名がある  → canonical_name表、またはAdvice IDの数字  
        - 既存の正規企業名がない  → 新規正規企業名（英字・数字）

    - **数分ごとに必ず保存！**

8. 編集が終わったら保存・閉じる

---

### 8.1 続きの処理

- ターミナル/PowerShell に戻る
- まだ閉じていなければ、`2` を入力、**エンターキー**
- 閉じた場合は `"python "`（スペース付き）と入力、再び `NA_launcher.py` をドラッグ&ドロップ、**エンターキー**
- 選択肢「2」を選ぶ（`2`を入力、**エンターキー**）

---

# 🎉 大完成！結果を見よう！ 🎉

---

## 📧 サポート

ご不明点や助言がありましたら，下記の作成者までお気軽にお問い合わせください:

**作者 / Author / 作成者：楊 天楽 (Shiame Yeung)**  
**协助 / In cooperation with / 協力：李 宗昊 李 佳璇**

---

## 📧 技术支持

如有任何疑问或改进建议，欢迎联系：

**作者 / Author / 作成者：楊 天楽 (Shiame Yeung)**  
**协助 / In cooperation with / 協力：李 宗昊 李 佳璇**

```
