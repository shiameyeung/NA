## 初回
# NA_launcher.pyをダウンロードする。

# （Windowsのみ、macOSは不要）以下リンクをダウンロード、インストール
https://aka.ms/vs/17/release/vc_redist.x64.exe

# Pythonを用意
# Python 3.11.9
# Windows：
# https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe
# macOS：
# https://www.python.org/ftp/python/3.11.9/python-3.11.9-macos11.pkg
# インストールする時、 “Add to PATH”をチェックよう


# 環境設定１
# macOSのターミナルに、WindowsのPowerShellにペーストして実行する
# macOS：
mkdir -p ~/NA_project && cd ~/NA_project
# Windows:
mkdir %USERPROFILE%\NA_project && cd %USERPROFILE%\NA_project

# 環境設定2
# ペーストして実行する
python -m venv NA_env

# 環境設定3
# ペーストして実行する
# macOS:
source NA_env/bin/activate
# Windows:
.\NA_env\Scripts\Activate.ps1

# 環境設定4
# ペーストして実行する
pip install -U pip setuptools wheel

# "python "を入力（pythonの後ろはスペース）する

# NA_launcher.pyをmacOSのターミナルに、WindowsのPowerShellにドラッグする、エンターキーを押す

# 最終の環境設定完了まで待ち


## 毎回
# 事前にダウンロードしたNA_launcher.pyを、処理したいの.DOCXファイルを内蔵したフォルダの同層に移動。

# macOSのターミナルに、WindowsのPowerShellにペーストして実行する
# macOS:
source NA_env/bin/activate
# Windows:
.\NA_env\Scripts\Activate.ps1

# "python "を入力（pythonの後ろはスペース）する

# NA_launcher.pyをmacOSのターミナルに、WindowsのPowerShellにドラッグする、エンターキーを押す

# 事前にもらったキーコードをペースト（初回だけ）、エンターキーを押す

# 選択肢１を選ぶ（１を入力、エンターキーを押す）

# 実行完了、NA_mapping_todo.csvを開けて、F列（canonical_name）を適度に完成する
# 参考：
  同表
  Bad_rateが高いほど、偽企業名の確率が高い
  Adviceは現存の正規企業名とマッチしたもの
    Advice IDはその正規企業名のID
    
  canonical_name表
  現有の正規企業名一覧

# 入力：
  偽企業名              0
  現存の正規企業名がある   ID（canonical_name表　まだは　　Advice IDの数字）
  現存の正規企業名がない   設定したい新規正規企業名（英字・数字）

# 数分ごとで必ず保存することを！！！

# 最後、保存する、閉じる。

# macOSのターミナルに、WindowsのPowerShellに戻る
  先閉じなかったら、２を入力、エンターキーを押す
  
  閉じたら、"python "を入力（pythonの後ろはスペース）する
  NA_launcher.pyをmacOSのターミナルに、WindowsのPowerShellにドラッグする、エンターキーを押す
  選択肢２を選ぶ（２を入力、エンターキーを押す）

## 大完成！！！結果を見よう！！！



## 📧 サポート

ご不明点や助言がありましたら，下記の作成者までお気軽にお問い合わせください:

**作者 / Author / 作成者：楊 天楽 (Shiame Yeung)**

**协助 / In cooperation with / 協力：李 宗昊 李 佳璇**

## 📧 技术支持

如有任何疑问或改进建议，欢迎联系：

**作者 / Author / 作成者：楊 天楽 (Shiame Yeung)**

**协助 / In cooperation with / 協力：李 宗昊 李 佳璇**

