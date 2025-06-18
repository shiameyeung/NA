#!/bin/bash

# 公共检查函数
check_success() {
  if [ $? -ne 0 ]; then
    echo "❌ 出现错误，安装中止。 / エラーが発生しました。インストールを中止します。"
    exit 1
  fi
}

echo "🚀 开始安装 Homebrew ... / Homebrew のインストールを開始します ..."
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
check_success

echo "✅ Homebrew 安装完成，开始安装 Python3 ... / Homebrew のインストールが完了しました。Python3 のインストールを開始します ..."
brew install python
check_success

echo "✅ Python3 安装完成，版本信息：/ Python3 のインストールが完了しました。バージョン情報："
python3 --version
pip3 --version

echo "🚀 安装 Python 第三方库：pandas tqdm python-docx spacy fuzzywuzzy ... / Python ライブラリをインストールします：pandas tqdm python-docx spacy fuzzywuzzy ..."
pip3 install --upgrade pandas tqdm python-docx spacy fuzzywuzzy
check_success

echo "✅ 第三方库安装完成 / ライブラリのインストールが完了しました"

echo "🚀 下载 Spacy 英文小模型 en_core_web_sm ... / Spacy の英語モデル en_core_web_sm をダウンロードします ..."
python3 -m spacy download en_core_web_sm
check_success

echo "🎉 环境安装全部完成！/ 環境のインストールが全て完了しました！"

echo "🔍 验证安装：导入模块测试中 ... / インストール確認：モジュールのインポートをテスト中 ..."
python3 -c "import pandas; import tqdm; import docx; import spacy; import fuzzywuzzy; print('✅ 所有模块导入成功！ / 全てのモジュールのインポートに成功しました！')"
check_success
