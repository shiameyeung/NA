#!/usr/bin/env bash
set -euo pipefail

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

echo "🚀 安装 Python 第三方库 ... / Python ライブラリをインストールします ..."
pip3 install --upgrade \
  requests \
  pandas \
  tqdm \
  python-docx \
  spacy \
  rapidfuzz \
  sqlalchemy \
  pymysql \
  numpy \
  torch \
  sentence-transformers
check_success

echo "✅ 第三方库安装完成 / ライブラリのインストールが完了しました"

echo "🚀 下载 spaCy 英文小模型 en_core_web_sm ... / spaCy の英語モデル en_core_web_sm をダウンロードします ..."
python3 -m spacy download en_core_web_sm
check_success

echo "🎉 环境安装全部完成！/ 環境のインストールが全て完了しました！"

echo "🔍 验证安装：导入模块测试中 ... / インストール確認：モジュールのインポートをテスト中 ..."
python3 << 'EOF'
import requests;              print("requests", requests.__version__)
import pandas;               print("pandas", pandas.__version__)
import tqdm;                 print("tqdm", tqdm.__version__)
import docx;                 print("python-docx", docx.__version__)
import spacy;                print("spacy", spacy.__version__)
import rapidfuzz;            print("rapidfuzz", rapidfuzz.__version__)
import sqlalchemy;           print("sqlalchemy", sqlalchemy.__version__)
import pymysql;              print("pymysql OK")
import numpy;                print("numpy", numpy.__version__)
import torch;                print("torch", torch.__version__)
import sentence_transformers;print("sentence-transformers", sentence_transformers.__version__)
print("✅ 所有模块导入成功！ / 全てのモジュールのインポートに成功しました！")
EOF
check_success
