#!/usr/bin/env python3
# coding: utf-8

import sys
import subprocess

REQUIRED_PACKAGES = [
    "requests",
    "pandas",
    "tqdm",
    "sqlalchemy",
    "rapidfuzz",
    "python-docx",
    "spacy"
]

def install_package(pkg):
    print(f"📦 正在安装 {pkg} ...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])
    except subprocess.CalledProcessError as e:
        error_text = str(e)
        if (
            "Microsoft Visual C++" in error_text
            or "error: Microsoft Visual C++" in error_text
            or "unable to find vcvarsall.bat" in error_text
        ):
            print("❌ 检测到缺少 Microsoft Visual C++ Build Tools (C++编译工具)。")
            print("请先下载安装：")
            print("🔗 https://aka.ms/vs/16/release/vs_buildtools.exe")
            print("安装时请勾选 'C++ build tools' 组件。")
        else:
            print("❌ 安装失败，错误信息：")
            print(error_text)
        sys.exit(1)

def install_all():
    for pkg in REQUIRED_PACKAGES:
        import_name = pkg.replace("-", "_")
        try:
            __import__(import_name)
            print(f"✅ 已安装: {pkg}")
        except ImportError:
            install_package(pkg)

def ensure_spacy_model():
    import spacy
    try:
        spacy.load("en_core_web_sm")
        print("✅ spaCy模型已安装")
    except OSError:
        print("🔄 正在下载spaCy模型 'en_core_web_sm' ...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

def main():
    install_all()
    ensure_spacy_model()
    print("\n✅ 所有依赖安装完毕！")

if __name__ == "__main__":
    try:
        import requests
    except ImportError:
        print("📦 requests未安装，先安装requests...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    main()