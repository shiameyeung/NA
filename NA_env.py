#!/usr/bin/env python3
# coding: utf-8

import sys
import subprocess

# pip 包 → import 时的名字映射
IMPORT_MAPPING = {
    "python-docx": "docx",
    "sentence-transformers": "sentence_transformers",
    "pymysql": "pymysql"
}

# 完整依赖列表
REQUIRED_PACKAGES = [
    "requests",
    "pandas",
    "tqdm",
    "sqlalchemy",
    "pymysql",
    "rapidfuzz",
    "python-docx",
    "spacy",
    "numpy",
    "torch",
    "sentence-transformers"
]


def install_package(pkg: str):
    print(f"📦 正在安装 {pkg} ...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", pkg])
    except subprocess.CalledProcessError as e:
        err = e.output.decode(errors='ignore') if hasattr(e, 'output') else str(e)
        # Windows 下常见的 C++ 编译器缺失错误
        if any(msg in err for msg in [
            "Microsoft Visual C++", "unable to find vcvarsall.bat",
            "error: Microsoft Visual C++"
        ]):
            print("❌ 检测到缺少 Microsoft Visual C++ Build Tools (C++ 编译工具)。")
            print("   请先下载安装：")
            print("   🔗 https://aka.ms/vs/16/release/vs_buildtools.exe")
            print("   安装时请勾选 “C++ build tools” 组件。")
        else:
            print(f"❌ 安装失败：\n{err}")
        sys.exit(1)


def install_all():
    for pkg in REQUIRED_PACKAGES:
        imp_name = IMPORT_MAPPING.get(pkg, pkg.replace('-', '_'))
        try:
            __import__(imp_name)
            print(f"✅ 已安装: {pkg}")
        except ImportError:
            install_package(pkg)


def ensure_spacy_model():
    import spacy
    try:
        spacy.load("en_core_web_sm")
        print("✅ spaCy 模型 en_core_web_sm 已安装")
    except OSError:
        print("🔄 下载 spaCy 模型 en_core_web_sm ...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"]).


def main():
    # requests 用于脚本自身，如果没装先装它
    try:
        import requests
    except ImportError:
        print("📦 requests 未安装，先安装 requests ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    install_all()
    ensure_spacy_model()
    print("\n🎉 所有依赖安装完毕！")


if __name__ == "__main__":
    main()
