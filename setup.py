
# -*- coding: utf-8 -*-
"""
FairyRecom 视频推荐系统安装脚本
"""

from setuptools import setup, find_packages
import os

# 读取README文件
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 读取requirements.txt
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fairyrecom",
    version="1.0.0",
    description="基于多种算法的智能视频推荐系统",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SISUBEN/FairyRecom",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "fairyrecom-api=fairyrecom.api.recommendation_api:main",
            "fairyrecom-query=fairyrecom.analysis.query_tool:main",
        ],
    },
    include_package_data=True,
    package_data={
        "fairyrecom": ["data/*.db", "static/*"],
    },
    zip_safe=False,
)