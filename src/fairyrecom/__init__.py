
# -*- coding: utf-8 -*-
"""
FairyRecom - 视频推荐系统

一个基于多种算法的智能视频推荐系统，包含协同过滤、内容推荐、
混合推荐等多种策略。

主要模块:
- core: 核心推荐算法
- api: RESTful API接口
- analysis: 数据分析工具
- demo: 演示程序
- utils: 工具函数
"""

__version__ = "1.0.0"
__author__ = "FairyRecom Team"
__email__ = "contact@fairyrecom.com"
__description__ = "一个基于多种算法的智能视频推荐系统"

# 导入核心类
from .core.recommendation_engine import VideoRecommendationEngine
from .core.affinity_analyzer import VideoAffinityAnalyzer
from .analysis.bigfive_analyzer import BigFiveSimilarityAnalyzer

__all__ = [
    'VideoRecommendationEngine',
    'VideoAffinityAnalyzer', 
    'BigFiveSimilarityAnalyzer'
]