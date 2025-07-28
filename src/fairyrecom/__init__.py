# -*- coding: utf-8 -*-
__version__ = "1.0.0"
__author__ = "SISUBENY"
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