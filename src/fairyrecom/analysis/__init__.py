
# -*- coding: utf-8 -*-
"""
FairyRecom Analysis Module

数据分析模块，包含各种分析工具和算法。
"""

from .bigfive_analyzer import BigFiveSimilarityAnalyzer
from .query_tool import VideoAffinityQuery

__all__ = [
    'BigFiveSimilarityAnalyzer',
    'VideoAffinityQuery'
]