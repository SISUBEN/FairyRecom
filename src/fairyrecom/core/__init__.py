
# -*- coding: utf-8 -*-
"""
FairyRecom Core Module

核心推荐算法模块，包含各种推荐策略的实现。
"""

from .recommendation_engine import VideoRecommendationEngine
from .affinity_analyzer import VideoAffinityAnalyzer

__all__ = [
    'VideoRecommendationEngine',
    'VideoAffinityAnalyzer'
]