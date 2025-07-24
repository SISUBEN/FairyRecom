
# -*- coding: utf-8 -*-
"""
FairyRecom API Module

RESTful API接口模块，提供HTTP服务。
"""

from .recommendation_api import create_app

__all__ = ['create_app']