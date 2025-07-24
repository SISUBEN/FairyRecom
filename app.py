# -*- coding: utf-8 -*-
"""
视频推荐系统API启动脚本
"""

import sys
import os

# 添加src目录到Python路径
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_path)

from fairyrecom.api.recommendation_api import create_app

# 创建 Flask 应用实例
app = create_app()

if __name__ == "__main__":
    # 禁用调试模式和热加载
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)