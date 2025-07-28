# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import sys
import os
from typing import Dict, List, Optional
import traceback
from datetime import datetime
import logging
from rich.logging import RichHandler
from rich.console import Console

# 配置日志
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        RichHandler(console=console, rich_tracebacks=True),
        # logging.FileHandler('fairyrecom_api.log', encoding='utf-8')   
    ]
)
logger = logging.getLogger(__name__)

# 导入推荐引擎
try:
    from ..core.recommendation_engine import VideoRecommendationEngine
    from ..core.affinity_simple import VideoAffinityAnalyzer
except ImportError:
    logger.error("[red]请确保推荐引擎模块存在[/red]")
    sys.exit(1)

import os
# 返回上级目录

print("當前目錄：", os.getcwd())
app = Flask(__name__, template_folder='templates')
CORS(app)  # 允许跨域请求

# 全局推荐引擎实例
recommendation_engine = None
sentiment_analyzer = None

def init_engine():
    """
    初始化推荐引擎
    """
    global recommendation_engine
    try:
        recommendation_engine = VideoRecommendationEngine()
        logger.info("✅ 推荐引擎初始化成功")
        return True
    except Exception as e:
        logger.error(f"❌ 推荐引擎初始化失败: {e}")
        return False

def init_sentiment_analyzer():
    """
    初始化情感分析器
    """
    global sentiment_analyzer
    try:
        sentiment_analyzer = VideoAffinityAnalyzer()
        logger.info("✅ 情感分析器初始化成功")
        return True
    except Exception as e:
        logger.error(f"❌ 情感分析器初始化失败: {e}")
        return False

def create_response(success: bool, data: any = None, message: str = "", error: str = "") -> Dict:
    """
    创建统一的API响应格式
    """
    response = {
        "success": success,
        "timestamp": datetime.now().isoformat(),
        "message": message
    }
    
    if success:
        response["data"] = data
    else:
        response["error"] = error
    
    return response

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    健康检查接口
    """
    if recommendation_engine is None:
        return jsonify(create_response(False, error="推荐引擎未初始化")), 500
    
    return jsonify(create_response(True, {
        "status": "healthy",
        "engine_status": "running",
        "total_users": len(recommendation_engine.user_item_matrix.index),
        "total_videos": len(recommendation_engine.user_item_matrix.columns)
    }, "推荐系统运行正常"))

@app.route('/api/recommend/user/<int:user_id>', methods=['GET'])
def recommend_for_user(user_id: int):
    """
    为指定用户生成推荐
    
    参数:
    - algorithm: 推荐算法 (collaborative_filtering, content_based, popularity, hybrid, diversified)
    - n: 推荐数量 (默认10)
    - weights: 混合推荐权重 (仅当algorithm=hybrid时有效)
    """
    try:
        # 获取参数
        algorithm = request.args.get('algorithm', 'hybrid')
        n = int(request.args.get('n', 10))
        weights_str = request.args.get('weights', None)
        
        # 解析权重参数
        weights = None
        if weights_str and algorithm == 'hybrid':
            try:
                weights = json.loads(weights_str)
            except json.JSONDecodeError:
                return jsonify(create_response(False, error="权重参数格式错误")), 400
        
        # 验证参数
        if n <= 0 or n > 100:
            return jsonify(create_response(False, error="推荐数量必须在1-100之间")), 400
        
        valid_algorithms = ['collaborative_filtering', 'content_based', 'popularity', 'hybrid', 'diversified', 'cold_start']
        if algorithm not in valid_algorithms:
            return jsonify(create_response(False, error=f"不支持的算法: {algorithm}")), 400
        
        # 生成推荐
        if algorithm == 'collaborative_filtering':
            recommendations = recommendation_engine.collaborative_filtering_recommend(user_id, n)
        elif algorithm == 'content_based':
            recommendations = recommendation_engine.content_based_recommend(user_id, n)
        elif algorithm == 'popularity':
            recommendations = recommendation_engine.popularity_based_recommend(user_id, n)
        elif algorithm == 'diversified':
            recommendations = recommendation_engine.diversified_recommend(user_id, n)
        elif algorithm == 'cold_start':
            recommendations = recommendation_engine.cold_start_recommend(user_id, n)
        else:  # hybrid
            recommendations = recommendation_engine.hybrid_recommend(user_id, n, weights)
        
        # 格式化结果
        result = {
            "user_id": user_id,
            "algorithm": algorithm,
            "recommendations": [
                {
                    "video_id": int(video_id),
                    "score": float(score),
                    "rank": i + 1
                }
                for i, (video_id, score) in enumerate(recommendations)
            ],
            "total_count": len(recommendations)
        }
        
        if weights:
            result["weights"] = weights
        
        return jsonify(create_response(True, result, f"成功为用户 {user_id} 生成 {len(recommendations)} 个推荐"))
        
    except Exception as e:
        return jsonify(create_response(False, error=f"推荐生成失败: {str(e)}")), 500
    
def recommend_for_custom_user(
    age: int,
    gender: int,
    education: int,
    hobby: List[str],
    address: int,
    income: float,
    career: int,
    user_id: int = "AUTO",
    algorithm: str = 'hybrid',
    n: int = 10,
    weights: Dict[str, float] = None
):
    """
    为自定用户生成推荐
    
    参数:
    - algorithm: 推荐算法 (collaborative_filtering, content_based, popularity, hybrid, diversified)
    - n: 推荐数量 (默认10)
    - weights: 混合推荐权重 (仅当algorithm=hybrid时有效)
    """
    # 先添加用戶
    recommendation_engine.add_user(age, gender, education, hobby, address, income, career, user_id)
    
    # 验证参数
    if n <= 0 or n > 100:
        raise ValueError("推荐数量必须在1-100之间")
    
    valid_algorithms = ['collaborative_filtering', 'content_based', 'popularity', 'hybrid', 'diversified', 'cold_start']
    if algorithm not in valid_algorithms:
        raise ValueError(f"不支持的算法: {algorithm}")
    
    # 生成推荐
    if algorithm == 'collaborative_filtering':
        recommendations = recommendation_engine.collaborative_filtering_recommend(user_id, n)
    elif algorithm == 'content_based':
        recommendations = recommendation_engine.content_based_recommend(user_id, n)
    elif algorithm == 'popularity':
        recommendations = recommendation_engine.popularity_based_recommend(user_id, n)
    elif algorithm == 'diversified':
        recommendations = recommendation_engine.diversified_recommend(user_id, n)
    elif algorithm == 'cold_start':
        recommendations = recommendation_engine.cold_start_recommend(user_id, n)
    else:  # hybrid
        recommendations = recommendation_engine.hybrid_recommend(user_id, n, weights)
    
    # 格式化结果并返回数据（不是jsonify响应）
    result = {
        "user_id": user_id,
        "algorithm": algorithm,
        "user_info": {
            "age": age,
            "gender": gender,
            "education": education,
            "hobby": hobby,
            "address": address,
            "income": income,
            "career": career
        },
        "recommendations": [
            {
                "video_id": int(video_id),
                "score": float(score),
                "rank": i + 1
            }
            for i, (video_id, score) in enumerate(recommendations)
        ],
        "total_count": len(recommendations)
    }
    
    return result  # Return data directly, not jsonify response

@app.route('/api/recommend/batch', methods=['POST'])
def batch_recommend():
    """
    批量推荐
    
    请求体:
    {
        "user_ids": [1, 2, 3],
        "algorithm": "hybrid",
        "n": 10
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'user_ids' not in data:
            return jsonify(create_response(False, error="缺少user_ids参数")), 400
        
        user_ids = data['user_ids']
        algorithm = data.get('algorithm', 'hybrid')
        n = data.get('n', 10)
        
        # 验证参数
        if not isinstance(user_ids, list) or len(user_ids) == 0:
            return jsonify(create_response(False, error="user_ids必须是非空列表")), 400
        
        if len(user_ids) > 100:
            return jsonify(create_response(False, error="批量推荐用户数量不能超过100")), 400
        
        # 批量推荐
        batch_results = recommendation_engine.batch_recommend(user_ids, n, algorithm)
        
        # 格式化结果
        result = {
            "algorithm": algorithm,
            "total_users": len(user_ids),
            "successful_users": len(batch_results),
            "results": {}
        }
        
        for user_id, recommendations in batch_results.items():
            result["results"][str(user_id)] = [
                {
                    "video_id": int(video_id),
                    "score": float(score),
                    "rank": i + 1
                }
                for i, (video_id, score) in enumerate(recommendations)
            ]
        
        return jsonify(create_response(True, result, f"成功为 {len(batch_results)} 个用户生成推荐"))
        
    except Exception as e:
        return jsonify(create_response(False, error=f"批量推荐失败: {str(e)}")), 500

@app.route('/api/similarity/users/<int:user_id>', methods=['GET'])
def find_similar_users(user_id: int):
    """
    查找相似用户
    
    参数:
    - method: 相似度计算方法 (cosine, pearson, jaccard)
    - n: 返回数量 (默认10)
    """
    try:
        method = request.args.get('method', 'cosine')
        n = int(request.args.get('n', 10))
        
        # 验证参数
        valid_methods = ['cosine', 'pearson', 'jaccard']
        if method not in valid_methods:
            return jsonify(create_response(False, error=f"不支持的相似度方法: {method}")), 400
        
        # 计算相似用户
        similar_users = recommendation_engine.calculate_user_similarity(user_id, method)
        
        # 格式化结果
        result = {
            "user_id": user_id,
            "method": method,
            "similar_users": [
                {
                    "user_id": int(similar_user_id),
                    "similarity": float(similarity),
                    "rank": i + 1
                }
                for i, (similar_user_id, similarity) in enumerate(similar_users.head(n).items())
            ],
            "total_count": len(similar_users)
        }
        
        return jsonify(create_response(True, result, f"找到 {len(similar_users)} 个相似用户"))
        
    except Exception as e:
        return jsonify(create_response(False, error=f"相似用户查找失败: {str(e)}")), 500

@app.route('/api/similarity/videos/<int:video_id>', methods=['GET'])
def find_similar_videos(video_id: int):
    """
    查找相似视频
    
    参数:
    - method: 相似度计算方法 (cosine, pearson)
    - n: 返回数量 (默认10)
    """
    try:
        method = request.args.get('method', 'cosine')
        n = int(request.args.get('n', 10))
        
        # 验证参数
        valid_methods = ['cosine', 'pearson']
        if method not in valid_methods:
            return jsonify(create_response(False, error=f"不支持的相似度方法: {method}")), 400
        
        # 计算相似视频
        similar_videos = recommendation_engine.calculate_item_similarity(video_id, method)
        
        # 格式化结果
        result = {
            "video_id": video_id,
            "method": method,
            "similar_videos": [
                {
                    "video_id": int(similar_video_id),
                    "similarity": float(similarity),
                    "rank": i + 1
                }
                for i, (similar_video_id, similarity) in enumerate(similar_videos.head(n).items())
            ],
            "total_count": len(similar_videos)
        }
        
        return jsonify(create_response(True, result, f"找到 {len(similar_videos)} 个相似视频"))
        
    except Exception as e:
        return jsonify(create_response(False, error=f"相似视频查找失败: {str(e)}")), 500

@app.route('/api/explain/<int:user_id>/<int:video_id>', methods=['GET'])
def explain_recommendation(user_id: int, video_id: int):
    """
    推荐解释
    """
    try:
        explanation = recommendation_engine.explain_recommendation(user_id, video_id)
        
        return jsonify(create_response(True, explanation, "推荐解释生成成功"))
        
    except Exception as e:
        return jsonify(create_response(False, error=f"推荐解释生成失败: {str(e)}")), 500

@app.route('/api/stats/user/<int:user_id>', methods=['GET'])
def get_user_stats(user_id: int):
    """
    获取用户统计信息
    """
    try:
        # 用户画像
        user_profile = {}
        if user_id in recommendation_engine.user_profiles.index:
            profile = recommendation_engine.user_profiles.loc[user_id]
            user_profile = {
                "total_videos": int(profile['total_videos']),
                "avg_affinity_score": float(profile['avg_affinity_score']),
                "avg_rating": float(profile['avg_rating']),
                "like_rate": float(profile['like_rate']),
                "avg_sentiment": float(profile['avg_sentiment'])
            }
        
        # 用户观看历史
        watched_videos = []
        if user_id in recommendation_engine.user_item_matrix.index:
            user_ratings = recommendation_engine.user_item_matrix.loc[user_id]
            watched_videos = [
                {
                    "video_id": int(video_id),
                    "affinity_score": float(score)
                }
                for video_id, score in user_ratings[user_ratings > 0].items()
            ]
            watched_videos.sort(key=lambda x: x['affinity_score'], reverse=True)
        
        result = {
            "user_id": user_id,
            "profile": user_profile,
            "watched_videos": watched_videos[:20],  # 只返回前20个
            "total_watched": len(watched_videos)
        }
        
        return jsonify(create_response(True, result, "用户统计信息获取成功"))
        
    except Exception as e:
        return jsonify(create_response(False, error=f"用户统计信息获取失败: {str(e)}")), 500

@app.route('/api/stats/video/<int:video_id>', methods=['GET'])
def get_video_stats(video_id: int):
    """
    获取视频统计信息
    """
    try:
        # 视频特征
        video_features = {}
        if video_id in recommendation_engine.item_features.index:
            features = recommendation_engine.item_features.loc[video_id]
            video_features = {
                "total_users": int(features['total_users']),
                "avg_affinity_score": float(features['avg_affinity_score']),
                "avg_rating": float(features['avg_rating']),
                "like_rate": float(features['like_rate']),
                "avg_sentiment": float(features['avg_sentiment'])
            }
        
        # 热门度排名
        popularity_rank = None
        if video_id in recommendation_engine.popularity_scores.index:
            popularity_rank = int(recommendation_engine.popularity_scores.rank(ascending=False)[video_id])
        
        # 观看用户
        viewers = []
        if video_id in recommendation_engine.user_item_matrix.columns:
            video_ratings = recommendation_engine.user_item_matrix[video_id]
            viewers = [
                {
                    "user_id": int(user_id),
                    "affinity_score": float(score)
                }
                for user_id, score in video_ratings[video_ratings > 0].items()
            ]
            viewers.sort(key=lambda x: x['affinity_score'], reverse=True)
        
        result = {
            "video_id": video_id,
            "features": video_features,
            "popularity_rank": popularity_rank,
            "viewers": viewers[:20],  # 只返回前20个
            "total_viewers": len(viewers)
        }
        
        return jsonify(create_response(True, result, "视频统计信息获取成功"))
        
    except Exception as e:
        return jsonify(create_response(False, error=f"视频统计信息获取失败: {str(e)}")), 500

@app.route('/api/stats/system', methods=['GET'])
def get_system_stats():
    """
    获取系统统计信息
    """
    try:
        result = {
            "total_users": len(recommendation_engine.user_item_matrix.index),
            "total_videos": len(recommendation_engine.user_item_matrix.columns),
            "total_interactions": int(recommendation_engine.user_item_matrix.sum().sum()),
            "avg_user_interactions": float(recommendation_engine.user_item_matrix.sum(axis=1).mean()),
            "avg_video_interactions": float(recommendation_engine.user_item_matrix.sum(axis=0).mean()),
            "sparsity": float(1 - (recommendation_engine.user_item_matrix > 0).sum().sum() / 
                             (len(recommendation_engine.user_item_matrix.index) * len(recommendation_engine.user_item_matrix.columns))),
            "top_videos": [
                {
                    "video_id": int(video_id),
                    "popularity_score": float(score)
                }
                for video_id, score in recommendation_engine.popularity_scores.nlargest(10).items()
            ],
            "most_active_users": [
                {
                    "user_id": int(user_id),
                    "interaction_count": int(count)
                }
                for user_id, count in recommendation_engine.user_item_matrix.sum(axis=1).nlargest(10).items()
            ]
        }
        
        return jsonify(create_response(True, result, "系统统计信息获取成功"))
        
    except Exception as e:
        return jsonify(create_response(False, error=f"系统统计信息获取失败: {str(e)}")), 500

@app.route('/api/sentiment/analyze', methods=['POST'])
def analyze_sentiment():
    """
    文本情感分析API
    
    请求体:
    {
        "text": "这个视频真的很棒！",
        "texts": ["文本1", "文本2"]  // 批量分析（可选）
    }
    
    响应体:
    {
        "success": true,
        "data": {
            "text": "这个视频真的很棒！",
            "sentiment_score": 0.8,
            "sentiment_label": "positive",
            "confidence": 0.6
        },
        "message": "情感分析完成"
    }
    """
    try:
        if sentiment_analyzer is None:
            return jsonify(create_response(False, error="情感分析器未初始化")), 500
        
        data = request.get_json()
        if not data:
            return jsonify(create_response(False, error="请求体不能为空")), 400
        
        # 单个文本分析
        if 'text' in data:
            text = data['text']
            if not text or not isinstance(text, str):
                return jsonify(create_response(False, error="文本内容不能为空")), 400
            
            sentiment_score = sentiment_analyzer.calculate_sentiment_score(text)
            
            # 判断情感倾向
            if sentiment_score > 0.6:
                sentiment_label = "positive"
            elif sentiment_score < 0.4:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
            
            result = {
                "text": text,
                "sentiment_score": float(sentiment_score),
                "sentiment_label": sentiment_label,
                "confidence": abs(sentiment_score - 0.5) * 2  # 置信度
            }
            
            return jsonify(create_response(True, result, "情感分析完成"))
        
        # 批量文本分析
        elif 'texts' in data:
            texts = data['texts']
            if not isinstance(texts, list) or len(texts) == 0:
                return jsonify(create_response(False, error="texts必须是非空列表")), 400
            
            if len(texts) > 100:
                return jsonify(create_response(False, error="批量分析文本数量不能超过100")), 400
            
            results = []
            for i, text in enumerate(texts):
                if not text or not isinstance(text, str):
                    continue
                
                sentiment_score = sentiment_analyzer.calculate_sentiment_score(text)
                
                # 判断情感倾向
                if sentiment_score > 0.6:
                    sentiment_label = "positive"
                elif sentiment_score < 0.4:
                    sentiment_label = "negative"
                else:
                    sentiment_label = "neutral"
                
                results.append({
                    "index": i,
                    "text": text,
                    "sentiment_score": float(sentiment_score),
                    "sentiment_label": sentiment_label,
                    "confidence": abs(sentiment_score - 0.5) * 2
                })
            
            result = {
                "total_texts": len(texts),
                "analyzed_texts": len(results),
                "results": results,
                "summary": {
                    "positive_count": len([r for r in results if r['sentiment_label'] == 'positive']),
                    "negative_count": len([r for r in results if r['sentiment_label'] == 'negative']),
                    "neutral_count": len([r for r in results if r['sentiment_label'] == 'neutral']),
                    "avg_sentiment_score": sum([r['sentiment_score'] for r in results]) / len(results) if results else 0
                }
            }
            
            return jsonify(create_response(True, result, f"批量情感分析完成，共分析{len(results)}条文本"))
        
        else:
            return jsonify(create_response(False, error="请提供text或texts参数")), 400
    
    except Exception as e:
        return jsonify(create_response(False, error=f"情感分析失败: {str(e)}"))

@app.route('/api/sentiment/batch_reviews', methods=['POST'])
def analyze_batch_reviews():
    """
    批量分析用户评论情感
    
    请求体:
    {
        "user_id": 123,
        "video_id": 456,
        "reviews": [
            {"review_id": 1, "content": "评论内容1"},
            {"review_id": 2, "content": "评论内容2"}
        ]
    }
    """
    try:
        if sentiment_analyzer is None:
            return jsonify(create_response(False, error="情感分析器未初始化")), 500
        
        data: dict = request.get_json()
        if not data or 'reviews' not in data:
            return jsonify(create_response(False, error="请提供reviews参数")), 400
        
        reviews: dict = data['reviews']
        user_id = data.get('user_id')
        video_id = data.get('video_id')
        
        if not isinstance(reviews, list) or len(reviews) == 0:
            return jsonify(create_response(False, error="reviews必须是非空列表")), 400
        
        results = []
        for review in reviews:
            if not isinstance(review, dict) or 'content' not in review:
                continue
            
            content = review['content']
            if not content:
                continue
            
            sentiment_score = sentiment_analyzer.calculate_sentiment_score(content)
            
            # 判断情感倾向
            if sentiment_score > 0.6:
                sentiment_label = "positive"
            elif sentiment_score < 0.4:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
            
            results.append({
                "review_id": review.get('review_id'),
                "content": content,
                "sentiment_score": float(sentiment_score),
                "sentiment_label": sentiment_label,
                "confidence": abs(sentiment_score - 0.5) * 2
            })
        
        # 计算统计信息
        positive_count = len([r for r in results if r['sentiment_label'] == 'positive'])
        negative_count = len([r for r in results if r['sentiment_label'] == 'negative'])
        neutral_count = len([r for r in results if r['sentiment_label'] == 'neutral'])
        avg_sentiment = sum([r['sentiment_score'] for r in results]) / len(results) if results else 0
        
        result = {
            "user_id": user_id,
            "video_id": video_id,
            "total_reviews": len(reviews),
            "analyzed_reviews": len(results),
            "results": results,
            "summary": {
                "positive_count": positive_count,
                "negative_count": negative_count,
                "neutral_count": neutral_count,
                "avg_sentiment_score": float(avg_sentiment),
                "overall_sentiment": "positive" if avg_sentiment > 0.6 else "negative" if avg_sentiment < 0.4 else "neutral"
            }
        }
        
        return jsonify(create_response(True, result, f"评论情感分析完成，共分析{len(results)}条评论"))
    
    except Exception as e:
        return jsonify(create_response(False, error=f"评论情感分析失败: {str(e)}"))
    
@app.route('/api/database/use_ext_db', methods=['POST'])
def use_external_database():
    """
    使用外部数据库
    """
    try:
        data = request.get_json()
        if 'db_path' not in data:
            return jsonify(create_response(False, error="数据库路径参数缺失")), 400
        
        db_path = data['db_path']
        if not os.path.exists(db_path):
            return jsonify(create_response(False, error="数据库文件不存在")), 400
        
        recommendation_engine.use_external_database(db_path)
        return jsonify(create_response(True, message="外部数据库已加载"))
        
    except Exception as e:
        return jsonify(create_response(False, error=f"外部数据库加载失败: {str(e)}")), 500

@app.route('/api/recom_form', methods=['POST'])
def recom_form():
    """
    通过填写表单获取推荐数据
    """
    try:
        import json
        req_data: dict = json.loads(request.data)
        
        if request.method == 'POST':
            age = int(req_data.get('age'))
            gender = int(req_data.get('gender'))
            education = int(req_data.get('education'))
            hobby = list(req_data.get('hobby'))
            address = int(req_data.get('address'))
            income = float(req_data.get('income'))
            career = int(req_data.get('career'))
            
            result = recommend_for_custom_user(
                age, gender, education, hobby, address, income, career
            )
            return jsonify(create_response(True, result, "推荐成功"))
        else:
            # 打印當前運行的目錄
            import os
            print("當前目錄：", os.getcwd())
            return render_template('index.html')
            
    except Exception as e:
        return jsonify(create_response(False, error=f"推荐表单生成失败: {str(e)}"))

@app.route('/api/user/find_or_create', methods=['POST'])
def find_or_create_user():
    """
    根据用户属性查找或创建用户
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify(create_response(False, error="缺少用户数据")), 400
        
        # 根据用户属性查找相似用户
        user_id = recommendation_engine.find_similar_user(data)
        
        result = {
            "user_id": user_id,
            "user_attributes": data,
            "is_new_user": user_id not in recommendation_engine.user_item_matrix.index
        }
        
        return jsonify(create_response(True, result, "用户查找/创建成功"))
        
    except Exception as e:
        return jsonify(create_response(False, error=f"用户查找/创建失败: {str(e)}"))

# @app.errorhandler(404)
# def not_found(error):
#     return jsonify(create_response(False, error="API接口不存在")), 404

# @app.errorhandler(405)
# def method_not_allowed(error):
#     return jsonify(create_response(False, error="HTTP方法不允许")), 405

# @app.errorhandler(500)
# def internal_error(error):
#     return jsonify(create_response(False, error="服务器内部错误")), 500

def create_app():
    """
    创建Flask应用实例
    """
    # 初始化推荐引擎
    if not init_engine():
        raise RuntimeError("推荐引擎初始化失败")
    
    # 初始化情感分析器
    if not init_sentiment_analyzer():
        raise RuntimeError("情感分析器初始化失败")
    
    return app

def main():
    """
    启动API服务器
    """
    logger.info("🚀 启动视频推荐系统API服务器...")
    
    # 初始化推荐引擎
    if not init_engine():
        logger.error("❌ 推荐引擎初始化失败，退出程序")
        return
    
    # 初始化情感分析器
    if not init_sentiment_analyzer():
        logger.error("❌ 情感分析器初始化失败，退出程序")
        return
    
    console.print("\n[bold cyan]📋 API接口列表:[/bold cyan]")
    console.print("[green]- GET  /api/health                           - 健康检查[/green]")
    console.print("[green]- GET  /api/recommend/user/<user_id>         - 用户推荐[/green]")
    console.print("[green]- POST /api/recommend/batch                  - 批量推荐[/green]")
    console.print("[green]- GET  /api/similarity/users/<user_id>       - 相似用户[/green]")
    console.print("[green]- GET  /api/similarity/videos/<video_id>     - 相似视频[/green]")
    console.print("[green]- GET  /api/explain/<user_id>/<video_id>     - 推荐解释[/green]")
    console.print("[green]- GET  /api/stats/user/<user_id>             - 用户统计[/green]")
    console.print("[green]- GET  /api/stats/video/<video_id>           - 视频统计[/green]")
    console.print("[green]- GET  /api/stats/system                     - 系统统计[/green]")
    console.print("[green]- POST /api/sentiment/analyze                - 文本情感分析[/green]")
    console.print("[green]- POST /api/sentiment/batch_reviews          - 批量评论情感分析[/green]")
    console.print("[green]- POST /api/recom_form                       - 推荐表单[/green]")
    
    console.print("\n[bold blue]🌐 服务器启动中...[/bold blue]")
    console.print("[cyan]访问地址: http://localhost:5000[/cyan]")
    console.print("[cyan]API文档: http://localhost:5000/api/health[/cyan]")
    console.print("\n[yellow]按 Ctrl+C 停止服务器[/yellow]")
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except KeyboardInterrupt:
        logger.info("\n[green]👋 服务器已停止[/green]")
    except Exception as e:
        logger.error(f"\n[red]❌ 服务器启动失败: {e}[/red]")
        
if __name__ == "__main__":
    main()