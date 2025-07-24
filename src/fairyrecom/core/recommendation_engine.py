# -*- coding: utf-8 -*-
"""
视频推荐算法引擎

这个模块实现了多种视频推荐算法，包括：
1. 协同过滤推荐 (Collaborative Filtering)
2. 内容过滤推荐 (Content-based Filtering)
3. 混合推荐算法 (Hybrid Recommendation)
4. 热门推荐 (Popularity-based)
5. 多样性推荐 (Diversity-aware)
"""

import sqlite3
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import json
import math
from datetime import datetime, timedelta
import warnings
import logging
from rich.logging import RichHandler
from rich.console import Console

warnings.filterwarnings("ignore")

# 配置Rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger(__name__)
console = Console()


class VideoRecommendationEngine:
    """
    视频推荐算法引擎

    支持多种推荐策略和算法组合
    """

    def __init__(self, db_path: str = "data.db"):
        """
        初始化推荐引擎

        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self.user_item_matrix = None
        self.item_features = None
        self.user_profiles = None
        self.popularity_scores = None

        # 推荐参数
        self.min_interactions = 5  # 最少交互次数
        self.similarity_threshold = 0.1  # 相似度阈值
        self.diversity_factor = 0.3  # 多样性因子

        # 加载数据
        self._load_data()

    def _load_data(self, db_path: Optional[str] = None):
        """
        从数据库加载推荐所需的数据
        """
        if db_path is None:
            db_path = self.db_path
        """
        从数据库加载推荐所需的数据
        """
        conn = sqlite3.connect(db_path)

        # 加载用户-视频交互矩阵
        query = """
            SELECT user_id, video_id, affinity_score
            FROM user_video_affinity
            ORDER BY user_id, video_id
        """
        interactions_df = pd.read_sql_query(query, conn)

        # 创建用户-物品矩阵
        self.user_item_matrix = interactions_df.pivot(
            index="user_id", columns="video_id", values="affinity_score"
        ).fillna(0)

        # 加载视频特征
        video_stats_query = """
            SELECT video_id, total_users, avg_affinity_score, 
                   avg_rating, like_rate, avg_sentiment
            FROM video_statistics
        """
        self.item_features = pd.read_sql_query(video_stats_query, conn)
        self.item_features.set_index("video_id", inplace=True)

        # 加载用户画像
        user_stats_query = """
            SELECT user_id, total_videos, avg_affinity_score,
                   avg_rating, like_rate, avg_sentiment
            FROM user_statistics
        """
        self.user_profiles = pd.read_sql_query(user_stats_query, conn)
        self.user_profiles.set_index("user_id", inplace=True)

        # 计算热门度分数
        self.popularity_scores = self.item_features["avg_affinity_score"] * np.log1p(
            self.item_features["total_users"]
        )

        conn.close()
        logger.info(
            f"数据加载完成: {len(self.user_item_matrix)} 用户, {len(self.user_item_matrix.columns)} 视频"
        )

    def use_external_database(self, db_path: str):
        """
        使用外部数据库

        Args:
            db_path: 外部数据库文件路径
        """
        self.db_path = db_path
        self._load_data()

    def calculate_user_similarity(
        self, user_id: int, method: str = "cosine"
    ) -> pd.Series:
        """
        计算用户相似度

        Args:
            user_id: 目标用户ID
            method: 相似度计算方法 ('cosine', 'pearson', 'jaccard')

        Returns:
            用户相似度Series
        """
        if user_id not in self.user_item_matrix.index:
            return pd.Series(dtype=float)

        user_vector = self.user_item_matrix.loc[user_id]
        similarities = {}

        for other_user in self.user_item_matrix.index:
            if other_user == user_id:
                continue

            other_vector = self.user_item_matrix.loc[other_user]

            if method == "cosine":
                sim = self._cosine_similarity(user_vector, other_vector)
            elif method == "pearson":
                sim = self._pearson_correlation(user_vector, other_vector)
            elif method == "jaccard":
                sim = self._jaccard_similarity(user_vector, other_vector)
            else:
                sim = self._cosine_similarity(user_vector, other_vector)

            if sim > self.similarity_threshold:
                similarities[other_user] = sim

        return pd.Series(similarities).sort_values(ascending=False)

    def calculate_item_similarity(
        self, video_id: int, method: str = "cosine"
    ) -> pd.Series:
        """
        计算视频相似度

        Args:
            video_id: 目标视频ID
            method: 相似度计算方法

        Returns:
            视频相似度Series
        """
        if video_id not in self.user_item_matrix.columns:
            return pd.Series(dtype=float)

        item_vector = self.user_item_matrix[video_id]
        similarities = {}

        for other_video in self.user_item_matrix.columns:
            if other_video == video_id:
                continue

            other_vector = self.user_item_matrix[other_video]

            if method == "cosine":
                sim = self._cosine_similarity(item_vector, other_vector)
            elif method == "pearson":
                sim = self._pearson_correlation(item_vector, other_vector)
            else:
                sim = self._cosine_similarity(item_vector, other_vector)

            if sim > self.similarity_threshold:
                similarities[other_video] = sim

        return pd.Series(similarities).sort_values(ascending=False)

    def _cosine_similarity(self, vec1: pd.Series, vec2: pd.Series) -> float:
        """
        计算余弦相似度
        """
        # 只考虑两个向量都有评分的项目
        mask = (vec1 > 0) & (vec2 > 0)
        if mask.sum() < 2:
            return 0.0

        v1 = vec1[mask]
        v2 = vec2[mask]

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(v1, v2) / (norm1 * norm2)

    def _pearson_correlation(self, vec1: pd.Series, vec2: pd.Series) -> float:
        """
        计算皮尔逊相关系数
        """
        mask = (vec1 > 0) & (vec2 > 0)
        if mask.sum() < 2:
            return 0.0

        v1 = vec1[mask]
        v2 = vec2[mask]

        if v1.std() == 0 or v2.std() == 0:
            return 0.0

        return v1.corr(v2)

    def _jaccard_similarity(self, vec1: pd.Series, vec2: pd.Series) -> float:
        """
        计算Jaccard相似度
        """
        set1 = set(vec1[vec1 > 0].index)
        set2 = set(vec2[vec2 > 0].index)

        if len(set1) == 0 and len(set2) == 0:
            return 0.0

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0

    def collaborative_filtering_recommend(
        self, user_id: int, n_recommendations: int = 10, method: str = "user_based"
    ) -> List[Tuple[int, float]]:
        """
        协同过滤推荐

        Args:
            user_id: 用户ID
            n_recommendations: 推荐数量
            method: 推荐方法 ('user_based', 'item_based')

        Returns:
            推荐列表 [(video_id, predicted_score), ...]
        """
        if user_id not in self.user_item_matrix.index:
            return []

        user_ratings = self.user_item_matrix.loc[user_id]
        watched_videos = set(user_ratings[user_ratings > 0].index)

        if method == "user_based":
            return self._user_based_cf(user_id, watched_videos, n_recommendations)
        else:
            return self._item_based_cf(user_id, watched_videos, n_recommendations)

    def _user_based_cf(
        self, user_id: int, watched_videos: Set[int], n_recommendations: int
    ) -> List[Tuple[int, float]]:
        """
        基于用户的协同过滤
        """
        # 找到相似用户
        similar_users = self.calculate_user_similarity(user_id)

        if len(similar_users) == 0:
            return []

        # 预测评分
        predictions = {}
        user_mean = self.user_item_matrix.loc[user_id].mean()

        for video_id in self.user_item_matrix.columns:
            if video_id in watched_videos:
                continue

            numerator = 0
            denominator = 0

            for similar_user, similarity in similar_users.head(50).items():
                similar_user_rating = self.user_item_matrix.loc[similar_user, video_id]
                if similar_user_rating > 0:
                    similar_user_mean = self.user_item_matrix.loc[similar_user].mean()
                    numerator += similarity * (similar_user_rating - similar_user_mean)
                    denominator += abs(similarity)

            if denominator > 0:
                predicted_rating = user_mean + numerator / denominator
                predictions[video_id] = max(0, min(1, predicted_rating))

        # 排序并返回top-N
        sorted_predictions = sorted(
            predictions.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_predictions[:n_recommendations]

    def _item_based_cf(
        self, user_id: int, watched_videos: Set[int], n_recommendations: int
    ) -> List[Tuple[int, float]]:
        """
        基于物品的协同过滤
        """
        user_ratings = self.user_item_matrix.loc[user_id]
        predictions = {}

        for video_id in self.user_item_matrix.columns:
            if video_id in watched_videos:
                continue

            # 找到与该视频相似的视频
            similar_items = self.calculate_item_similarity(video_id)

            if len(similar_items) == 0:
                continue

            numerator = 0
            denominator = 0

            for similar_video, similarity in similar_items.head(20).items():
                if similar_video in watched_videos:
                    user_rating = user_ratings[similar_video]
                    numerator += similarity * user_rating
                    denominator += abs(similarity)

            if denominator > 0:
                predicted_rating = numerator / denominator
                predictions[video_id] = max(0, min(1, predicted_rating))

        sorted_predictions = sorted(
            predictions.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_predictions[:n_recommendations]

    def content_based_recommend(
        self, user_id: int, n_recommendations: int = 10
    ) -> List[Tuple[int, float]]:
        """
        基于内容的推荐

        Args:
            user_id: 用户ID
            n_recommendations: 推荐数量

        Returns:
            推荐列表
        """
        if user_id not in self.user_profiles.index:
            return []

        user_profile = self.user_profiles.loc[user_id]
        user_ratings = self.user_item_matrix.loc[user_id]
        watched_videos = set(user_ratings[user_ratings > 0].index)

        # 计算用户偏好特征
        user_preferences = self._build_user_content_profile(user_id, watched_videos)

        # 为未观看的视频计算匹配度
        recommendations = {}

        for video_id in self.item_features.index:
            if video_id in watched_videos:
                continue

            video_features = self.item_features.loc[video_id]

            # 计算内容相似度
            content_score = self._calculate_content_similarity(
                user_preferences, video_features
            )

            # 结合视频质量分数
            quality_score = video_features["avg_affinity_score"]

            # 综合评分
            final_score = 0.7 * content_score + 0.3 * quality_score
            recommendations[video_id] = final_score

        sorted_recommendations = sorted(
            recommendations.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_recommendations[:n_recommendations]

    def _build_user_content_profile(
        self, user_id: int, watched_videos: Set[int]
    ) -> Dict[str, float]:
        """
        构建用户内容偏好画像
        """
        user_ratings = self.user_item_matrix.loc[user_id]

        # 加权平均用户观看过的视频特征
        weighted_features = {
            "avg_rating": 0,
            "like_rate": 0,
            "avg_sentiment": 0,
            "total_users": 0,
        }

        total_weight = 0

        for video_id in watched_videos:
            if video_id in self.item_features.index:
                rating = user_ratings[video_id]
                video_features = self.item_features.loc[video_id]

                for feature in weighted_features.keys():
                    weighted_features[feature] += rating * video_features[feature]

                total_weight += rating

        if total_weight > 0:
            for feature in weighted_features.keys():
                weighted_features[feature] /= total_weight

        return weighted_features

    def _calculate_content_similarity(
        self, user_preferences: Dict[str, float], video_features: pd.Series
    ) -> float:
        """
        计算内容相似度
        """
        similarity = 0

        # 特征权重
        feature_weights = {
            "avg_rating": 0.3,
            "like_rate": 0.3,
            "avg_sentiment": 0.2,
            "total_users": 0.2,
        }

        for feature, weight in feature_weights.items():
            if feature in user_preferences:
                # 归一化差异
                max_val = self.item_features[feature].max()
                min_val = self.item_features[feature].min()

                if max_val > min_val:
                    user_norm = (user_preferences[feature] - min_val) / (
                        max_val - min_val
                    )
                    video_norm = (video_features[feature] - min_val) / (
                        max_val - min_val
                    )

                    # 计算相似度 (1 - 归一化距离)
                    feature_sim = 1 - abs(user_norm - video_norm)
                    similarity += weight * feature_sim

        return similarity

    def popularity_based_recommend(
        self, user_id: int, n_recommendations: int = 10, time_decay: bool = True
    ) -> List[Tuple[int, float]]:
        """
        基于热门度的推荐

        Args:
            user_id: 用户ID
            n_recommendations: 推荐数量
            time_decay: 是否考虑时间衰减

        Returns:
            推荐列表
        """
        if user_id in self.user_item_matrix.index:
            user_ratings = self.user_item_matrix.loc[user_id]
            watched_videos = set(user_ratings[user_ratings > 0].index)
        else:
            watched_videos = set()

        # 获取热门视频
        popularity_scores = self.popularity_scores.copy()

        # 移除已观看的视频
        for video_id in watched_videos:
            if video_id in popularity_scores.index:
                popularity_scores = popularity_scores.drop(video_id)

        # 时间衰减 (假设数据有时间信息)
        if time_decay:
            # 这里可以根据实际的时间数据进行衰减
            # 暂时使用简单的衰减策略
            pass

        # 排序并返回
        sorted_popularity = popularity_scores.sort_values(ascending=False)
        recommendations = [
            (int(video_id), float(score))
            for video_id, score in sorted_popularity.head(n_recommendations).items()
        ]

        return recommendations

    def hybrid_recommend(
        self,
        user_id: int,
        n_recommendations: int = 10,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[Tuple[int, float]]:
        """
        混合推荐算法

        Args:
            user_id: 用户ID
            n_recommendations: 推荐数量
            weights: 各算法权重 {'cf': 0.4, 'content': 0.3, 'popularity': 0.3}

        Returns:
            推荐列表
        """
        if weights is None:
            weights = {"cf": 0.4, "content": 0.3, "popularity": 0.3}

        # 获取各种推荐结果
        cf_recommendations = self.collaborative_filtering_recommend(
            user_id, n_recommendations * 2
        )
        content_recommendations = self.content_based_recommend(
            user_id, n_recommendations * 2
        )
        popularity_recommendations = self.popularity_based_recommend(
            user_id, n_recommendations * 2
        )

        # 合并推荐结果
        combined_scores = defaultdict(float)

        # 协同过滤
        for video_id, score in cf_recommendations:
            combined_scores[video_id] += weights["cf"] * score

        # 内容过滤
        for video_id, score in content_recommendations:
            combined_scores[video_id] += weights["content"] * score

        # 热门推荐
        max_pop_score = (
            max([score for _, score in popularity_recommendations])
            if popularity_recommendations
            else 1
        )
        for video_id, score in popularity_recommendations:
            normalized_score = score / max_pop_score
            combined_scores[video_id] += weights["popularity"] * normalized_score

        # 排序并返回
        sorted_recommendations = sorted(
            combined_scores.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_recommendations[:n_recommendations]

    def diversified_recommend(
        self,
        user_id: int,
        n_recommendations: int = 10,
        diversity_threshold: float = 0.7,
    ) -> List[Tuple[int, float]]:
        """
        多样性推荐算法

        Args:
            user_id: 用户ID
            n_recommendations: 推荐数量
            diversity_threshold: 多样性阈值

        Returns:
            推荐列表
        """
        # 获取初始推荐列表
        initial_recommendations = self.hybrid_recommend(user_id, n_recommendations * 3)

        if not initial_recommendations:
            return []

        # 多样性选择
        selected_recommendations = []
        candidate_videos = [video_id for video_id, _ in initial_recommendations]

        # 选择第一个推荐
        first_video, first_score = initial_recommendations[0]
        selected_recommendations.append((first_video, first_score))
        candidate_videos.remove(first_video)

        # 迭代选择剩余推荐
        while len(selected_recommendations) < n_recommendations and candidate_videos:
            best_video = None
            best_score = -1

            for candidate_video in candidate_videos:
                # 计算与已选择视频的平均相似度
                avg_similarity = 0
                for selected_video, _ in selected_recommendations:
                    similarity = self._calculate_video_content_similarity(
                        candidate_video, selected_video
                    )
                    avg_similarity += similarity

                avg_similarity /= len(selected_recommendations)

                # 多样性分数 = 原始分数 * (1 - 平均相似度)
                original_score = next(
                    score
                    for vid, score in initial_recommendations
                    if vid == candidate_video
                )
                diversity_score = original_score * (
                    1 - avg_similarity * self.diversity_factor
                )

                if diversity_score > best_score:
                    best_score = diversity_score
                    best_video = candidate_video

            if best_video is not None:
                selected_recommendations.append((best_video, best_score))
                candidate_videos.remove(best_video)
            else:
                break

        return selected_recommendations

    def _calculate_video_content_similarity(self, video1: int, video2: int) -> float:
        """
        计算两个视频的内容相似度
        """
        if (
            video1 not in self.item_features.index
            or video2 not in self.item_features.index
        ):
            return 0.0

        features1 = self.item_features.loc[video1]
        features2 = self.item_features.loc[video2]

        # 计算特征向量的余弦相似度
        feature_vector1 = np.array(
            [
                features1["avg_rating"],
                features1["like_rate"],
                features1["avg_sentiment"],
            ]
        )
        feature_vector2 = np.array(
            [
                features2["avg_rating"],
                features2["like_rate"],
                features2["avg_sentiment"],
            ]
        )

        norm1 = np.linalg.norm(feature_vector1)
        norm2 = np.linalg.norm(feature_vector2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(feature_vector1, feature_vector2) / (norm1 * norm2)

    def cold_start_recommend(
        self, user_id: int, n_recommendations: int = 10
    ) -> List[Tuple[int, float]]:
        """
        冷启动推荐 (新用户)

        Args:
            user_id: 用户ID
            n_recommendations: 推荐数量

        Returns:
            推荐列表
        """
        # 对于新用户，主要基于热门度和多样性
        popular_videos = self.popularity_based_recommend(user_id, n_recommendations * 2)

        # 确保推荐的多样性
        if len(popular_videos) <= n_recommendations:
            return popular_videos

        # 选择多样化的热门视频
        selected_videos = []
        candidate_videos = [video_id for video_id, _ in popular_videos]

        # 选择第一个
        first_video, first_score = popular_videos[0]
        selected_videos.append((first_video, first_score))
        candidate_videos.remove(first_video)

        # 选择剩余的，确保多样性
        while len(selected_videos) < n_recommendations and candidate_videos:
            best_video = None
            best_diversity_score = -1

            for candidate_video in candidate_videos:
                # 计算多样性分数
                min_similarity = 1.0
                for selected_video, _ in selected_videos:
                    similarity = self._calculate_video_content_similarity(
                        candidate_video, selected_video
                    )
                    min_similarity = min(min_similarity, similarity)

                diversity_score = 1 - min_similarity

                if diversity_score > best_diversity_score:
                    best_diversity_score = diversity_score
                    best_video = candidate_video

            if best_video is not None:
                original_score = next(
                    score for vid, score in popular_videos if vid == best_video
                )
                selected_videos.append((best_video, original_score))
                candidate_videos.remove(best_video)
            else:
                break

        return selected_videos

    def explain_recommendation(self, user_id: int, video_id: int) -> Dict[str, any]:
        """
        推荐解释

        Args:
            user_id: 用户ID
            video_id: 视频ID

        Returns:
            推荐解释信息
        """
        explanation = {
            "user_id": user_id,
            "video_id": video_id,
            "reasons": [],
            "similar_users": [],
            "similar_videos": [],
            "video_features": {},
            "confidence": 0.0,
        }

        # 视频特征
        if video_id in self.item_features.index:
            video_features = self.item_features.loc[video_id]
            explanation["video_features"] = {
                "avg_affinity_score": float(video_features["avg_affinity_score"]),
                "total_users": int(video_features["total_users"]),
                "avg_rating": float(video_features["avg_rating"]),
                "like_rate": float(video_features["like_rate"]),
                "avg_sentiment": float(video_features["avg_sentiment"]),
            }

        if user_id not in self.user_item_matrix.index:
            explanation["reasons"].append("新用户推荐：基于热门度")
            explanation["confidence"] = 0.6
            return explanation

        # 查找相似用户
        similar_users = self.calculate_user_similarity(user_id)
        if len(similar_users) > 0:
            explanation["similar_users"] = [
                {"user_id": int(uid), "similarity": float(sim)}
                for uid, sim in similar_users.head(3).items()
            ]
            explanation["reasons"].append(f"相似用户也喜欢这个视频")

        # 查找相似视频
        user_ratings = self.user_item_matrix.loc[user_id]
        watched_videos = user_ratings[user_ratings > 0].index

        if len(watched_videos) > 0:
            # 找到用户最喜欢的视频
            top_watched = user_ratings.nlargest(3)
            similar_videos_info = []

            for watched_video in top_watched.index:
                similar_videos = self.calculate_item_similarity(watched_video)
                if video_id in similar_videos.index:
                    similarity = similar_videos[video_id]
                    similar_videos_info.append(
                        {
                            "video_id": int(watched_video),
                            "similarity": float(similarity),
                            "user_rating": float(top_watched[watched_video]),
                        }
                    )

            if similar_videos_info:
                explanation["similar_videos"] = similar_videos_info
                explanation["reasons"].append("与您喜欢的视频相似")

        # 内容匹配
        if user_id in self.user_profiles.index:
            user_profile = self.user_profiles.loc[user_id]
            if video_id in self.item_features.index:
                video_features = self.item_features.loc[video_id]

                # 检查特征匹配
                if (
                    abs(user_profile["avg_sentiment"] - video_features["avg_sentiment"])
                    < 0.2
                ):
                    explanation["reasons"].append("情感倾向匹配您的偏好")

                if abs(user_profile["like_rate"] - video_features["like_rate"]) < 0.3:
                    explanation["reasons"].append("点赞模式符合您的习惯")

        # 热门推荐
        if video_id in self.popularity_scores.index:
            popularity_rank = self.popularity_scores.rank(ascending=False)[video_id]
            total_videos = len(self.popularity_scores)
            if popularity_rank <= total_videos * 0.1:  # Top 10%
                explanation["reasons"].append("热门视频推荐")

        # 计算置信度
        confidence_factors = [
            len(explanation["similar_users"]) * 0.2,
            len(explanation["similar_videos"]) * 0.3,
            len(explanation["reasons"]) * 0.1,
            (
                0.4
                if explanation["video_features"].get("avg_affinity_score", 0) > 0.8
                else 0.2
            ),
        ]
        explanation["confidence"] = min(1.0, sum(confidence_factors))

        return explanation

    def batch_recommend(
        self,
        user_ids: List[int],
        n_recommendations: int = 10,
        algorithm: str = "hybrid",
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        批量推荐

        Args:
            user_ids: 用户ID列表
            n_recommendations: 推荐数量
            algorithm: 推荐算法

        Returns:
            批量推荐结果
        """
        results = {}

        for user_id in user_ids:
            if algorithm == "collaborative_filtering":
                recommendations = self.collaborative_filtering_recommend(
                    user_id, n_recommendations
                )
            elif algorithm == "content_based":
                recommendations = self.content_based_recommend(
                    user_id, n_recommendations
                )
            elif algorithm == "popularity":
                recommendations = self.popularity_based_recommend(
                    user_id, n_recommendations
                )
            elif algorithm == "diversified":
                recommendations = self.diversified_recommend(user_id, n_recommendations)
            elif algorithm == "cold_start":
                recommendations = self.cold_start_recommend(user_id, n_recommendations)
            else:  # hybrid
                recommendations = self.hybrid_recommend(user_id, n_recommendations)

            results[user_id] = recommendations

        return results

    def add_user(
        self,
        age: int,
        gender: str,
        education: str,
        hobby: str,
        address: str,
        income: float,
        career: str,
        user_id: int = "AUTO",
    ):
        """
        添加用户信息

        Args:
            age: 年龄
            gender: 性别
            education: 学历
            hobby: 爱好
            address: 地址
            income: 收入
            career: 职业
            user_id: 用户ID
        """
        if user_id == "AUTO":
            user_id = self.user_profiles.index.max() + 1
        self.user_profiles.loc[user_id] = {
            "age": age,
            "gender": gender,
            "education": education,
            "hobby": hobby,
            "address": address,
            "income": income,
            "career": career,
        }
        # 持久化
        self.user_profiles.to_csv("user_profiles.csv")

    def evaluate_recommendations(
        self, test_data: Dict[int, List[int]], algorithm: str = "hybrid", k: int = 10
    ) -> Dict[str, float]:
        """
        评估推荐算法

        Args:
            test_data: 测试数据 {user_id: [liked_video_ids]}
            algorithm: 推荐算法
            k: 推荐数量

        Returns:
            评估指标
        """
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for user_id, liked_videos in test_data.items():
            # 获取推荐
            if algorithm == "collaborative_filtering":
                recommendations = self.collaborative_filtering_recommend(user_id, k)
            elif algorithm == "content_based":
                recommendations = self.content_based_recommend(user_id, k)
            elif algorithm == "popularity":
                recommendations = self.popularity_based_recommend(user_id, k)
            else:  # hybrid
                recommendations = self.hybrid_recommend(user_id, k)

            recommended_videos = [video_id for video_id, _ in recommendations]

            # 计算指标
            if len(recommended_videos) > 0 and len(liked_videos) > 0:
                hits = len(set(recommended_videos) & set(liked_videos))
                precision = hits / len(recommended_videos)
                recall = hits / len(liked_videos)
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )

                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)

        return {
            "precision": np.mean(precision_scores) if precision_scores else 0,
            "recall": np.mean(recall_scores) if recall_scores else 0,
            "f1_score": np.mean(f1_scores) if f1_scores else 0,
            "coverage": len(
                set(
                    [
                        vid
                        for recs in [
                            self.hybrid_recommend(uid, k) for uid in test_data.keys()
                        ]
                        for vid, _ in recs
                    ]
                )
            )
            / len(self.user_item_matrix.columns),
        }


# def main():
#     """
#     推荐引擎演示
#     """
#     from rich.panel import Panel
#     from rich.table import Table

#     console.print(Panel("[bold blue]🎬 视频推荐算法引擎演示[/bold blue]", style="blue"))

#     try:
#         # 创建推荐引擎
#         engine = VideoRecommendationEngine()

#         # 选择测试用户
#         test_user_id = 2017

#         console.print(Panel(f"[bold yellow]👤 为用户 {test_user_id} 生成推荐[/bold yellow]", style="yellow"))

#         # 1. 协同过滤推荐
#         logger.info("[cyan]🤝 协同过滤推荐 (Top 5):[/cyan]")
#         cf_recommendations = engine.collaborative_filtering_recommend(test_user_id, 5)
#         cf_table = Table(title="协同过滤推荐")
#         cf_table.add_column("排名", style="cyan")
#         cf_table.add_column("视频ID", style="magenta")
#         cf_table.add_column("预测评分", style="green")
#         for i, (video_id, score) in enumerate(cf_recommendations, 1):
#             cf_table.add_row(str(i), str(video_id), f"{score:.4f}")
#         console.print(cf_table)

#         # 2. 内容推荐
#         logger.info("[cyan]📄 基于内容推荐 (Top 5):[/cyan]")
#         content_recommendations = engine.content_based_recommend(test_user_id, 5)
#         content_table = Table(title="基于内容推荐")
#         content_table.add_column("排名", style="cyan")
#         content_table.add_column("视频ID", style="magenta")
#         content_table.add_column("内容匹配度", style="green")
#         for i, (video_id, score) in enumerate(content_recommendations, 1):
#             content_table.add_row(str(i), str(video_id), f"{score:.4f}")
#         console.print(content_table)

#         # 3. 热门推荐
#         logger.info("[cyan]🔥 热门推荐 (Top 5):[/cyan]")
#         popularity_recommendations = engine.popularity_based_recommend(test_user_id, 5)
#         pop_table = Table(title="热门推荐")
#         pop_table.add_column("排名", style="cyan")
#         pop_table.add_column("视频ID", style="magenta")
#         pop_table.add_column("热门度", style="green")
#         for i, (video_id, score) in enumerate(popularity_recommendations, 1):
#             pop_table.add_row(str(i), str(video_id), f"{score:.4f}")
#         console.print(pop_table)

#         # 4. 混合推荐
#         logger.info("[cyan]🎯 混合推荐 (Top 5):[/cyan]")
#         hybrid_recommendations = engine.hybrid_recommend(test_user_id, 5)
#         hybrid_table = Table(title="混合推荐")
#         hybrid_table.add_column("排名", style="cyan")
#         hybrid_table.add_column("视频ID", style="magenta")
#         hybrid_table.add_column("综合评分", style="green")
#         for i, (video_id, score) in enumerate(hybrid_recommendations, 1):
#             hybrid_table.add_row(str(i), str(video_id), f"{score:.4f}")
#         console.print(hybrid_table)

#         # 5. 多样性推荐
#         logger.info("[cyan]🌈 多样性推荐 (Top 5):[/cyan]")
#         diverse_recommendations = engine.diversified_recommend(test_user_id, 5)
#         diverse_table = Table(title="多样性推荐")
#         diverse_table.add_column("排名", style="cyan")
#         diverse_table.add_column("视频ID", style="magenta")
#         diverse_table.add_column("多样性评分", style="green")
#         for i, (video_id, score) in enumerate(diverse_recommendations, 1):
#             diverse_table.add_row(str(i), str(video_id), f"{score:.4f}")
#         console.print(diverse_table)

#         # 6. 推荐解释
#         if hybrid_recommendations:
#             top_recommendation = hybrid_recommendations[0][0]
#             logger.info(f"[cyan]💡 推荐解释 - 视频 {top_recommendation}:[/cyan]")
#             explanation = engine.explain_recommendation(test_user_id, top_recommendation)

#             explanation_content = f"[bold]置信度:[/bold] {explanation['confidence']:.2f}\n\n[bold]推荐原因:[/bold]\n"
#             for reason in explanation['reasons']:
#                 explanation_content += f"  • {reason}\n"

#             if explanation['similar_users']:
#                 explanation_content += "\n[bold]相似用户:[/bold]\n"
#                 for user_info in explanation['similar_users']:
#                     explanation_content += f"  • 用户 {user_info['user_id']}: 相似度 {user_info['similarity']:.3f}\n"

#             console.print(Panel(explanation_content, title="推荐解释", style="green"))

#         # 7. 冷启动演示
#         logger.info("[cyan]❄️ 冷启动推荐演示 (新用户):[/cyan]")
#         cold_start_recommendations = engine.cold_start_recommend(9999, 5)
#         cold_table = Table(title="冷启动推荐")
#         cold_table.add_column("排名", style="cyan")
#         cold_table.add_column("视频ID", style="magenta")
#         cold_table.add_column("评分", style="green")
#         for i, (video_id, score) in enumerate(cold_start_recommendations, 1):
#             cold_table.add_row(str(i), str(video_id), f"{score:.4f}")
#         console.print(cold_table)

#         console.print(Panel("[bold green]🎉 推荐引擎演示完成！[/bold green]", style="green"))

#     except Exception as e:
#         logger.error(f"[red]❌ 演示过程中发生错误: {e}[/red]")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()
