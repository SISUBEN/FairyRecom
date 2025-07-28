# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sqlite3
import json
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from textblob import TextBlob
import warnings
import logging
from rich.logging import RichHandler
from rich.console import Console
warnings.filterwarnings('ignore')

# 配置Rich logging
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

class VideoAffinityAnalyzer:
    """
    视频喜爱度分析器
    
    功能:
    1. 加载用户-视频交互数据
    2. 计算多维度喜爱度分数
    3. 存储分析结果到SQLite数据库
    4. 提供查询和统计功能
    """
    
    def __init__(self, db_path: str = "data.db"):
        """
        初始化分析器
        
        Args:
            db_path: SQLite数据库文件路径
        """
        self.db_path = db_path
        self.data = None
        self.affinity_scores = {}
        
        # 初始化数据库
        self._init_database()
    
    def _init_database(self):
        """
        初始化SQLite数据库和表结构
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建用户视频喜爱度表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_video_affinity (
                user_id INTEGER NOT NULL,
                video_id INTEGER NOT NULL,
                affinity_score REAL CHECK (affinity_score BETWEEN 0 AND 1),
                rating_score REAL,
                like_score REAL,
                sentiment_score REAL,
                engagement_score REAL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (user_id, video_id)
            )
        """)
        
        # 创建视频统计表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS video_statistics (
                video_id INTEGER PRIMARY KEY,
                total_users INTEGER,
                avg_affinity_score REAL,
                avg_rating REAL,
                like_rate REAL,
                avg_sentiment REAL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 创建用户统计表
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_statistics (
                user_id INTEGER PRIMARY KEY,
                total_videos INTEGER,
                avg_affinity_score REAL,
                avg_rating REAL,
                like_rate REAL,
                avg_sentiment REAL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info(f"[green]数据库初始化完成: {self.db_path}[/green]")
    
    def load_data(self, file_path: str = "reasoner/dataset/dataset/interaction.csv") -> pd.DataFrame:
        """
        加载用户-视频交互数据
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            加载的数据DataFrame
        """
        try:
            # 尝试不同的分隔符
            for sep in ['\t', ',', ';']:
                try:
                    self.data = pd.read_csv(file_path, sep=sep)
                    if len(self.data.columns) > 5:  # 确保有足够的列
                        break
                except:
                    continue
            
            if self.data is None:
                raise ValueError("无法读取数据文件")
            
            logger.info(f"[green]数据加载成功，共{len(self.data)}条记录[/green]")
            logger.info(f"[cyan]数据列: {list(self.data.columns)}[/cyan]")
            
            # 数据预处理
            self._preprocess_data()
            
            return self.data
            
        except Exception as e:
            logger.error(f"[red]数据加载失败: {e}[/red]")
            raise
    
    def _preprocess_data(self):
        """
        数据预处理
        """
        # 处理缺失值
        self.data['rating'] = pd.to_numeric(self.data['rating'], errors='coerce')
        self.data['like'] = pd.to_numeric(self.data['like'], errors='coerce')
        
        # 填充缺失值
        self.data['rating'].fillna(self.data['rating'].mean(), inplace=True)
        self.data['like'].fillna(0, inplace=True)
        self.data['review'].fillna('', inplace=True)
        
        logger.info("[green]数据预处理完成[/green]")
    
    def calculate_sentiment_score(self, text: str) -> float:
        """
        计算文本情感分数
        
        Args:
            text: 评论文本
            
        Returns:
            情感分数 (0-1)
        """
        if not text or pd.isna(text):
            return 0.5  # 中性
        
        try:
            # 使用TextBlob进行情感分析
            blob = TextBlob(str(text))
            sentiment = blob.sentiment.polarity
            
            # 将情感分数从[-1,1]映射到[0,1]
            return (sentiment + 1) / 2
        except:
            # 简单的关键词情感分析作为备选
            positive_words = ['like', 'love', 'good', 'great', 'excellent', 'amazing', 'wonderful', 
                            '喜欢', '爱', '好', '棒', '优秀', '精彩', '有趣']
            negative_words = ['hate', 'bad', 'terrible', 'awful', 'boring', 'stupid',
                            '讨厌', '坏', '糟糕', '无聊', '愚蠢', '差']
            
            text_lower = str(text).lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            if positive_count + negative_count == 0:
                return 0.5
            
            return positive_count / (positive_count + negative_count)
    
    def calculate_engagement_score(self, row) -> float:
        """
        计算用户参与度分数
        
        Args:
            row: 数据行
            
        Returns:
            参与度分数 (0-1)
        """
        score = 0.0
        
        # 评论长度贡献
        review_length = len(str(row.get('review', '')))
        if review_length > 0:
            score += min(review_length / 100, 0.3)  # 最多贡献0.3分
        
        # 标签数量贡献
        for tag_col in ['reason_tag', 'video_tag', 'interest_tag']:
            if tag_col in row and pd.notna(row[tag_col]):
                try:
                    tags = eval(str(row[tag_col])) if isinstance(row[tag_col], str) else row[tag_col]
                    if isinstance(tags, list):
                        score += min(len(tags) / 10, 0.2)  # 每个标签列最多贡献0.2分
                except:
                    pass
        
        # 是否愿意再次观看
        if 'watch_again' in row and row['watch_again'] == 1:
            score += 0.3
        
        return min(score, 1.0)
    
    def calculate_affinity_score(self, row) -> Dict[str, float]:
        """
        计算综合喜爱度分数
        
        Args:
            row: 数据行
            
        Returns:
            包含各维度分数的字典
        """
        # 评分分数 (1-5 -> 0-1)
        rating_score = (row['rating'] - 1) / 4 if pd.notna(row['rating']) else 0.5
        
        # 点赞分数
        like_score = float(row['like']) if pd.notna(row['like']) else 0.0
        
        # 情感分数
        sentiment_score = self.calculate_sentiment_score(row.get('review', ''))
        
        # 参与度分数
        engagement_score = self.calculate_engagement_score(row)
        
        # 综合喜爱度分数 (加权平均)
        weights = {
            'rating': 0.3,
            'like': 0.25,
            'sentiment': 0.25,
            'engagement': 0.2
        }
        
        affinity_score = (
            weights['rating'] * rating_score +
            weights['like'] * like_score +
            weights['sentiment'] * sentiment_score +
            weights['engagement'] * engagement_score
        )
        
        return {
            'affinity_score': affinity_score,
            'rating_score': rating_score,
            'like_score': like_score,
            'sentiment_score': sentiment_score,
            'engagement_score': engagement_score
        }
    
    def analyze_all_users(self) -> Dict[Tuple[int, int], Dict[str, float]]:
        """
        分析所有用户的视频喜爱度
        
        Returns:
            用户-视频喜爱度分数字典
        """
        if self.data is None:
            raise ValueError("请先加载数据")
        
        logger.info("[yellow]开始分析用户视频喜爱度...[/yellow]")
        
        affinity_results = {}
        
        for idx, row in self.data.iterrows():
            user_id = int(row['user_id'])
            video_id = int(row['video_id'])
            
            scores = self.calculate_affinity_score(row)
            affinity_results[(user_id, video_id)] = scores
            
            if (idx + 1) % 1000 == 0:
                logger.info(f"[cyan]已处理 {idx + 1} 条记录[/cyan]")
        
        self.affinity_scores = affinity_results
        logger.info(f"[green]分析完成，共处理 {len(affinity_results)} 个用户-视频对[/green]")
        
        return affinity_results
    
    def save_to_database(self, batch_size: int = 1000):
        """
        将分析结果保存到数据库
        
        Args:
            batch_size: 批量插入大小
        """
        if not self.affinity_scores:
            raise ValueError("请先进行喜爱度分析")
        
        logger.info("[yellow]开始保存数据到数据库...[/yellow]")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 清空旧数据
        cursor.execute("DELETE FROM user_video_affinity")
        
        # 批量插入数据
        batch_data = []
        for (user_id, video_id), scores in self.affinity_scores.items():
            batch_data.append((
                user_id,
                video_id,
                scores['affinity_score'],
                scores['rating_score'],
                scores['like_score'],
                scores['sentiment_score'],
                scores['engagement_score'],
                datetime.now()
            ))
            
            if len(batch_data) >= batch_size:
                cursor.executemany("""
                    INSERT OR REPLACE INTO user_video_affinity 
                    (user_id, video_id, affinity_score, rating_score, like_score, 
                     sentiment_score, engagement_score, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, batch_data)
                batch_data = []
        
        # 插入剩余数据
        if batch_data:
            cursor.executemany("""
                INSERT OR REPLACE INTO user_video_affinity 
                (user_id, video_id, affinity_score, rating_score, like_score, 
                 sentiment_score, engagement_score, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, batch_data)
        
        conn.commit()
        
        # 更新统计表
        self._update_statistics(cursor)
        
        conn.commit()
        conn.close()
        
        logger.info(f"[green]数据保存完成，共保存 {len(self.affinity_scores)} 条记录[/green]")
    
    def _update_statistics(self, cursor):
        """
        更新统计表
        """
        logger.info("[yellow]更新统计信息...[/yellow]")
        
        # 更新视频统计
        cursor.execute("""
            INSERT OR REPLACE INTO video_statistics 
            (video_id, total_users, avg_affinity_score, avg_rating, like_rate, avg_sentiment, last_updated)
            SELECT 
                video_id,
                COUNT(*) as total_users,
                AVG(affinity_score) as avg_affinity_score,
                AVG(rating_score) as avg_rating,
                AVG(like_score) as like_rate,
                AVG(sentiment_score) as avg_sentiment,
                datetime('now') as last_updated
            FROM user_video_affinity
            GROUP BY video_id
        """)
        
        # 更新用户统计
        cursor.execute("""
            INSERT OR REPLACE INTO user_statistics 
            (user_id, total_videos, avg_affinity_score, avg_rating, like_rate, avg_sentiment, last_updated)
            SELECT 
                user_id,
                COUNT(*) as total_videos,
                AVG(affinity_score) as avg_affinity_score,
                AVG(rating_score) as avg_rating,
                AVG(like_score) as like_rate,
                AVG(sentiment_score) as avg_sentiment,
                datetime('now') as last_updated
            FROM user_video_affinity
            GROUP BY user_id
        """)
    
    def get_user_affinity(self, user_id: int, limit: int = 10) -> List[Tuple]:
        """
        获取用户的视频喜爱度排行
        
        Args:
            user_id: 用户ID
            limit: 返回数量限制
            
        Returns:
            视频喜爱度列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT video_id, affinity_score, rating_score, like_score, sentiment_score, engagement_score
            FROM user_video_affinity
            WHERE user_id = ?
            ORDER BY affinity_score DESC
            LIMIT ?
        """, (user_id, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        return results
    
    def get_video_affinity(self, video_id: int, limit: int = 10) -> List[Tuple]:
        """
        获取视频的用户喜爱度排行
        
        Args:
            video_id: 视频ID
            limit: 返回数量限制
            
        Returns:
            用户喜爱度列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT user_id, affinity_score, rating_score, like_score, sentiment_score, engagement_score
            FROM user_video_affinity
            WHERE video_id = ?
            ORDER BY affinity_score DESC
            LIMIT ?
        """, (video_id, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        return results
    
    def get_top_videos(self, limit: int = 10) -> List[Tuple]:
        """
        获取最受欢迎的视频
        
        Args:
            limit: 返回数量限制
            
        Returns:
            视频统计列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT video_id, total_users, avg_affinity_score, avg_rating, like_rate, avg_sentiment
            FROM video_statistics
            ORDER BY avg_affinity_score DESC, total_users DESC
            LIMIT ?
        """, (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        return results
    
    def get_top_users(self, limit: int = 10) -> List[Tuple]:
        """
        获取最活跃的用户
        
        Args:
            limit: 返回数量限制
            
        Returns:
            用户统计列表
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT user_id, total_videos, avg_affinity_score, avg_rating, like_rate, avg_sentiment
            FROM user_statistics
            ORDER BY total_videos DESC, avg_affinity_score DESC
            LIMIT ?
        """, (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        return results
    
    def generate_report(self, output_file: str = "video_affinity_report.txt"):
        """
        生成分析报告
        
        Args:
            output_file: 输出文件路径
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 获取基本统计信息
        cursor.execute("SELECT COUNT(*) FROM user_video_affinity")
        total_records = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT user_id) FROM user_video_affinity")
        total_users = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT video_id) FROM user_video_affinity")
        total_videos = cursor.fetchone()[0]
        
        cursor.execute("SELECT AVG(affinity_score) FROM user_video_affinity")
        avg_affinity = cursor.fetchone()[0]
        
        conn.close()
        
        # 生成报告
        report = f"""
视频喜爱度分析报告
{'='*50}

基本统计信息:
- 总记录数: {total_records:,}
- 用户数量: {total_users:,}
- 视频数量: {total_videos:,}
- 平均喜爱度: {avg_affinity:.4f}

最受欢迎的视频 (Top 10):
{'视频ID':<10} {'用户数':<8} {'平均喜爱度':<12} {'平均评分':<10} {'点赞率':<8} {'情感分数':<10}
{'-'*70}
"""
        
        top_videos = self.get_top_videos(10)
        for video_id, users, affinity, rating, like_rate, sentiment in top_videos:
            report += f"{video_id:<10} {users:<8} {affinity:<12.4f} {rating:<10.4f} {like_rate:<8.4f} {sentiment:<10.4f}\n"
        
        report += f"""

最活跃的用户 (Top 10):
{'用户ID':<10} {'视频数':<8} {'平均喜爱度':<12} {'平均评分':<10} {'点赞率':<8} {'情感分数':<10}
{'-'*70}
"""
        
        top_users = self.get_top_users(10)
        for user_id, videos, affinity, rating, like_rate, sentiment in top_users:
            report += f"{user_id:<10} {videos:<8} {affinity:<12.4f} {rating:<10.4f} {like_rate:<8.4f} {sentiment:<10.4f}\n"
        
        report += f"""

分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
数据库文件: {self.db_path}
"""
        
        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"[green]分析报告已保存到: {output_file}[/green]")
        return report

def main():
    """
    主函数 - 演示视频喜爱度分析流程
    """
    logger.info("[bold blue]视频喜爱度分析系统[/bold blue]")
    logger.info("[blue]=" * 40 + "[/blue]")
    
    try:
        # 创建分析器
        analyzer = VideoAffinityAnalyzer()
        
        # 加载数据
        data = analyzer.load_data()
        logger.info(f"\n[cyan]数据概览:[/cyan]")
        logger.info(f"[white]- 数据形状: {data.shape}[/white]")
        logger.info(f"[white]- 用户数量: {data['user_id'].nunique()}[/white]")
        logger.info(f"[white]- 视频数量: {data['video_id'].nunique()}[/white]")
        logger.info(f"[white]- 平均评分: {data['rating'].mean():.2f}[/white]")
        logger.info(f"[white]- 点赞率: {data['like'].mean():.2%}[/white]")
        
        # 分析喜爱度
        affinity_scores = analyzer.analyze_all_users()
        
        # 保存到数据库
        analyzer.save_to_database()
        
        # 生成报告
        report = analyzer.generate_report()
        logger.info("\n" + "[blue]=" * 50 + "[/blue]")
        console.print(report)
        
        # 示例查询
        logger.info("\n" + "[blue]=" * 50 + "[/blue]")
        logger.info("[yellow]示例查询:[/yellow]")
        
        # 查看用户2017的喜爱视频
        user_affinity = analyzer.get_user_affinity(2017, 5)
        logger.info(f"\n[cyan]用户2017最喜爱的5个视频:[/cyan]")
        for video_id, affinity, rating, like, sentiment, engagement in user_affinity:
            logger.info(f"[white]视频{video_id}: 喜爱度{affinity:.4f} (评分{rating:.4f}, 点赞{like:.4f}, 情感{sentiment:.4f}, 参与度{engagement:.4f})[/white]")
        
        # 查看视频3539的用户喜爱度
        video_affinity = analyzer.get_video_affinity(3539, 5)
        logger.info(f"\n[cyan]最喜爱视频3539的5个用户:[/cyan]")
        for user_id, affinity, rating, like, sentiment, engagement in video_affinity:
            logger.info(f"[white]用户{user_id}: 喜爱度{affinity:.4f} (评分{rating:.4f}, 点赞{like:.4f}, 情感{sentiment:.4f}, 参与度{engagement:.4f})[/white]")
        
        logger.info("\n[green]🎉 视频喜爱度分析完成！[/green]")
        logger.info(f"[green]数据已保存到数据库: {analyzer.db_path}[/green]")
        
    except Exception as e:
        logger.error(f"\n[red]❌ 分析过程中发生错误: {e}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()