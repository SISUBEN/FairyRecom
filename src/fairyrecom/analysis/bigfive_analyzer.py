import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict
import os
import logging
import warnings
from rich.logging import RichHandler
from rich.console import Console
from rich.panel import Panel
from rich import print as rprint

# 配置Rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)
console = Console()

warnings.filterwarnings('ignore')

class BigFiveSimilarityAnalyzer:
    """
    BigFive性格相似度分析器
    用于分析用户间基于BigFive性格测试结果的相似度
    """
    
    def __init__(self, data_path: str = None):
        """
        初始化分析器
        
        Args:
            data_path: BigFive数据文件路径
        """
        self.data_path = data_path or "reasoner/dataset/dataset/bigfive.csv"
        self.data = None
        self.features = None
        self.scaler = StandardScaler()
        self.scaled_features = None
        
    def load_data(self) -> pd.DataFrame:
        """
        加载BigFive数据
        
        Returns:
            加载的数据DataFrame
        """
        try:
            # 尝试不同的分隔符
            try:
                self.data = pd.read_csv(self.data_path, sep='\t')
            except:
                self.data = pd.read_csv(self.data_path)
            
            logger.info(f"[green]数据加载成功，共{len(self.data)}个用户[/green]")
            logger.info(f"[cyan]数据列: {list(self.data.columns)}[/cyan]")
            
            # 提取特征列（Q1-Q15）
            feature_cols = [f'Q{i}' for i in range(1, 16)]
            missing_cols = [col for col in feature_cols if col not in self.data.columns]
            if missing_cols:
                logger.warning(f"[yellow]警告: 缺少列 {missing_cols}[/yellow]")
                feature_cols = [col for col in feature_cols if col in self.data.columns]
            
            self.features = self.data[feature_cols].values
            
            # 标准化特征
            self.scaled_features = self.scaler.fit_transform(self.features)
            
            return self.data
        except Exception as e:
            logger.error(f"[red]数据加载失败: {e}[/red]")
            return None
    
    def calculate_cosine_similarity(self, user_ids: List[int] = None) -> np.ndarray:
        """
        计算余弦相似度
        
        Args:
            user_ids: 指定用户ID列表，如果为None则计算所有用户
            
        Returns:
            相似度矩阵
        """
        if self.scaled_features is None:
            raise ValueError("请先加载数据")
            
        if user_ids is not None:
            # 获取指定用户的特征
            indices = [self.data[self.data['user_id'] == uid].index[0] for uid in user_ids if uid in self.data['user_id'].values]
            features_subset = self.scaled_features[indices]
            return cosine_similarity(features_subset)
        else:
            return cosine_similarity(self.scaled_features)
    
    def calculate_euclidean_similarity(self, user_ids: List[int] = None) -> np.ndarray:
        """
        计算欧几里得距离（转换为相似度）
        
        Args:
            user_ids: 指定用户ID列表，如果为None则计算所有用户
            
        Returns:
            相似度矩阵（距离越小，相似度越高）
        """
        if self.scaled_features is None:
            raise ValueError("请先加载数据")
            
        if user_ids is not None:
            indices = [self.data[self.data['user_id'] == uid].index[0] for uid in user_ids if uid in self.data['user_id'].values]
            features_subset = self.scaled_features[indices]
            distances = euclidean_distances(features_subset)
        else:
            distances = euclidean_distances(self.scaled_features)
        
        # 将距离转换为相似度（使用负指数函数）
        similarities = np.exp(-distances)
        return similarities
    
    def find_most_similar_users(self, target_user_id: int, top_k: int = 10, method: str = 'cosine') -> List[Tuple[int, float]]:
        """
        找到与目标用户最相似的K个用户
        
        Args:
            target_user_id: 目标用户ID
            top_k: 返回最相似的K个用户
            method: 相似度计算方法 ('cosine' 或 'euclidean')
            
        Returns:
            [(user_id, similarity_score), ...] 按相似度降序排列
        """
        if target_user_id not in self.data['user_id'].values:
            raise ValueError(f"用户ID {target_user_id} 不存在")
        
        # 计算相似度矩阵
        if method == 'cosine':
            similarity_matrix = self.calculate_cosine_similarity()
        elif method == 'euclidean':
            similarity_matrix = self.calculate_euclidean_similarity()
        else:
            raise ValueError("method必须是'cosine'或'euclidean'")
        
        # 获取目标用户的索引
        target_index = self.data[self.data['user_id'] == target_user_id].index[0]
        
        # 获取与目标用户的相似度
        similarities = similarity_matrix[target_index]
        
        # 排序并获取top_k（排除自己）
        user_similarities = [(self.data.iloc[i]['user_id'], similarities[i]) 
                           for i in range(len(similarities)) 
                           if i != target_index]
        
        user_similarities.sort(key=lambda x: x[1], reverse=True)
        
        return user_similarities[:top_k]
    
    def analyze_personality_profile(self, user_id: int) -> Dict[str, float]:
        """
        分析用户的性格特征
        
        Args:
            user_id: 用户ID
            
        Returns:
            性格特征字典
        """
        if user_id not in self.data['user_id'].values:
            raise ValueError(f"用户ID {user_id} 不存在")
        
        user_data = self.data[self.data['user_id'] == user_id].iloc[0]
        
        # BigFive五大性格维度的问题映射（基于常见的BigFive问卷）
        personality_dimensions = {
            '外向性 (Extraversion)': [1, 6, 11],  # 社交性、活跃性
            '宜人性 (Agreeableness)': [2, 7, 12],  # 合作性、信任
            '尽责性 (Conscientiousness)': [3, 8, 13],  # 组织性、自律
            '神经质 (Neuroticism)': [4, 9, 14],  # 情绪稳定性
            '开放性 (Openness)': [5, 10, 15]  # 创造性、好奇心
        }
        
        profile = {}
        for dimension, questions in personality_dimensions.items():
            scores = [user_data[f'Q{q}'] for q in questions]
            profile[dimension] = np.mean(scores)
        
        return profile
    
    def visualize_similarity_heatmap(self, user_ids: List[int] = None, figsize: Tuple[int, int] = (12, 10)):
        """
        可视化相似度热力图
        
        Args:
            user_ids: 指定用户ID列表，如果为None则随机选择20个用户
            figsize: 图形大小
        """
        if user_ids is None:
            # 随机选择20个用户进行可视化
            user_ids = np.random.choice(self.data['user_id'].values, min(20, len(self.data)), replace=False)
        
        similarity_matrix = self.calculate_cosine_similarity(user_ids)
        
        plt.figure(figsize=figsize)
        sns.heatmap(similarity_matrix, 
                   xticklabels=user_ids, 
                   yticklabels=user_ids,
                   annot=True, 
                   cmap='viridis', 
                   fmt='.2f')
        plt.title('用户性格相似度热力图 (余弦相似度)')
        plt.xlabel('用户ID')
        plt.ylabel('用户ID')
        plt.tight_layout()
        plt.show()
    
    def visualize_personality_profile(self, user_id: int):
        """
        可视化用户性格特征雷达图
        
        Args:
            user_id: 用户ID
        """
        profile = self.analyze_personality_profile(user_id)
        
        # 创建雷达图
        categories = list(profile.keys())
        values = list(profile.values())
        
        # 计算角度
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # 闭合图形
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        ax.plot(angles, values, 'o-', linewidth=2, label=f'用户 {user_id}')
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 5)
        ax.set_title(f'用户 {user_id} 性格特征雷达图', size=16, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        plt.tight_layout()
        plt.show()
    
    def generate_similarity_report(self, target_user_id: int, top_k: int = 5) -> str:
        """
        生成相似度分析报告
        
        Args:
            target_user_id: 目标用户ID
            top_k: 最相似的用户数量
            
        Returns:
            分析报告字符串
        """
        # 获取最相似用户
        similar_users_cosine = self.find_most_similar_users(target_user_id, top_k, 'cosine')
        similar_users_euclidean = self.find_most_similar_users(target_user_id, top_k, 'euclidean')
        
        # 获取性格特征
        target_profile = self.analyze_personality_profile(target_user_id)
        
        report = f"""
=== 用户 {target_user_id} 性格相似度分析报告 ===

【性格特征分析】
"""
        for dimension, score in target_profile.items():
            report += f"{dimension}: {score:.2f}/5.0\n"
        
        report += f"""

【最相似用户 - 余弦相似度】
"""
        for i, (user_id, similarity) in enumerate(similar_users_cosine, 1):
            report += f"{i}. 用户 {user_id}: 相似度 {similarity:.4f}\n"
        
        report += f"""

【最相似用户 - 欧几里得相似度】
"""
        for i, (user_id, similarity) in enumerate(similar_users_euclidean, 1):
            report += f"{i}. 用户 {user_id}: 相似度 {similarity:.4f}\n"
        
        return report


def main():
    """
    主函数 - 演示程序功能
    """
    # 创建分析器实例
    analyzer = BigFiveSimilarityAnalyzer()
    
    # 加载数据
    data = analyzer.load_data()
    if data is None:
        return
    
    console.print(Panel("[bold blue]BigFive性格相似度分析器[/bold blue]", style="blue"))
    console.print("[cyan]数据概览:[/cyan]")
    console.print(f"[green]- 用户数量: {len(data)}[/green]")
    console.print(f"[green]- 特征维度: 15个问题 (Q1-Q15)[/green]")
    console.print(f"[green]- 数据范围: {data.iloc[:, 1:].min().min()} - {data.iloc[:, 1:].max().max()}[/green]")
    
    # 示例分析
    target_user = data['user_id'].iloc[0]  # 选择第一个用户作为示例
    console.print(Panel(f"[bold yellow]示例分析: 用户 {target_user}[/bold yellow]", style="yellow"))
    
    # 生成分析报告
    report = analyzer.generate_similarity_report(target_user, top_k=5)
    console.print(Panel(report, title="[bold green]分析报告[/bold green]", style="green"))
    
    # 可视化性格特征
    logger.info("[blue]正在生成性格特征雷达图...[/blue]")
    analyzer.visualize_personality_profile(target_user)
    
    # 可视化相似度热力图
    logger.info("[blue]正在生成相似度热力图...[/blue]")
    sample_users = data['user_id'].head(10).tolist()  # 选择前10个用户
    analyzer.visualize_similarity_heatmap(sample_users, figsize=(10, 8))
    
    logger.info("[bold green]分析完成！[/bold green]")


if __name__ == "__main__":
    main()