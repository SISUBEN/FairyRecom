
# -*- coding: utf-8 -*-
"""
视频喜爱度数据查询工具

这个工具用于查询和分析存储在data.db中的视频喜爱度数据。
提供多种查询功能，帮助用户深入了解数据分析结果。
"""

import sqlite3
import pandas as pd
from typing import List, Dict, Optional
import argparse
import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table
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

class VideoAffinityQuery:
    """
    视频喜爱度数据查询类
    """
    
    def __init__(self, db_path: str = "data.db"):
        """
        初始化查询器
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self._check_database()
    
    def _check_database(self):
        """
        检查数据库是否存在和有效
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 检查表是否存在
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name IN ('user_video_affinity', 'video_statistics', 'user_statistics')
            """)
            
            tables = [row[0] for row in cursor.fetchall()]
            required_tables = ['user_video_affinity', 'video_statistics', 'user_statistics']
            
            missing_tables = set(required_tables) - set(tables)
            if missing_tables:
                raise ValueError(f"数据库中缺少表: {missing_tables}")
            
            conn.close()
            logger.info(f"[green]数据库连接成功: {self.db_path}[/green]")
            
        except Exception as e:
            raise ValueError(f"数据库检查失败: {e}")
    
    def get_database_info(self) -> Dict:
        """
        获取数据库基本信息
        
        Returns:
            数据库信息字典
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        info = {}
        
        # 获取各表记录数
        cursor.execute("SELECT COUNT(*) FROM user_video_affinity")
        info['total_records'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM video_statistics")
        info['total_videos'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM user_statistics")
        info['total_users'] = cursor.fetchone()[0]
        
        # 获取喜爱度统计
        cursor.execute("SELECT MIN(affinity_score), MAX(affinity_score), AVG(affinity_score) FROM user_video_affinity")
        min_affinity, max_affinity, avg_affinity = cursor.fetchone()
        info['affinity_stats'] = {
            'min': min_affinity,
            'max': max_affinity,
            'avg': avg_affinity
        }
        
        conn.close()
        return info
    
    def query_user_preferences(self, user_id: int, limit: int = 20) -> pd.DataFrame:
        """
        查询用户的视频偏好
        
        Args:
            user_id: 用户ID
            limit: 返回记录数限制
            
        Returns:
            用户偏好DataFrame
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                video_id,
                affinity_score,
                rating_score,
                like_score,
                sentiment_score,
                engagement_score,
                last_updated
            FROM user_video_affinity
            WHERE user_id = ?
            ORDER BY affinity_score DESC
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(user_id, limit))
        conn.close()
        
        return df
    
    def query_video_popularity(self, video_id: int, limit: int = 20) -> pd.DataFrame:
        """
        查询视频的用户喜爱度
        
        Args:
            video_id: 视频ID
            limit: 返回记录数限制
            
        Returns:
            视频受欢迎程度DataFrame
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                user_id,
                affinity_score,
                rating_score,
                like_score,
                sentiment_score,
                engagement_score,
                last_updated
            FROM user_video_affinity
            WHERE video_id = ?
            ORDER BY affinity_score DESC
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(video_id, limit))
        conn.close()
        
        return df
    
    def get_top_videos(self, limit: int = 20, sort_by: str = 'avg_affinity_score') -> pd.DataFrame:
        """
        获取最受欢迎的视频
        
        Args:
            limit: 返回记录数限制
            sort_by: 排序字段 ('avg_affinity_score', 'total_users', 'like_rate')
            
        Returns:
            热门视频DataFrame
        """
        valid_sort_fields = ['avg_affinity_score', 'total_users', 'like_rate', 'avg_rating', 'avg_sentiment']
        if sort_by not in valid_sort_fields:
            sort_by = 'avg_affinity_score'
        
        conn = sqlite3.connect(self.db_path)
        
        query = f"""
            SELECT 
                video_id,
                total_users,
                avg_affinity_score,
                avg_rating,
                like_rate,
                avg_sentiment,
                last_updated
            FROM video_statistics
            ORDER BY {sort_by} DESC
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        
        return df
    
    def get_top_users(self, limit: int = 20, sort_by: str = 'total_videos') -> pd.DataFrame:
        """
        获取最活跃的用户
        
        Args:
            limit: 返回记录数限制
            sort_by: 排序字段 ('total_videos', 'avg_affinity_score', 'like_rate')
            
        Returns:
            活跃用户DataFrame
        """
        valid_sort_fields = ['total_videos', 'avg_affinity_score', 'like_rate', 'avg_rating', 'avg_sentiment']
        if sort_by not in valid_sort_fields:
            sort_by = 'total_videos'
        
        conn = sqlite3.connect(self.db_path)
        
        query = f"""
            SELECT 
                user_id,
                total_videos,
                avg_affinity_score,
                avg_rating,
                like_rate,
                avg_sentiment,
                last_updated
            FROM user_statistics
            ORDER BY {sort_by} DESC
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        
        return df
    
    def find_similar_users(self, user_id: int, limit: int = 10) -> pd.DataFrame:
        """
        查找相似用户（基于喜爱度模式）
        
        Args:
            user_id: 目标用户ID
            limit: 返回用户数限制
            
        Returns:
            相似用户DataFrame
        """
        conn = sqlite3.connect(self.db_path)
        
        # 获取目标用户的平均喜爱度特征
        query = """
            SELECT 
                u2.user_id,
                u2.avg_affinity_score,
                u2.avg_rating,
                u2.like_rate,
                u2.avg_sentiment,
                ABS(u1.avg_affinity_score - u2.avg_affinity_score) + 
                ABS(u1.avg_rating - u2.avg_rating) + 
                ABS(u1.like_rate - u2.like_rate) + 
                ABS(u1.avg_sentiment - u2.avg_sentiment) as similarity_distance
            FROM user_statistics u1
            CROSS JOIN user_statistics u2
            WHERE u1.user_id = ? AND u2.user_id != ?
            ORDER BY similarity_distance ASC
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(user_id, user_id, limit))
        conn.close()
        
        return df
    
    def get_affinity_distribution(self, bins: int = 10) -> pd.DataFrame:
        """
        获取喜爱度分布统计
        
        Args:
            bins: 分箱数量
            
        Returns:
            分布统计DataFrame
        """
        conn = sqlite3.connect(self.db_path)
        
        # 创建分箱查询
        query = f"""
            SELECT 
                ROUND(affinity_score * {bins}) / {bins} as affinity_range,
                COUNT(*) as count,
                COUNT(*) * 100.0 / (SELECT COUNT(*) FROM user_video_affinity) as percentage
            FROM user_video_affinity
            GROUP BY ROUND(affinity_score * {bins})
            ORDER BY affinity_range
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def search_by_affinity_range(self, min_affinity: float, max_affinity: float, limit: int = 100) -> pd.DataFrame:
        """
        按喜爱度范围搜索
        
        Args:
            min_affinity: 最小喜爱度
            max_affinity: 最大喜爱度
            limit: 返回记录数限制
            
        Returns:
            搜索结果DataFrame
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                user_id,
                video_id,
                affinity_score,
                rating_score,
                like_score,
                sentiment_score,
                engagement_score
            FROM user_video_affinity
            WHERE affinity_score BETWEEN ? AND ?
            ORDER BY affinity_score DESC
            LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(min_affinity, max_affinity, limit))
        conn.close()
        
        return df
    
    def export_user_data(self, user_id: int, output_file: str):
        """
        导出用户的完整数据
        
        Args:
            user_id: 用户ID
            output_file: 输出文件路径
        """
        df = self.query_user_preferences(user_id, limit=10000)
        df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"[green]用户 {user_id} 的数据已导出到: {output_file}[/green]")
    
    def export_video_data(self, video_id: int, output_file: str):
        """
        导出视频的完整数据
        
        Args:
            video_id: 视频ID
            output_file: 输出文件路径
        """
        df = self.query_video_popularity(video_id, limit=10000)
        df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"[green]视频 {video_id} 的数据已导出到: {output_file}[/green]")

def main():
    """
    命令行交互界面
    """
    parser = argparse.ArgumentParser(description='视频喜爱度数据查询工具')
    parser.add_argument('--db', default='data.db', help='数据库文件路径')
    parser.add_argument('--user', type=int, help='查询指定用户的偏好')
    parser.add_argument('--video', type=int, help='查询指定视频的受欢迎程度')
    parser.add_argument('--top-videos', type=int, help='显示最受欢迎的视频数量')
    parser.add_argument('--top-users', type=int, help='显示最活跃的用户数量')
    parser.add_argument('--similar-users', type=int, help='查找与指定用户相似的用户')
    parser.add_argument('--info', action='store_true', help='显示数据库基本信息')
    parser.add_argument('--distribution', action='store_true', help='显示喜爱度分布')
    
    args = parser.parse_args()
    
    try:
        query_tool = VideoAffinityQuery(args.db)
        
        if args.info:
            info = query_tool.get_database_info()
            console.print(Panel("[bold cyan]数据库信息[/bold cyan]", style="cyan"))
            console.print(f"[green]- 总记录数: {info['total_records']:,}[/green]")
            console.print(f"[green]- 视频数量: {info['total_videos']:,}[/green]")
            console.print(f"[green]- 用户数量: {info['total_users']:,}[/green]")
            console.print(f"[green]- 喜爱度范围: {info['affinity_stats']['min']:.4f} - {info['affinity_stats']['max']:.4f}[/green]")
            console.print(f"[green]- 平均喜爱度: {info['affinity_stats']['avg']:.4f}[/green]")
        
        if args.user:
            console.print(Panel(f"[bold yellow]用户 {args.user} 的视频偏好[/bold yellow]", style="yellow"))
            df = query_tool.query_user_preferences(args.user)
            console.print(df.to_string(index=False))
        
        if args.video:
            console.print(Panel(f"[bold blue]视频 {args.video} 的用户喜爱度[/bold blue]", style="blue"))
            df = query_tool.query_video_popularity(args.video)
            console.print(df.to_string(index=False))
        
        if args.top_videos:
            console.print(Panel(f"[bold green]最受欢迎的 {args.top_videos} 个视频[/bold green]", style="green"))
            df = query_tool.get_top_videos(args.top_videos)
            console.print(df.to_string(index=False))
        
        if args.top_users:
            console.print(Panel(f"[bold magenta]最活跃的 {args.top_users} 个用户[/bold magenta]", style="magenta"))
            df = query_tool.get_top_users(args.top_users)
            console.print(df.to_string(index=False))
        
        if args.similar_users:
            console.print(Panel(f"[bold cyan]与用户 {args.similar_users} 相似的用户[/bold cyan]", style="cyan"))
            df = query_tool.find_similar_users(args.similar_users)
            console.print(df.to_string(index=False))
        
        if args.distribution:
            console.print(Panel("[bold red]喜爱度分布[/bold red]", style="red"))
            df = query_tool.get_affinity_distribution()
            console.print(df.to_string(index=False))
        
        # 如果没有指定任何参数，显示交互式菜单
        if not any([args.info, args.user, args.video, args.top_videos, 
                   args.top_users, args.similar_users, args.distribution]):
            interactive_menu(query_tool)
    
    except Exception as e:
        logger.error(f"[red]错误: {e}[/red]")

def interactive_menu(query_tool: VideoAffinityQuery):
    """
    交互式菜单
    """
    while True:
        console.print(Panel("[bold blue]视频喜爱度数据查询工具[/bold blue]", style="blue"))
        
        table = Table(title="功能菜单", show_header=False)
        table.add_column("选项", style="cyan", width=4)
        table.add_column("功能", style="green")
        
        table.add_row("1", "显示数据库信息")
        table.add_row("2", "查询用户偏好")
        table.add_row("3", "查询视频受欢迎程度")
        table.add_row("4", "显示最受欢迎的视频")
        table.add_row("5", "显示最活跃的用户")
        table.add_row("6", "查找相似用户")
        table.add_row("7", "显示喜爱度分布")
        table.add_row("8", "按喜爱度范围搜索")
        table.add_row("9", "导出用户数据")
        table.add_row("10", "导出视频数据")
        table.add_row("0", "退出")
        
        console.print(table)
        
        choice = input("\n请选择功能 (0-10): ").strip()
        
        try:
            if choice == '0':
                console.print("[green]感谢使用！[/green]")
                break
            elif choice == '1':
                info = query_tool.get_database_info()
                console.print(Panel("[bold cyan]数据库信息[/bold cyan]", style="cyan"))
                console.print(f"[green]- 总记录数: {info['total_records']:,}[/green]")
                console.print(f"[green]- 视频数量: {info['total_videos']:,}[/green]")
                console.print(f"[green]- 用户数量: {info['total_users']:,}[/green]")
                console.print(f"[green]- 喜爱度范围: {info['affinity_stats']['min']:.4f} - {info['affinity_stats']['max']:.4f}[/green]")
                console.print(f"[green]- 平均喜爱度: {info['affinity_stats']['avg']:.4f}[/green]")
            
            elif choice == '2':
                user_id = int(input("请输入用户ID: "))
                limit = int(input("显示记录数 (默认20): ") or "20")
                df = query_tool.query_user_preferences(user_id, limit)
                console.print(Panel(f"[bold yellow]用户 {user_id} 的视频偏好[/bold yellow]", style="yellow"))
                console.print(df.to_string(index=False))
            
            elif choice == '3':
                video_id = int(input("请输入视频ID: "))
                limit = int(input("显示记录数 (默认20): ") or "20")
                df = query_tool.query_video_popularity(video_id, limit)
                console.print(Panel(f"[bold blue]视频 {video_id} 的用户喜爱度[/bold blue]", style="blue"))
                console.print(df.to_string(index=False))
            
            elif choice == '4':
                limit = int(input("显示视频数量 (默认10): ") or "10")
                sort_by = input("排序字段 (avg_affinity_score/total_users/like_rate, 默认avg_affinity_score): ") or "avg_affinity_score"
                df = query_tool.get_top_videos(limit, sort_by)
                console.print(Panel(f"[bold green]最受欢迎的 {limit} 个视频[/bold green]", style="green"))
                console.print(df.to_string(index=False))
            
            elif choice == '5':
                limit = int(input("显示用户数量 (默认10): ") or "10")
                sort_by = input("排序字段 (total_videos/avg_affinity_score/like_rate, 默认total_videos): ") or "total_videos"
                df = query_tool.get_top_users(limit, sort_by)
                console.print(Panel(f"[bold magenta]最活跃的 {limit} 个用户[/bold magenta]", style="magenta"))
                console.print(df.to_string(index=False))
            
            elif choice == '6':
                user_id = int(input("请输入用户ID: "))
                limit = int(input("显示相似用户数量 (默认10): ") or "10")
                df = query_tool.find_similar_users(user_id, limit)
                console.print(Panel(f"[bold cyan]与用户 {user_id} 相似的用户[/bold cyan]", style="cyan"))
                console.print(df.to_string(index=False))
            
            elif choice == '7':
                bins = int(input("分箱数量 (默认10): ") or "10")
                df = query_tool.get_affinity_distribution(bins)
                console.print(Panel("[bold red]喜爱度分布[/bold red]", style="red"))
                console.print(df.to_string(index=False))
            
            elif choice == '8':
                min_affinity = float(input("最小喜爱度 (0-1): "))
                max_affinity = float(input("最大喜爱度 (0-1): "))
                limit = int(input("显示记录数 (默认50): ") or "50")
                df = query_tool.search_by_affinity_range(min_affinity, max_affinity, limit)
                console.print(Panel(f"[bold green]喜爱度在 {min_affinity}-{max_affinity} 范围内的记录[/bold green]", style="green"))
                console.print(df.to_string(index=False))
            
            elif choice == '9':
                user_id = int(input("请输入用户ID: "))
                output_file = input("输出文件名 (默认user_data.csv): ") or "user_data.csv"
                query_tool.export_user_data(user_id, output_file)
            
            elif choice == '10':
                video_id = int(input("请输入视频ID: "))
                output_file = input("输出文件名 (默认video_data.csv): ") or "video_data.csv"
                query_tool.export_video_data(video_id, output_file)
            
            else:
                logger.warning("[yellow]无效选择，请重新输入。[/yellow]")
        
        except ValueError as e:
            logger.error(f"[red]输入错误: {e}[/red]")
        except Exception as e:
            logger.error(f"[red]操作失败: {e}[/red]")
        
        input("\n按回车键继续...")

if __name__ == "__main__":
    main()