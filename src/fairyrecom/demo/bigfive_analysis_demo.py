#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BigFive性格相似度分析演示程序

功能:
1. 加载BigFive性格测试数据
2. 计算用户间的性格相似度
3. 找到最相似的用户
4. 生成性格分析报告
5. 可视化分析结果

使用方法:
python demo_bigfive_analysis.py
"""

import sys
import os
import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
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

# 添加src目录到Python路径
src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from fairyrecom.analysis.bigfive_analyzer import BigFiveSimilarityAnalyzer
except ImportError as e:
    logger.error(f"[red]导入错误: {e}[/red]")
    logger.error("[red]请确保项目结构正确[/red]")
    sys.exit(1)

import pandas as pd
import numpy as np

def interactive_analysis():
    """
    交互式分析功能
    """
    console.print(Panel("[bold blue]交互式BigFive性格相似度分析[/bold blue]", expand=False))
    
    # 创建分析器
    analyzer = BigFiveSimilarityAnalyzer()
    
    # 加载数据
    logger.info("[cyan]正在加载数据...[/cyan]")
    data = analyzer.load_data()
    if data is None:
        logger.error("[red]数据加载失败，请检查文件路径[/red]")
        return
    
    while True:
        menu_table = Table(title="功能菜单", show_header=False)
        menu_table.add_column("选项", style="cyan", width=3)
        menu_table.add_column("功能", style="white")
        
        menu_table.add_row("1", "查看数据概览")
        menu_table.add_row("2", "分析指定用户的性格特征")
        menu_table.add_row("3", "找到最相似的用户")
        menu_table.add_row("4", "生成完整分析报告")
        menu_table.add_row("5", "可视化性格特征雷达图")
        menu_table.add_row("6", "可视化相似度热力图")
        menu_table.add_row("7", "批量相似度分析")
        menu_table.add_row("0", "退出")
        
        console.print(menu_table)
        
        choice = input("\n请输入选择 (0-7): ").strip()
        
        if choice == '0':
            console.print("[green]感谢使用！[/green]")
            break
        elif choice == '1':
            show_data_overview(data)
        elif choice == '2':
            analyze_user_personality(analyzer, data)
        elif choice == '3':
            find_similar_users(analyzer, data)
        elif choice == '4':
            generate_full_report(analyzer, data)
        elif choice == '5':
            visualize_personality_radar(analyzer, data)
        elif choice == '6':
            visualize_similarity_heatmap(analyzer, data)
        elif choice == '7':
            batch_similarity_analysis(analyzer, data)
        else:
            logger.warning("[yellow]无效选择，请重新输入[/yellow]")

def show_data_overview(data):
    """
    显示数据概览
    """
    console.print(Panel("[bold green]数据概览[/bold green]", expand=False))
    
    # 基本信息表格
    info_table = Table(title="基本信息", show_header=False)
    info_table.add_column("项目", style="cyan")
    info_table.add_column("值", style="white")
    
    info_table.add_row("总用户数", str(len(data)))
    info_table.add_row("用户ID范围", f"{data['user_id'].min()} - {data['user_id'].max()}")
    
    console.print(info_table)
    
    # 统计各问题的答案分布
    stats_table = Table(title="问题答案分布统计")
    stats_table.add_column("问题", style="cyan")
    stats_table.add_column("均值", style="green")
    stats_table.add_column("标准差", style="yellow")
    stats_table.add_column("范围", style="blue")
    
    for i in range(1, 16):
        col = f'Q{i}'
        if col in data.columns:
            stats_table.add_row(
                f"Q{i}",
                f"{data[col].mean():.2f}",
                f"{data[col].std():.2f}",
                f"[{data[col].min()}-{data[col].max()}]"
            )
    
    console.print(stats_table)
    
    # 显示前5个用户的数据
    console.print("\n[bold]前5个用户的数据样例:[/bold]")
    rprint(data.head())

def analyze_user_personality(analyzer, data):
    """
    分析指定用户的性格特征
    """
    try:
        user_id = int(input("\n请输入要分析的用户ID: "))
        if user_id not in data['user_id'].values:
            logger.error(f"[red]用户ID {user_id} 不存在[/red]")
            return
        
        profile = analyzer.analyze_personality_profile(user_id)
        
        console.print(Panel(f"[bold cyan]用户 {user_id} 性格特征分析[/bold cyan]", expand=False))
        
        # 性格特征表格
        personality_table = Table(title="性格特征")
        personality_table.add_column("维度", style="cyan")
        personality_table.add_column("分数", style="green")
        personality_table.add_column("等级", style="yellow")
        
        for dimension, score in profile.items():
            level = get_score_level(score)
            personality_table.add_row(dimension, f"{score:.2f}/5.0", level)
        
        console.print(personality_table)
        
        # 显示原始答案
        user_data = data[data['user_id'] == user_id].iloc[0]
        answers_table = Table(title="原始答案")
        answers_table.add_column("问题", style="cyan")
        answers_table.add_column("答案", style="white")
        
        for i in range(1, 16):
            answers_table.add_row(f"Q{i}", str(user_data[f'Q{i}']))
        
        console.print(answers_table)
            
    except ValueError:
        logger.error("[red]请输入有效的用户ID[/red]")

def get_score_level(score):
    """
    根据分数获取等级描述
    """
    if score >= 4.0:
        return "很高"
    elif score >= 3.5:
        return "高"
    elif score >= 2.5:
        return "中等"
    elif score >= 2.0:
        return "低"
    else:
        return "很低"

def find_similar_users(analyzer, data):
    """
    找到最相似的用户
    """
    try:
        user_id = int(input("\n请输入目标用户ID: "))
        if user_id not in data['user_id'].values:
            logger.error(f"[red]用户ID {user_id} 不存在[/red]")
            return
        
        top_k = int(input("请输入要显示的相似用户数量 (默认5): ") or "5")
        method = input("请选择相似度计算方法 (cosine/euclidean, 默认cosine): ").strip() or "cosine"
        
        similar_users = analyzer.find_most_similar_users(user_id, top_k, method)
        
        console.print(Panel(f"[bold cyan]与用户 {user_id} 最相似的 {top_k} 个用户 ({method}相似度)[/bold cyan]", expand=False))
        
        similar_table = Table(title="相似用户")
        similar_table.add_column("排名", style="cyan")
        similar_table.add_column("用户ID", style="green")
        similar_table.add_column("相似度", style="yellow")
        
        for i, (similar_id, similarity) in enumerate(similar_users, 1):
            similar_table.add_row(str(i), str(similar_id), f"{similarity:.4f}")
        
        console.print(similar_table)
            
    except ValueError:
        logger.error("[red]请输入有效的数值[/red]")

def generate_full_report(analyzer, data):
    """
    生成完整分析报告
    """
    try:
        user_id = int(input("\n请输入要生成报告的用户ID: "))
        if user_id not in data['user_id'].values:
            logger.error(f"[red]用户ID {user_id} 不存在[/red]")
            return
        
        report = analyzer.generate_similarity_report(user_id, top_k=5)
        console.print(Panel("[bold green]分析报告[/bold green]", expand=False))
        console.print(report)
        
        # 询问是否保存报告
        save = input("\n是否保存报告到文件? (y/n): ").strip().lower()
        if save == 'y':
            filename = f"user_{user_id}_similarity_report.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"[green]报告已保存到 {filename}[/green]")
            
    except ValueError:
        logger.error("[red]请输入有效的用户ID[/red]")

def visualize_personality_radar(analyzer, data):
    """
    可视化性格特征雷达图
    """
    try:
        user_id = int(input("\n请输入要可视化的用户ID: "))
        if user_id not in data['user_id'].values:
            logger.error(f"[red]用户ID {user_id} 不存在[/red]")
            return
        
        logger.info("[cyan]正在生成雷达图...[/cyan]")
        analyzer.visualize_personality_profile(user_id)
        
    except ValueError:
        logger.error("[red]请输入有效的用户ID[/red]")
    except Exception as e:
        logger.error(f"[red]可视化失败: {e}[/red]")
        logger.error("[red]请确保已安装matplotlib[/red]")

def visualize_similarity_heatmap(analyzer, data):
    """
    可视化相似度热力图
    """
    try:
        console.print("\n[bold]选择可视化方式:[/bold]")
        
        choice_table = Table(show_header=False)
        choice_table.add_column("选项", style="cyan", width=3)
        choice_table.add_column("方式", style="white")
        
        choice_table.add_row("1", "随机选择20个用户")
        choice_table.add_row("2", "手动输入用户ID列表")
        choice_table.add_row("3", "选择前N个用户")
        
        console.print(choice_table)
        
        choice = input("请选择 (1-3): ").strip()
        
        user_ids = None
        if choice == '2':
            ids_input = input("请输入用户ID列表 (用逗号分隔): ")
            user_ids = [int(x.strip()) for x in ids_input.split(',')]
            # 验证用户ID是否存在
            valid_ids = [uid for uid in user_ids if uid in data['user_id'].values]
            if len(valid_ids) != len(user_ids):
                logger.warning(f"[yellow]警告: 部分用户ID不存在，将使用有效的ID: {valid_ids}[/yellow]")
            user_ids = valid_ids
        elif choice == '3':
            n = int(input("请输入用户数量: "))
            user_ids = data['user_id'].head(n).tolist()
        
        logger.info("[cyan]正在生成热力图...[/cyan]")
        analyzer.visualize_similarity_heatmap(user_ids)
        
    except ValueError:
        logger.error("[red]请输入有效的数值[/red]")
    except Exception as e:
        logger.error(f"[red]可视化失败: {e}[/red]")
        logger.error("[red]请确保已安装matplotlib和seaborn[/red]")

def batch_similarity_analysis(analyzer, data):
    """
    批量相似度分析
    """
    try:
        console.print(Panel("[bold cyan]批量相似度分析[/bold cyan]", expand=False))
        
        # 选择分析的用户数量
        n_users = int(input("请输入要分析的用户数量 (默认10): ") or "10")
        sample_users = data['user_id'].head(n_users).tolist()
        
        logger.info(f"\n[cyan]正在分析前 {n_users} 个用户的相似度...[/cyan]")
        
        # 计算相似度矩阵
        similarity_matrix = analyzer.calculate_cosine_similarity(sample_users)
        
        # 找到最相似的用户对
        max_similarity = 0
        most_similar_pair = None
        
        for i in range(len(sample_users)):
            for j in range(i+1, len(sample_users)):
                sim = similarity_matrix[i][j]
                if sim > max_similarity:
                    max_similarity = sim
                    most_similar_pair = (sample_users[i], sample_users[j])
        
        logger.info(f"\n[green]最相似的用户对: 用户 {most_similar_pair[0]} 和 用户 {most_similar_pair[1]}[/green]")
        logger.info(f"[green]相似度: {max_similarity:.4f}[/green]")
        
        # 显示这两个用户的性格特征对比
        console.print(Panel("[bold yellow]性格特征对比[/bold yellow]", expand=False))
        profile1 = analyzer.analyze_personality_profile(most_similar_pair[0])
        profile2 = analyzer.analyze_personality_profile(most_similar_pair[1])
        
        comparison_table = Table(title="性格特征对比")
        comparison_table.add_column("维度", style="cyan")
        comparison_table.add_column(f"用户{most_similar_pair[0]}", style="green")
        comparison_table.add_column(f"用户{most_similar_pair[1]}", style="blue")
        comparison_table.add_column("差异", style="yellow")
        
        for dim in profile1.keys():
            diff = abs(profile1[dim] - profile2[dim])
            comparison_table.add_row(
                dim,
                f"{profile1[dim]:.2f}",
                f"{profile2[dim]:.2f}",
                f"{diff:.2f}"
            )
        
        console.print(comparison_table)
        
    except ValueError:
        logger.error("[red]请输入有效的数值[/red]")
    except Exception as e:
        logger.error(f"[red]分析失败: {e}[/red]")

def main():
    """
    主函数
    """
    console.print(Panel("[bold blue]BigFive性格相似度分析程序[/bold blue]", expand=False))
    console.print("[blue]=" * 50 + "[/blue]")
    
    # 检查数据文件是否存在
    data_path = "reasoner/dataset/dataset/bigfive.csv"
    if not os.path.exists(data_path):
        logger.error(f"[red]错误: 数据文件 {data_path} 不存在[/red]")
        logger.error("[red]请确保数据文件路径正确[/red]")
        return
    
    try:
        interactive_analysis()
    except KeyboardInterrupt:
        logger.info("\n\n[yellow]程序被用户中断[/yellow]")
    except Exception as e:
        logger.error(f"\n[red]程序运行出错: {e}[/red]")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()