# -*- coding: utf-8 -*-
"""
Web演示界面启动脚本
"""

import http.server
import socketserver
import os
import webbrowser
import threading
import time
import logging
from rich.logging import RichHandler
from rich.console import Console

# 配置日志
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[RichHandler(console=console, rich_tracebacks=True)]
)
logger = logging.getLogger(__name__)

def start_web_server(port=8080):
    """
    启动Web服务器
    """
    # 切换到项目根目录
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            super().end_headers()
    
    with socketserver.TCPServer(("", port), CustomHTTPRequestHandler) as httpd:
        logger.info(f"🌐 Web演示服务器启动成功")
        logger.info(f"📍 访问地址: http://localhost:{port}/static/recom_form.html")
        logger.info(f"📁 静态文件目录: {os.getcwd()}/static")
        logger.info("\n按 Ctrl+C 停止服务器")
        
        # 延迟打开浏览器
        def open_browser():
            time.sleep(2)
            webbrowser.open(f'http://localhost:{port}/static/recom_form.html')
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            logger.info("\n👋 Web服务器已停止")

if __name__ == "__main__":
    start_web_server()