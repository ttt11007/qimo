# install_chinese_font.py
import matplotlib
import os
import sys

def install_chinese_font():
    """安装中文字体到Matplotlib字体目录"""
    
    # 检查是否在Streamlit Cloud上
    is_streamlit_cloud = 'STREAMLIT_SHARING_MODE' in os.environ
    
    if is_streamlit_cloud:
        print("检测到Streamlit Cloud环境，安装中文字体...")
        
        # 下载中文字体文件
        import requests
        
        # 下载微软雅黑字体
        font_url = "https://github.com/adobe-fonts/source-han-sans/raw/release/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf"
        font_path = os.path.join(os.path.expanduser("~"), ".fonts", "SourceHanSansSC-Regular.otf")
        
        # 创建字体目录
        os.makedirs(os.path.dirname(font_path), exist_ok=True)
        
        try:
            # 下载字体文件
            response = requests.get(font_url)
            with open(font_path, 'wb') as f:
                f.write(response.content)
            
            print(f"已下载字体到: {font_path}")
            
            # 更新Matplotlib字体缓存
            matplotlib.font_manager._rebuild()
            print("Matplotlib字体缓存已更新")
            
            return True
        except Exception as e:
            print(f"字体下载失败: {e}")
            return False
    else:
        print("本地环境，使用系统字体")
        return True

if __name__ == "__main__":
    install_chinese_font()
