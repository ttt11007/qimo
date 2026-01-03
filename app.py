import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
import json
from datetime import datetime

with st.sidebar:
    st.title('导航菜单')
    page = st.radio(        
        "选择页面",
        ("项目介绍", "专业数据分析", "成绩预测")
    )
if page == "项目介绍":
    st.title("学生成绩分析与预测系统")
    st.divider()
    top1, top2 = st.columns(2)
    with top1:
        st.header("项目概述")
        st.text("本项目是一个基于streamlit的学生成绩分析平台，通过数据可视化和机器学习技术，帮助教育工作者和学生深入了解学业表现，并预测期末考试成绩")
        st.subheader("主要特点：")
        st.text("· 数据可视化：多维度展示学生学业数据")
        st.text("· 专业分析：按专业分类的详细统计分析")
        st.text("· 学习建议：根据预测结果提供个性化反馈")
    with top2:       
        image_url =[
            {
                'url':'1.png'                
                },
            {
                'url':'2.png'                
                },
            {
                'url':'3.png'                
                },

            ]
        if 'ind' not in st.session_state:
            st.session_state['ind']=0

        st.image(image_url[st.session_state['ind']]['url'])

        c1,c2=st.columns(2)

        def lastImg():
            st.session_state['ind']=(st.session_state['ind']-1) % len(image_url)

        def nextImg():
            st.session_state['ind']=(st.session_state['ind']+1) % len(image_url)

        with c1:
            st.button('上一张',use_container_width=True,on_click=lastImg)

        with c2:
            st.button('下一张',use_container_width=True,on_click=nextImg)

    st.divider()
    
    st.header("项目目标")
    cen1, cen2, cen3 = st.columns(3)
    with cen1:
        st.subheader("目标一")
        st.text("分析影响因素")
        st.text("· 识别关键学习指标")
        st.text("· 探索成绩相关因素")
        st.text("· 提供数据支持决策")
    with cen2:
        st.subheader("目标二")
        st.text("可视化展示")
        st.text("· 专业对比分析")
        st.text("· 性别差异研究")
        st.text("· 学习模式识别")
    with cen3:
        st.subheader("目标三")
        st.text("成绩预测")
        st.text("· 机器学习模型")
        st.text("· 个性化预测")
        st.text("· 及时干预预警")

    st.divider()
    
    st.header("技术架构")
    btm1, btm2, btm3, btm4 = st.columns(4)
    with btm1:
        st.text("前端框架")
        web= '''Streamlit'''
        st.code(web)
    with btm2:        
        st.text("数据处理")
        sj= '''Pandas
        NumPy
        '''
        st.code(sj)
    with btm3:        
        st.text("可视化")
        ksh= '''Plotly
        Natplotib
        '''
        st.code(ksh)
    with btm4:        
        st.text("机器学习")
        jq= '''Scikit-learn'''
        st.code(jq)

elif page == "专业数据分析":
    st.title("专业数据分析")
    st.subheader("1.各专业男女性别比例：")
        # 设置中文字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    try:
        # 直接读取数据文件
        df = pd.read_csv("student_data_adjusted_rounded.csv")    
        
    except FileNotFoundError:
        st.error("未找到数据文件: student_data_adjusted_rounded.csv")
        st.info("请确保 student_data_adjusted_rounded.csv 文件与当前脚本在同一目录下")
        st.stop()

    # 检查必要的列是否存在
    required_columns = ["学号", "性别", "专业"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"数据文件缺少必要的列: {', '.join(missing_columns)}")
        st.stop()

    # 数据清洗和准备
    df["性别"] = df["性别"].astype(str).str.strip()

    # 计算各专业的性别比例
    gender_counts = df.groupby(["专业", "性别"]).size().unstack(fill_value=0)

    # 确保有"男"和"女"列
    if "男" not in gender_counts.columns:
        gender_counts["男"] = 0
    if "女" not in gender_counts.columns:
        gender_counts["女"] = 0

    # 计算比例
    gender_counts["男比例"] = gender_counts["男"] / (gender_counts["男"] + gender_counts["女"])
    gender_counts["女比例"] = gender_counts["女"] / (gender_counts["男"] + gender_counts["女"])

    # 重置索引并排序
    gender_counts = gender_counts.reset_index()
    gender_counts = gender_counts.sort_values("专业")

    # 创建两列布局
    col1, col2 = st.columns([2, 1])

    with col1:
            
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 设置条形图位置
        x = np.arange(len(gender_counts))
        width = 0.35
        
        # 绘制男性比例条形
        bars1 = ax.bar(x - width/2, gender_counts["男比例"], width, 
                       color='#1E90FF', label='男')
        
        # 绘制女性比例条形
        bars2 = ax.bar(x + width/2, gender_counts["女比例"], width, 
                       color='#FF69B4', label='女')
        
        # 设置x轴标签
        ax.set_xticks(x)
        ax.set_xticklabels(gender_counts["专业"], fontsize=10)
        
        # 设置y轴
        ax.set_ylabel('比例')
        ax.set_ylim(0, 1.0)
        
        # 添加图例
        ax.legend()
        
        # 在条形上显示数值
        for i, (male, female) in enumerate(zip(gender_counts["男比例"], gender_counts["女比例"])):
            ax.text(i - width/2, male + 0.01, f'{male:.1%}', 
                    ha='center', va='bottom', fontsize=9)
            ax.text(i + width/2, female + 0.01, f'{female:.1%}', 
                    ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
            
        # 创建百分比显示的数据框
        display_df = gender_counts[["专业", "男", "女", "男比例", "女比例"]].copy()
        display_df["男比例"] = display_df["男比例"].apply(lambda x: f"{x:.2%}")
        display_df["女比例"] = display_df["女比例"].apply(lambda x: f"{x:.2%}")
        
        # 显示数据表
        st.dataframe(display_df, hide_index=True, use_container_width=True)
        
        # 添加统计信息
        total_students = gender_counts["男"].sum() + gender_counts["女"].sum()
        total_male = gender_counts["男"].sum()
        total_female = gender_counts["女"].sum()
        
    st.divider()

    st.subheader("2.各专业学习指标对比：")
        
    st.markdown('各专业期中期末成绩趋势')
            # 设置中文字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    try:
        # 直接读取数据文件
        df = pd.read_csv("student_data_adjusted_rounded.csv")
        
    except FileNotFoundError:
        st.error("未找到数据文件: student_data_adjusted_rounded.csv")
        st.info("请确保 student_data_adjusted_rounded.csv 文件与当前脚本在同一目录下")
        st.stop()

    # 检查必要的列是否存在
    required_columns = ["专业", "期中考试分数", "期末考试分数", "每周学习时长（小时）"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"数据文件缺少必要的列: {', '.join(missing_columns)}")
        st.stop()

    # 数据清洗和准备
    numeric_cols = ["期中考试分数", "期末考试分数", "每周学习时长（小时）"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 计算各专业的平均值
    major_stats = df.groupby("专业")[numeric_cols].mean().reset_index()

    # 按专业名称排序
    major_stats = major_stats.sort_values("专业")

    # 创建一行两列的布局
    col1, col2 = st.columns([2, 1])

    with col1:
        # 创建双Y轴折线图    
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # 设置x轴位置
        x = range(len(major_stats))
        
        # 绘制期中考试分数折线（使用左侧Y轴）
        ax1.plot(x, major_stats["期中考试分数"], marker='o', color='#FF6B6B', 
                linewidth=2, markersize=8, label='期中考试分数')
        
        # 绘制期末考试分数折线（使用左侧Y轴）
        ax1.plot(x, major_stats["期末考试分数"], marker='s', color='#4ECDC4', 
                linewidth=2, markersize=8, label='期末考试分数')
        
        # 设置左侧Y轴（分数）
        ax1.set_xlabel('专业', fontsize=12)
        ax1.set_ylabel('分数', fontsize=12, color='black')
        
        # 设置x轴刻度为专业名称
        ax1.set_xticks(x)
        ax1.set_xticklabels(major_stats["专业"], fontsize=10, rotation=45, ha='right')
        
        # 创建右侧Y轴用于每周学习时长
        ax2 = ax1.twinx()
        
        # 绘制每周学习时长折线（使用右侧Y轴）
        ax2.plot(x, major_stats["每周学习时长（小时）"], marker='^', color='#45B7D1', 
                linewidth=2, markersize=8, label='每周学习时长（小时）')
        
        # 设置右侧Y轴（学习时长）
        ax2.set_ylabel('学习时长（小时）', fontsize=12, color='#45B7D1')
        ax2.tick_params(axis='y', labelcolor='#45B7D1')
        
        # 添加网格
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 添加图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        # 显示数据表    
        display_df = major_stats.copy()
        display_df["期中考试分数"] = display_df["期中考试分数"].apply(lambda x: f"{x:.2f}")
        display_df["期末考试分数"] = display_df["期末考试分数"].apply(lambda x: f"{x:.2f}")
        display_df["每周学习时长（小时）"] = display_df["每周学习时长（小时）"].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(
            display_df,
            column_config={
                "专业": st.column_config.TextColumn("专业", width="medium"),
                "期中考试分数": st.column_config.TextColumn("期中考试分数", width="small"),
                "期末考试分数": st.column_config.TextColumn("期末考试分数", width="small"),
                "每周学习时长（小时）": st.column_config.TextColumn("每周学习时长", width="small")
            },
            hide_index=True,
            use_container_width=True
        )
        
    st.divider()

    st.subheader("3.各专业出勤率分析：")
        
         #设置中文字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    try:
        # 直接读取数据文件
        df = pd.read_csv("student_data_adjusted_rounded.csv")
           
    except FileNotFoundError:
        st.error("未找到数据文件: student_data_adjusted_rounded.csv")
        st.info("请确保 student_data_adjusted_rounded.csv 文件与当前脚本在同一目录下")
        st.stop()

    # 检查必要的列是否存在
    required_columns = ["专业", "上课出勤率"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"数据文件缺少必要的列: {', '.join(missing_columns)}")
        st.stop()

    # 数据清洗和准备
    # 确保上课出勤率是数字类型
    df["上课出勤率"] = pd.to_numeric(df["上课出勤率"], errors='coerce')

    # 计算各专业的平均上课出勤率
    attendance_stats = df.groupby("专业")["上课出勤率"].mean().reset_index()
    attendance_stats.columns = ["专业", "平均出勤率"]

    # 按平均出勤率从高到低排序
    attendance_stats = attendance_stats.sort_values("平均出勤率", ascending=False)

    # 创建一行两列的布局
    col1, col2 = st.columns([2, 1])

    with col1:
        # 创建条形图       
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')
        fig.patch.set_alpha(0.0)
        
        # 设置条形图位置
        x = np.arange(len(attendance_stats))
        bars = ax.bar(x, attendance_stats["平均出勤率"], color='#FF9999', edgecolor='white', linewidth=1.2)
        
        # 设置x轴标签 - 使用自适应颜色
        ax.set_xticks(x)
        ax.set_xticklabels(attendance_stats["专业"], fontsize=10, rotation=45, ha='right', color='white')
        
        # 设置x轴标签颜色
        ax.set_xlabel('专业', fontsize=12, color='white')
        
        # 设置y轴 - 使用自适应颜色
        ax.set_ylabel('平均出勤率', fontsize=12, color='white')
        ax.set_ylim(0, 1.0)
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        
        # 设置刻度标签颜色为白色以适应深色背景
        ax.tick_params(axis='y', colors='white')
        ax.tick_params(axis='x', colors='white')
        
        # 设置坐标轴背景为透明
        ax.set_facecolor('none')
        
        # 设置坐标轴颜色为白色
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # 在条形上显示数值 - 使用深色文字确保在浅色条形上可见
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='black')
        
        # 添加网格 - 使用浅灰色确保在深色背景下可见
        ax.grid(axis='y', alpha=0.2, linestyle='--', color='lightgray')
        
        plt.tight_layout()
        st.pyplot(fig)

    with col2:  
            
        # 格式化显示，保留4位小数
        display_df = attendance_stats.copy()
        display_df["平均出勤率"] = display_df["平均出勤率"].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(
            display_df,
            column_config={
                "专业": st.column_config.TextColumn("专业", width="medium"),
                "平均出勤率": st.column_config.TextColumn("平均出勤率", width="small")
            },
            hide_index=True,
            use_container_width=True
        )
        
    st.divider()

    st.subheader("4.大数据管理专业专项分析：")

        # 设置中文字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False

    try:
        # 直接读取数据文件
        df = pd.read_csv("student_data_adjusted_rounded.csv")    
        
    except FileNotFoundError:
        st.error("未找到数据文件: student_data_adjusted_rounded.csv")
        st.info("请确保 student_data_adjusted_rounded.csv 文件与当前脚本在同一目录下")
        st.stop()

    # 检查必要的列是否存在
    required_columns = ["专业", "上课出勤率", "期末考试分数", "每周学习时长（小时）"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        st.error(f"数据文件缺少必要的列: {', '.join(missing_columns)}")
        st.stop()

    # 筛选大数据管理专业的数据
    big_data_df = df[df["专业"] == "大数据管理"].copy()

    if big_data_df.empty:
        st.error("数据文件中没有找到大数据管理专业的数据")
        st.stop()

    # 数据清洗和准备
    numeric_cols = ["上课出勤率", "期末考试分数", "每周学习时长（小时）"]
    for col in numeric_cols:
        big_data_df[col] = pd.to_numeric(big_data_df[col], errors='coerce')

    # 计算各项指标
    # 1. 平均出勤率
    avg_attendance = big_data_df["上课出勤率"].mean()

    # 2. 平均期末分数
    avg_final_score = big_data_df["期末考试分数"].mean()

    # 3. 通过率 (假设60分及以上为通过)
    pass_rate = (big_data_df["期末考试分数"] >= 60).mean()

    # 4. 平均学习时长 (每周学习时长)
    avg_study_hours = big_data_df["每周学习时长（小时）"].mean()

    # 创建四个指标卡片
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "平均出勤率",
            f"{avg_attendance:.1%}",
            help="大数据管理专业学生的平均上课出勤率"
        )

    with col2:
        st.metric(
            "平均期末分数",
            f"{avg_final_score:.1f}分",
            help="大数据管理专业学生的期末考试平均分数"
        )

    with col3:
        st.metric(
            "通过率",
            f"{pass_rate:.1%}",
            help="大数据管理专业学生期末考试通过率(60分及以上)"
        )

    with col4:
        st.metric(
            "平均学习时长",
            f"{avg_study_hours:.1f}小时",
            help="大数据管理专业学生平均每周学习时长"
        )

    # 添加分隔线
    st.markdown("---")

    # 创建期末成绩分布垂直条形图
    st.markdown("大数据管理专业期末成绩分布")

    # 创建两列布局
    chart_col1, chart_col2 = st.columns([2, 1])

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='none')
    fig.patch.set_alpha(0.0)
        
    # 定义分数区间
    bins = [0, 40, 50, 60, 70, 80, 90, 100]
    bin_labels = ['0-40', '41-50', '51-60', '61-70', '71-80', '81-90', '91-100']
        
        # 将分数分到区间
    big_data_df['分数区间'] = pd.cut(big_data_df['期末考试分数'], bins=bins, labels=bin_labels)
        
        # 计算每个区间的人数
    score_distribution = big_data_df['分数区间'].value_counts().sort_index()
        
        # 创建垂直条形图
    bars = ax.bar(score_distribution.index, score_distribution.values, 
                      color='#1f77b4', edgecolor='white', linewidth=1.2, width=0.7)
        
        # 设置坐标轴标签 - 使用自适应颜色
    ax.set_xlabel('期末考试分数', fontsize=12, color='white')
    ax.set_ylabel('人数', fontsize=12, color='white')
        
        # 设置y轴范围
    max_count = score_distribution.values.max()
    ax.set_ylim(0, max_count * 1.1)
        
        # 设置刻度标签颜色为白色以适应深色背景
    ax.tick_params(axis='both', colors='white')
        
        # 设置坐标轴背景为透明
    ax.set_facecolor('none')
        
        # 设置坐标轴颜色为白色
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
        
        # 在条形上显示数值
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max_count * 0.01,
                f'{int(height)}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold', color='white')
        
        # 添加网格
    ax.grid(axis='y', alpha=0.2, linestyle='--', color='lightgray')
        
        # 添加图表标题
    ax.set_title('大数据管理专业期末成绩分布', fontsize=14, color='white')
        
    plt.tight_layout()
    st.pyplot(fig)
    
else:
    st.title("成绩预测")
    # 加载模型和编码器
    @st.cache_resource
    def load_model():
        try:
            with open('model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            return model_data
        except FileNotFoundError:
            st.error("未找到模型文件: model.pkl")
            st.info("请先运行 train_model.py 生成模型文件")
            return None

    model_data = load_model()

    if model_data is None:
        st.stop()

    # 从模型中提取组件
    model = model_data['model']
    scaler = model_data['scaler']
    label_encoder_gender = model_data['label_encoder_gender']
    label_encoder_major = model_data['label_encoder_major']
    feature_columns = model_data['feature_columns']

    # 获取编码器中的类别
    gender_options = list(label_encoder_gender.classes_)
    major_options = list(label_encoder_major.classes_)

    # 创建两列布局
    col_left, col_right = st.columns(2)

    with col_left:
        # 学生基本信息
        st.subheader("📋 学生基本信息")
        
        # 学号
        student_id = st.text_input("学号", value="20230001", 
                                   help="请输入学生的学号")
        
        # 性别
        gender = st.selectbox("性别", gender_options, index=0,
                              help="请选择学生的性别")
        
        # 专业
        major = st.selectbox("专业", major_options, index=0,
                             help="请选择学生的专业")

    with col_right:
        # 学习成绩信息
        st.subheader("📊 学习成绩信息")
        
        # 每周学习时长
        study_hours = st.slider(
            "每周学习时长(小时)", 
            min_value=0, max_value=40, value=15,
            help="学生每周的平均学习时长"
        )
        
        # 上课出勤率
        attendance_rate = st.slider(
            "上课出勤率(%)", 
            min_value=0, max_value=100, value=90,
            help="学生的上课出勤率"
        )
        
        # 期中考试分数
        midterm_score = st.slider(
            "期中考试分数", 
            min_value=0, max_value=100, value=70,
            help="学生的期中考试分数"
        )
        
        # 作业完成率
        homework_rate = st.slider(
            "作业完成率(%)", 
            min_value=0, max_value=100, value=85,
            help="学生的作业完成率"
        )

    # 添加分隔线
    st.markdown("---")

    # 预测按钮
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button(
            "🚀 预测期末成绩", 
            type="primary", 
            use_container_width=True
        )

    # 执行预测
    if predict_button:
        # 显示加载状态
        with st.spinner("正在预测期末成绩..."):
            # 准备输入数据
            try:
                # 编码分类变量
                gender_encoded = label_encoder_gender.transform([gender])[0]
                major_encoded = label_encoder_major.transform([major])[0]
                
                # 创建特征数组
                features = np.array([[study_hours, attendance_rate, midterm_score, 
                                     homework_rate, gender_encoded, major_encoded]])
                
                # 特征标准化
                features_scaled = scaler.transform(features)
                
                # 进行预测
                predicted_score = model.predict(features_scaled)[0]
                
                # 确保预测分数在合理范围内
                predicted_score = max(0, min(100, predicted_score))
                
                # 模拟一下预测过程
                import time
                time.sleep(1.5)
                
            except Exception as e:
                st.error(f"预测过程中出现错误: {str(e)}")
                st.stop()
        
        # 显示预测结果
        st.markdown("---")
        st.subheader("📈 预测结果")
        
        # 创建漂亮的预测结果卡片
        st.markdown(f"""
        预测期末成绩
            {predicted_score:.1f}分      
        
        """, unsafe_allow_html=True)
        
        # 根据预测分数提供学习建议
        st.subheader("🎯 学习建议")
        
        # 创建建议卡片
        if predicted_score >= 85:
            suggestion_card = """
            🌟 优秀表现建议
                    1. 继续保持当前的学习习惯和方法
                    2. 尝试挑战更高难度的学习内容
                    3. 帮助其他同学，教学相长
                    4. 参与学术研究或项目实践，提升综合能力            
            """
            st.markdown(suggestion_card, unsafe_allow_html=True)
            st.balloons()
            
        elif predicted_score >= 70:
            suggestion_card = """
               📈 良好表现建议            
                    1. 巩固已有知识点，建立完整的知识体系
                    2. 针对薄弱环节进行专项练习
                    3. 多与老师交流，获取个性化指导
                    4. 参与小组学习，互相促进提高       
            """
            st.markdown(suggestion_card, unsafe_allow_html=True)
            
        elif predicted_score >= 60:
            suggestion_card = """
            ⚠️ 需努力提升建议            
                    1. 制定详细的学习计划，合理安排时间
                    2. 优先掌握核心知识点和考试重点
                    3. 及时向老师和同学求助，解决疑难问题
                    4. 增加学习时间，全力冲刺及格线            
            """
            st.markdown(suggestion_card, unsafe_allow_html=True)
            
        else:
            suggestion_card = """
            🚨 急需改进建议
                    1. ⚠️ 立即制定紧急学习计划，增加学习时间
                    2. 📚 重点复习基础知识，确保掌握核心概念
                    3. 👨‍🏫 主动联系老师，寻求一对一辅导
                    4. 👥 加入学习小组，向优秀同学请教
                    5. 📝 多做练习题，提高解题能力
                    6. 🕐 合理安排时间，避免临时抱佛脚
            
            """
            st.markdown(suggestion_card, unsafe_allow_html=True)
        

        
        # 添加个性化改进建议
        st.markdown("#### 💡 个性化改进建议")
        
        improvement_list = []
        
        if midterm_score < 60:
            improvement_list.append("1. **期中成绩较低**：建议重点复习期中考试内容，确保掌握基础知识")
        
        if study_hours < 15:
            improvement_list.append("2. **学习时长不足**：建议增加每周学习时间至15小时以上")
        
        if attendance_rate < 90:
            improvement_list.append("3. **出勤率有待提高**：确保按时上课，课堂学习非常重要")
        
        if homework_rate < 85:
            improvement_list.append("4. **作业完成率不足**：认真完成每次作业，这是巩固知识的重要方式")
        
        if predicted_score < 60:
            improvement_list.append("5. **紧急行动计划**：制定每日学习计划，优先复习重点章节")
        
        if not improvement_list:
            improvement_list.append("**当前学习状态良好，继续保持！**")
        
        for item in improvement_list:
            st.write(item)
        
        # 添加预测时间戳
        st.caption(f"预测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")



