# app_final_score_predictor.py - Streamlit期末成绩预测应用
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime

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

