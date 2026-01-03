# train_final_score_model.py - 训练并保存预测期末成绩的模型
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import joblib

print("开始训练期末成绩预测模型...")

try:
    # 1. 读取数据
    df = pd.read_csv("student_data_adjusted_rounded.csv")
    print(f"成功读取数据文件，共 {df.shape[0]} 条记录，{df.shape[1]} 个字段")
    
except FileNotFoundError:
    print("错误: 未找到数据文件: student_data_adjusted_rounded.csv")
    print("请确保 student_data_adjusted_rounded.csv 文件与当前脚本在同一目录下")
    exit(1)

# 2. 检查必要的列是否存在
required_columns = ["性别", "专业", "每周学习时长（小时）", "上课出勤率", "期中考试分数", "作业完成率", "期末考试分数"]
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print(f"错误: 数据文件缺少必要的列: {', '.join(missing_columns)}")
    exit(1)

# 3. 数据清洗和准备
print("正在数据清洗和准备...")

# 确保数值列是数字类型
numeric_cols = ["每周学习时长（小时）", "上课出勤率", "期中考试分数", "作业完成率", "期末考试分数"]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 处理缺失值
print(f"处理前数据量: {df.shape[0]}")
df = df.dropna(subset=numeric_cols + ["性别", "专业"])
print(f"处理后数据量: {df.shape[0]}")

# 4. 对分类变量进行编码
print("正在对分类变量进行编码...")

# 创建标签编码器
label_encoder_gender = LabelEncoder()
label_encoder_major = LabelEncoder()

# 编码性别和专业
df["性别编码"] = label_encoder_gender.fit_transform(df["性别"])
df["专业编码"] = label_encoder_major.fit_transform(df["专业"])

print(f"性别类别: {label_encoder_gender.classes_}")
print(f"专业类别: {label_encoder_major.classes_}")

# 5. 定义特征和目标变量
print("正在准备特征和目标变量...")

# 特征：使用所有可用的特征
feature_columns = ["每周学习时长（小时）", "上课出勤率", "期中考试分数", "作业完成率", "性别编码", "专业编码"]
target_column = "期末考试分数"

X = df[feature_columns]
y = df[target_column]

print(f"特征维度: {X.shape}")
print(f"目标变量统计: 均值={y.mean():.2f}, 标准差={y.std():.2f}, 最小值={y.min():.2f}, 最大值={y.max():.2f}")

# 6. 特征标准化
print("正在标准化特征...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 7. 划分训练集和测试集
print("正在划分训练集和测试集...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

print(f"训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# 8. 训练随机森林回归模型
print("正在训练随机森林回归模型...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X_train, y_train)

# 9. 评估模型
print("正在评估模型性能...")
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 计算各种评估指标
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n" + "="*50)
print("模型性能评估:")
print("="*50)
print(f"训练集 MSE: {train_mse:.4f}")
print(f"测试集 MSE: {test_mse:.4f}")
print("-"*50)
print(f"训练集 MAE: {train_mae:.4f}")
print(f"测试集 MAE: {test_mae:.4f}")
print("-"*50)
print(f"训练集 R²分数: {train_r2:.4f}")
print(f"测试集 R²分数: {test_r2:.4f}")
print("="*50)

# 10. 特征重要性分析
print("\n特征重要性分析:")
feature_importance = pd.DataFrame({
    '特征': feature_columns,
    '重要性': model.feature_importances_
})
feature_importance = feature_importance.sort_values('重要性', ascending=False)
print(feature_importance.to_string(index=False))

# 11. 保存模型和相关对象
print("\n正在保存模型和相关对象...")

# 创建要保存的字典
model_data = {
    'model': model,
    'scaler': scaler,
    'label_encoder_gender': label_encoder_gender,
    'label_encoder_major': label_encoder_major,
    'feature_columns': feature_columns,
    'model_performance': {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae
    }
}

# 保存为pkl文件
with open('model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

# 也可以使用joblib保存（通常对大型模型更有效）
joblib.dump(model_data, 'model.joblib')

print("模型已成功保存为 model.pkl 和 model.joblib")

# 12. 保存编码器信息以便在Streamlit应用中使用
encoder_info = {
    'gender_classes': label_encoder_gender.classes_.tolist(),
    'major_classes': label_encoder_major.classes_.tolist(),
    'feature_columns': feature_columns
}

with open('model_encoders.json', 'w') as f:
    import json
    json.dump(encoder_info, f, ensure_ascii=False, indent=2)

print("编码器信息已保存为 model_encoders.json")

# 13. 创建示例预测函数用于测试
print("\n测试模型预测...")

# 创建一个示例数据
example_data = {
    '每周学习时长（小时）': [15],
    '上课出勤率': [90],
    '期中考试分数': [75],
    '作业完成率': [85],
    '性别': ['男'],
    '专业': ['大数据管理']
}

# 转换示例数据
example_df = pd.DataFrame(example_data)

# 编码分类变量
example_df['性别编码'] = label_encoder_gender.transform(example_df['性别'])
example_df['专业编码'] = label_encoder_major.transform(example_df['专业'])

# 提取特征并标准化
example_features = example_df[feature_columns]
example_scaled = scaler.transform(example_features)

# 进行预测
predicted_score = model.predict(example_scaled)[0]
print(f"示例预测结果: {predicted_score:.2f}分")

print("\n模型训练和保存完成！")
