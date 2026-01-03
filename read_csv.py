import pandas as pd

# 设置输出右对齐，防止中文不对齐
pd.set_option('display.unicode.east_asian_width', True)

# 读取数据集，并指定字符编码为gbk，防止中文报错
insurance_df = pd.read_csv('student_data_adjusted_rounded.csv')

# 输出数据框的前5行
print("输出数据框的前5行记录，如下")
print(insurance_df.head())
print()  # 换行分割

# 查看数据框的各列信息
print("输出数据框的各列的详细信息，如下：")
insurance_df.info()
