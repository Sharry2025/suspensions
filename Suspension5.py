import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import os
from openpyxl import load_workbook

# Page configuration
st.set_page_config(layout="wide", page_title="Suspension Formulation Analysis System")

# File paths - 修改为GitHub仓库中的相对路径
DATA_FILE = "data.xlsx"

# 初始化数据文件（仅在本地开发时使用，云部署时注释掉）
# def init_data_file():
#     if not os.path.exists(DATA_FILE):
#         columns = [
#             'Student ID', 'Student name',
#             'Sulfadiazine (g)', 'Glycerol (ml)', 'Tragacanth gum (g)',
#             'CMC-Na (g)', 'sodium citrate (g)', 'Purified water (ml)', 'F40'
#         ]
#         pd.DataFrame(columns=columns).to_excel(DATA_FILE, index=False)
# init_data_file()

# Load data - 强化错误处理
def load_data():
    try:
        df = pd.read_excel(DATA_FILE)
        if df.empty:
            st.error("数据文件存在但为空！请上传包含数据的 data.xlsx")
            return pd.DataFrame()
        return df
    except FileNotFoundError:
        st.error(f"未找到数据文件 {DATA_FILE}！请确保：")
        st.markdown("""
        1. 在GitHub仓库中已上传 `data.xlsx` 文件
        2. 文件与 `Suspension5.py` 在同一目录下
        3. 文件包含以下列：
           - Student ID, Student name
           - Sulfadiazine (g), Glycerol (ml), Tragacanth gum (g)
           - CMC-Na (g), sodium citrate (g), Purified water (ml), F40
        """)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"数据加载失败: {str(e)}")
        return pd.DataFrame()

# Save data - 仅在本地开发时使用（云部署时只读）
def save_data(df):
    try:
        df.to_excel(DATA_FILE, index=False)
        return True
    except Exception as e:
        st.error(f"云部署模式下不允许保存数据: {e}")
        return False

# 其余函数保持不变（train_model, plot_single_factor, plot_two_factors, plot_three_factors, find_similar_formulations）
# ... [保持原有代码不变] ...

# Page 1: Data Analysis - 移除数据删除功能（云部署只读）
def data_analysis():
    st.header("📊 Data Analysis")
    df = load_data()
    
    if df.empty:
        return  # 错误信息已在 load_data() 中显示
    
    # 仅显示数据（移除删除功能）
    st.subheader("Experimental Data")
    st.dataframe(df, use_container_width=True)
    
    # 保持原有的分析代码...
    # ... [保持原有分析代码不变] ...

# Page 2: New Data Submission - 改为显示数据提交说明（云部署禁用提交）
def new_data_submission():
    st.header("➕ Data Submission Instructions")
    st.warning("""
    **云部署模式下无法直接提交数据**  
    请通过以下方式更新数据：
    1. 本地修改 `data.xlsx` 文件
    2. 重新上传到GitHub仓库
    3. 重新部署Streamlit应用
    """)
    
    # 显示当前数据示例
    df = load_data()
    if not df.empty:
        st.subheader("Current Data Structure")
        st.dataframe(df.head(3))

# Page 3: F40 Prediction - 保持不变（依赖现有数据）
def f40_prediction():
    st.header("🔮 F40 Prediction")
    df = load_data()
    
    if df.empty:
        return
    
    # ... [保持原有预测代码不变] ...

# Main app
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", 
                           ["Data Analysis", "Data Submission Instructions", "F40 Prediction"])
    
    if page == "Data Analysis":
        data_analysis()
    elif page == "Data Submission Instructions":
        new_data_submission()
    elif page == "F40 Prediction":
        f40_prediction()

if __name__ == "__main__":
    main()
