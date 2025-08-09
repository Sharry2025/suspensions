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

# File paths - 修改为GitHub仓库中的文件名
DATA_FILE = "data5.xlsx"

# 移除本地初始化函数，完全依赖上传的data5.xlsx文件

# Load data - 仅做最小修改
def load_data():
    try:
        df = pd.read_excel(DATA_FILE)
        if df.empty:
            st.error("Error: Data file is empty. Please upload a valid data5.xlsx file.")
            return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}\n\n"
                 "Please ensure:\n"
                 "1. data5.xlsx exists in the GitHub repository\n"
                 "2. File is in the same directory as Suspension5.py\n"
                 "3. File contains all required columns")
        return pd.DataFrame()

# Save data - 修改为返回数据供下载
def save_data(df):
    try:
        # 在Streamlit Cloud中不能直接写入文件，改为提供下载链接
        towrite = BytesIO()
        df.to_excel(towrite, index=False, engine='openpyxl')
        towrite.seek(0)
        b64 = base64.b64encode(towrite.read()).decode()
        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="updated_data.xlsx">Download Updated Data</a>'
        st.markdown(href, unsafe_allow_html=True)
        return True
    except Exception as e:
        st.error(f"Data saving failed: {e}\n\n"
                "In Streamlit Cloud, you need to download the updated file "
                "and upload it back to GitHub manually.")
        return False

# 其余所有函数保持完全不变
# [保持原样的train_model, plot_single_factor, plot_two_factors, 
#  plot_three_factors, find_similar_formulations, 
#  data_analysis, new_data_submission, f40_prediction函数]

# Main app - 保持不变
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", 
                           ["Data Analysis", "New Data Submission", "F40 Prediction"])
    
    if page == "Data Analysis":
        data_analysis()
    elif page == "New Data Submission":
        new_data_submission()
    elif page == "F40 Prediction":
        f40_prediction()

if __name__ == "__main__":
    main()
