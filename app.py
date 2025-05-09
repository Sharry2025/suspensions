import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
import openpyxl
import os
import base64
from github import Github
import io
import time

# 配置页面
st.set_page_config(layout="wide", page_title="Pharmaceutical Formulation Analysis")

# 全局变量
GITHUB_REPO = "Sharry2025/suspensions"  # 你的GitHub用户名/仓库名
DATA_FILE = "data.xlsx"   # 仓库中的文件名
TARGET_COL = "F40"
FEATURES = [
    "Sulfadiazine (g)", "Glycerol (ml)", "Tragacanth gum (g)", 
    "CMC-Na (g)", "sodium citrate (g)", "Purified water (ml)"
]

# 初始化GitHub API
@st.cache_resource
def get_github_client():
    try:
        if "GITHUB_TOKEN" not in st.secrets:
            st.error("""
            ❌ 未配置GitHub Token！请按以下步骤操作：
            1. 确保已在Streamlit Cloud的Settings → Secrets中添加：
               [secrets]
               GITHUB_TOKEN = "你的Token"
            2. 确保Token有repo权限
            """)
            return None
        return Github(st.secrets["GITHUB_TOKEN"])
    except Exception as e:
        st.error(f"❌ GitHub连接失败: {str(e)}")
        return None

# 数据加载（从GitHub）
@st.cache_data(ttl=300)  # 5分钟缓存
def load_data():
    try:
        # 尝试从GitHub加载
        g = get_github_client()
        if g:
            repo = g.get_repo(GITHUB_REPO)
            contents = repo.get_contents(DATA_FILE)
            file_data = base64.b64decode(contents.content)
            df = pd.read_excel(io.BytesIO(file_data), engine='openpyxl')
            return df.dropna()
        
        # 如果GitHub不可用，尝试从原始URL加载
        raw_url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/{DATA_FILE}"
        df = pd.read_excel(raw_url, engine='openpyxl')
        return df.dropna()
    except Exception as e:
        st.warning(f"⚠️ 数据加载失败: {str(e)}，使用空数据框架")
        return pd.DataFrame(columns=FEATURES + [TARGET_COL])

# 保存数据到GitHub
def save_to_github(df):
    try:
        g = get_github_client()
        if not g:
            return False
            
        repo = g.get_repo(GITHUB_REPO)
        
        # 检查文件是否存在
        try:
            contents = repo.get_contents(DATA_FILE)
            sha = contents.sha
        except:
            sha = None  # 文件不存在时创建新文件
            
        # 生成Excel文件
        output = io.BytesIO()
        df.to_excel(output, index=False, engine='openpyxl')
        excel_data = output.getvalue()
        
        # 提交更改
        repo.update_file(
            path=DATA_FILE,
            message=f"Update {DATA_FILE} via Streamlit App",
            content=excel_data,
            sha=sha
        )
        return True
    except Exception as e:
        st.error(f"❌ 保存失败: {str(e)}")
        return False

# 模型训练
@st.cache_resource
def train_model():
    df = load_data()
    if len(df) < 10:
        st.warning("⚠️ 至少需要10条数据训练模型")
        return None
    
    X = df[FEATURES]
    y = df[TARGET_COL]
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 7]
    }
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    return grid_search.best_estimator_

# 界面1: 数据分析
def show_data_analysis():
    st.header("📊 Data Analysis")
    df = load_data()
    
    if len(df) < 3:
        st.warning("⚠️ 至少需要3条数据进行分析")
        return
    
    with st.expander("View Raw Data"):
        st.dataframe(df)
    
    st.subheader("Correlation Analysis")
    corr_matrix = df.corr().round(2)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    st.pyplot(fig)
    
    model = train_model()
    if model:
        st.subheader("Model Information")
        
        # 特征重要性
        importance_df = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Feature Importance:**")
            st.dataframe(importance_df)
        
        with col2:
            # 计算R²分数
            predictions = model.predict(df[FEATURES])
            r2 = r2_score(df[TARGET_COL], predictions)
            st.markdown("**Model Performance:**")
            st.metric(label="R² Score", value=f"{r2:.4f}")
            
            # 绘制实际vs预测值
            fig, ax = plt.subplots()
            ax.scatter(df[TARGET_COL], predictions)
            ax.plot([0, 1], [0, 1], 'r--')
            ax.set_xlabel("Actual F40")
            ax.set_ylabel("Predicted F40")
            st.pyplot(fig)

# 界面2: 数据管理
def show_data_management():
    st.header("📝 Data Management")
    df = load_data()

    # 添加新数据
    st.subheader("Add New Data")
    new_data = {}
    cols = st.columns(3)
    
    for i, col in enumerate(FEATURES):
        new_data[col] = cols[i%3].number_input(col, value=0.0, key=f"input_{col}")
    
    new_data[TARGET_COL] = st.number_input(TARGET_COL, value=0.0, key="input_F40")
    
    if st.button("Add Data"):
        new_df = pd.DataFrame([new_data])
        df = pd.concat([df, new_df], ignore_index=True)
        st.success("Data added to local cache! Click 'Save to GitHub' to persist.")
    
    # 数据删除功能
    st.subheader("Manage Existing Data")
    st.write("Select rows to delete:")
    
    df_with_checkbox = df.copy()
    df_with_checkbox.insert(0, "Select", False)
    
    edited_df = st.data_editor(
        df_with_checkbox,
        column_config={
            "Select": st.column_config.CheckboxColumn(
                "Select",
                help="Select rows to delete",
                default=False,
            )
        },
        hide_index=True,
        use_container_width=True
    )
    
    selected_rows = edited_df[edited_df["Select"]]
    
    if not selected_rows.empty and st.button("Delete Selected Rows"):
        df = df.drop(selected_rows.index)
        st.success(f"Marked {len(selected_rows)} row(s) for deletion. Click 'Save to GitHub' to confirm.")
    
    # 保存到GitHub
    if st.button("💾 Save to GitHub", type="primary"):
        if save_to_github(df):
            st.success("✅ Data saved to GitHub successfully!")
            st.cache_data.clear()
            st.cache_resource.clear()
            time.sleep(1)
            st.rerun()
        else:
            st.error("❌ Failed to save to GitHub")

# 界面3: F40预测
def show_prediction():
    st.header("🔮 F40 Prediction")
    model = train_model()
    
    if not model:
        st.warning("Model not trained yet (need at least 10 records)")
        return
    
    # 输入参数
    st.subheader("Input Parameters")
    input_data = {}
    cols = st.columns(3)
    
    for i, col in enumerate(FEATURES):
        input_data[col] = cols[i%3].number_input(col, value=0.0, key=f"pred_{col}")
    
    # 预测
    if st.button("Predict F40"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        prediction = np.clip(prediction, 0, 1)
        
        st.subheader("Prediction Result")
        st.metric(label="Predicted F40", value=f"{prediction:.4f}")
        
        # 显示特征贡献
        st.subheader("Feature Contributions")
        contributions = model.feature_importances_ * input_df.values[0]
        contrib_df = pd.DataFrame({
            "Feature": FEATURES,
            "Contribution": contributions,
            "Percentage": 100 * np.abs(contributions) / np.sum(np.abs(contributions))
        }).sort_values("Percentage", ascending=False)
        
        st.dataframe(contrib_df)

# 主界面
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Data Analysis", "Data Management", "F40 Prediction"])
    
    if page == "Data Analysis":
        show_data_analysis()
    elif page == "Data Management":
        show_data_management()
    elif page == "F40 Prediction":
        show_prediction()

if __name__ == "__main__":
    main()
