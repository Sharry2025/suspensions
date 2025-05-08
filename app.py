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
from github import Github, InputGitTreeElement
import io
import time

# 配置页面
st.set_page_config(
    layout="wide",
    page_title="Pharmaceutical Formulation Analysis",
    page_icon="🧪"
)

# 全局变量
GITHUB_REPO = "Sharry2025/suspensions"  # 格式：用户名/仓库名
DATA_FILE = "data.xlsx"   # 仓库中的文件名
TARGET_COL = "F40"
FEATURES = [
    "Sulfadiazine (g)", "Glycerol (ml)", "Tragacanth gum (g)", 
    "CMC-Na (g)", "sodium citrate (g)", "Purified water (ml)"
]

# 初始化GitHub API（带完整错误处理）
@st.cache_resource
def get_github_client():
    """初始化GitHub客户端，包含详细的错误处理"""
    # 检查是否存在Token
    if "GITHUB_TOKEN" not in st.secrets:
        st.error("""
        ❌ GitHub Token未配置！请按以下步骤操作：
        1. 访问 [GitHub Token设置](https://github.com/settings/tokens) 生成新Token（需repo权限）
        2. 在Streamlit Cloud的Settings → Secrets中添加：
           ```toml
           [secrets]
           GITHUB_TOKEN = "你的Token"
           ```
        """, icon="⚠️")
        return None
    
    try:
        # 测试Token有效性
        g = Github(st.secrets["GITHUB_TOKEN"])
        test_repo = g.get_repo(GITHUB_REPO)  # 测试仓库访问权限
        return g
    except ImportError:
        st.error("""
        ❌ 缺少PyGithub库！请在requirements.txt中添加：
        ```text
        PyGithub>=1.55
        ```
        """, icon="🛠️")
    except Exception as e:
        error_msg = str(e)
        if "Bad credentials" in error_msg:
            st.error("❌ Token无效或已过期！请重新生成Token", icon="🔑")
        elif "Not Found" in error_msg:
            st.error(f"❌ 仓库不存在或无权限访问：{GITHUB_REPO}", icon="📂")
        else:
            st.error(f"❌ GitHub连接失败：{error_msg}", icon="🚨")
    return None

# 数据加载（带缓存和重试机制）
@st.cache_data(ttl=300)  # 5分钟缓存
def load_data(max_retries=3):
    """从GitHub加载数据，支持重试机制"""
    for attempt in range(max_retries):
        try:
            raw_url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/main/{DATA_FILE}"
            df = pd.read_excel(raw_url, engine='openpyxl')
            if df.empty:
                st.warning("⚠️ 数据文件为空，正在创建新数据框架")
                return pd.DataFrame(columns=FEATURES + [TARGET_COL])
            return df.dropna()
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"❌ 数据加载失败（最终尝试）：{str(e)}", icon="🚨")
                return pd.DataFrame(columns=FEATURES + [TARGET_COL])
            time.sleep(2)  # 延迟重试

# 保存数据到GitHub（增强版）
def save_to_github(df):
    """保存数据到GitHub仓库，包含完整事务处理"""
    if df.empty:
        st.error("❌ 不能保存空数据", icon="📭")
        return False
    
    try:
        g = get_github_client()
        if not g:
            return False
        
        with st.spinner("🚀 正在保存到GitHub..."):
            repo = g.get_repo(GITHUB_REPO)
            
            # 检查文件是否存在
            try:
                contents = repo.get_contents(DATA_FILE)
                sha = contents.sha
                action = "更新"
            except:
                sha = None
                action = "创建"
            
            # 生成Excel文件
            output = io.BytesIO()
            df.to_excel(output, index=False, engine='openpyxl')
            excel_data = output.getvalue()
            
            # 提交更改
            repo.update_file(
                path=DATA_FILE,
                message=f"{action} {DATA_FILE} (来自Streamlit应用)",
                content=excel_data,
                sha=sha
            )
            
            st.toast(f"✅ 数据已成功{action}到GitHub！", icon="✅")
            return True
            
    except Exception as e:
        error_msg = str(e)
        if "This repository is empty" in error_msg:
            st.error("❌ 目标仓库为空，请先提交一个初始文件", icon="📦")
        elif "large files" in error_msg:
            st.error("❌ 文件过大，请减小数据规模", icon="📏")
        else:
            st.error(f"❌ 保存失败：{error_msg}", icon="🚨")
        return False

# 模型训练（优化版）
@st.cache_resource(ttl=3600)  # 1小时缓存
def train_model():
    """训练模型，带数据量检查"""
    df = load_data()
    if len(df) < 10:
        st.warning("⚠️ 至少需要10条数据才能训练模型", icon="📉")
        return None
    
    try:
        X = df[FEATURES]
        y = df[TARGET_COL]
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7]
        }
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring='r2'
        )
        
        with st.spinner("🤖 正在训练模型..."):
            grid_search.fit(X, y)
        
        return grid_search.best_estimator_
        
    except Exception as e:
        st.error(f"❌ 模型训练失败：{str(e)}", icon="⚙️")
        return None

# 界面1: 数据分析（优化版）
def show_data_analysis():
    st.header("📊 数据分析", divider="rainbow")
    df = load_data()
    
    if len(df) < 3:
        st.warning("⚠️ 至少需要3条数据进行分析", icon="📊")
        return
    
    with st.expander("📂 查看原始数据", expanded=True):
        st.dataframe(df, use_container_width=True)
    
    st.subheader("🔗 相关性分析")
    try:
        corr_matrix = df.corr().round(2)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ 相关性分析失败：{str(e)}", icon="📈")
    
    model = train_model()
    if model:
        st.subheader("🤖 模型信息")
        
        tab1, tab2 = st.tabs(["特征重要性", "模型性能"])
        
        with tab1:
            importance_df = pd.DataFrame({
                "特征": FEATURES,
                "重要性": model.feature_importances_
            }).sort_values("重要性", ascending=False)
            
            st.dataframe(
                importance_df,
                column_config={
                    "特征": st.column_config.TextColumn(width="medium"),
                    "重要性": st.column_config.ProgressColumn(
                        min_value=0,
                        max_value=float(importance_df["重要性"].max()),
                },
                hide_index=True,
                use_container_width=True
            )
        
        with tab2:
            predictions = model.predict(df[FEATURES])
            r2 = r2_score(df[TARGET_COL], predictions)
            
            col1, col2 = st.columns(2)
            col1.metric("R² 分数", f"{r2:.4f}")
            
            fig, ax = plt.subplots()
            ax.scatter(df[TARGET_COL], predictions, alpha=0.6)
            ax.plot([0, 1], [0, 1], 'r--')
            ax.set_xlabel("实际值")
            ax.set_ylabel("预测值")
            col2.pyplot(fig)

# 界面2: 数据管理（完整版）
def show_data_management():
    st.header("📝 数据管理", divider="rainbow")
    df = load_data()
    
    # 添加新数据
    with st.expander("➕ 添加新数据", expanded=True):
        new_data = {}
        cols = st.columns(3)
        for i, col in enumerate(FEATURES):
            new_data[col] = cols[i%3].number_input(
                col,
                value=0.0,
                key=f"input_{col}",
                help=f"输入 {col} 的值"
            )
        new_data[TARGET_COL] = st.number_input(
            TARGET_COL,
            value=0.0,
            key="input_F40",
            help="输入目标值"
        )
        
        if st.button("添加数据", type="primary"):
            new_df = pd.DataFrame([new_data])
            df = pd.concat([df, new_df], ignore_index=True)
            st.success("✅ 数据已暂存到本地，请点击下方保存到GitHub")
    
    # 数据编辑与删除
    with st.expander("✏️ 编辑现有数据", expanded=True):
        st.write("勾选要删除的行：")
        df_with_checkbox = df.copy()
        df_with_checkbox.insert(0, "选择", False)
        
        edited_df = st.data_editor(
            df_with_checkbox,
            column_config={
                "选择": st.column_config.CheckboxColumn(
                    "选择",
                    help="选择要删除的行",
                    default=False
                )
            },
            hide_index=True,
            use_container_width=True,
            key="data_editor"
        )
        
        if st.button("删除选定行", type="secondary"):
            selected_rows = edited_df[edited_df["选择"]]
            if not selected_rows.empty:
                df = df.drop(selected_rows.index)
                st.success(f"✅ 已标记删除 {len(selected_rows)} 行，请点击下方保存到GitHub")
            else:
                st.warning("⚠️ 未选择任何行")
    
    # 保存操作
    st.divider()
    if st.button("💾 保存到GitHub", type="primary", use_container_width=True):
        if save_to_github(df):
            st.cache_data.clear()  # 清除数据缓存
            st.cache_resource.clear()  # 清除模型缓存
            time.sleep(1)
            st.rerun()

# 界面3: F40预测（优化版）
def show_prediction():
    st.header("🔮 F40预测", divider="rainbow")
    model = train_model()
    
    if not model:
        st.warning("⚠️ 模型未训练（需要至少10条数据）", icon="🤖")
        return
    
    st.subheader("📥 输入参数")
    input_data = {}
    cols = st.columns(3)
    for i, col in enumerate(FEATURES):
        input_data[col] = cols[i%3].number_input(
            col,
            value=0.0,
            min_value=0.0,
            step=0.1,
            key=f"pred_{col}"
        )
    
    if st.button("预测F40", type="primary"):
        with st.spinner("🔮 正在预测..."):
            input_df = pd.DataFrame([input_data])
            try:
                prediction = np.clip(model.predict(input_df)[0], 0, 1)
                
                st.subheader("📤 预测结果")
                st.metric(
                    label="预测值",
                    value=f"{prediction:.4f}",
                    help="F40预测值（0-1范围）"
                )
                
                st.subheader("📊 特征贡献度")
                contributions = model.feature_importances_ * input_df.values[0]
                contrib_df = pd.DataFrame({
                    "特征": FEATURES,
                    "贡献值": contributions,
                    "贡献百分比": 100 * np.abs(contributions) / np.sum(np.abs(contributions))
                }).sort_values("贡献百分比", ascending=False)
                
                st.dataframe(
                    contrib_df,
                    column_config={
                        "贡献百分比": st.column_config.ProgressColumn(
                            format="%.1f%%",
                            min_value=0,
                            max_value=100
                        )
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"❌ 预测失败：{str(e)}", icon="🚨")

# 主界面
def main():
    st.sidebar.title("导航")
    with st.sidebar:
        st.image("https://streamlit.io/images/brand/streamlit-mark-color.png", width=50)
        page = st.radio(
            "选择功能",
            ["数据分析", "数据管理", "F40预测"],
            index=0,
            label_visibility="collapsed"
        )
    
    if page == "数据分析":
        show_data_analysis()
    elif page == "数据管理":
        show_data_management()
    elif page == "F40预测":
        show_prediction()

    # 页脚
    st.sidebar.divider()
    st.sidebar.caption(f"最后更新：{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")

if __name__ == "__main__":
    main()
