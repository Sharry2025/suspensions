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



# 配置页面

st.set_page_config(layout="wide", page_title="Pharmaceutical Formulation Analysis")



# 全局变量

DATA_PATH = "E:/suspensions/data.xlsx"

TARGET_COL = "F40"

FEATURES = [

    "Sulfadiazine (g)", "Glycerol (ml)", "Tragacanth gum (g)", 

    "CMC-Na (g)", "sodium citrate (g)", "Purified water (ml)"

]



# 数据加载与保存

@st.cache_data

def load_data():

    if os.path.exists(DATA_PATH):

        df = pd.read_excel(DATA_PATH, engine='openpyxl')

        return df.dropna()

    else:

        # 如果文件不存在，创建带列名的空DataFrame

        cols = FEATURES + [TARGET_COL]

        return pd.DataFrame(columns=cols)



def save_data(df):

    df.to_excel(DATA_PATH, index=False, engine='openpyxl')



# 模型训练

@st.cache_resource

def train_model():

    df = load_data()

    if len(df) < 10:  # 数据量太少时不训练模型

        return None

    

    X = df[FEATURES]

    y = df[TARGET_COL]

    

    # 使用随机森林回归

    model = RandomForestRegressor(

        n_estimators=100,

        max_depth=5,

        random_state=42

    )

    

    # 添加简单的参数搜索

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

    best_model = grid_search.best_estimator_

    

    return best_model



# 界面1: 数据分析

def show_data_analysis():

    st.header("📊 Data Analysis")

    df = load_data()

    

    if len(df) < 3:

        st.warning("Not enough data for analysis (minimum 3 records required)")

        return

    

    # 显示原始数据

    with st.expander("View Raw Data"):

        st.dataframe(df)

    

    # 相关性分析

    st.subheader("Correlation Analysis")

    corr_matrix = df.corr().round(2)

    

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)

    st.pyplot(fig)

    

    # 模型训练和评估

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

        save_data(df)

        st.success("Data added successfully!")

        st.cache_data.clear()

        st.cache_resource.clear()

        st.rerun()

    

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

        save_data(df)

        st.success(f"Deleted {len(selected_rows)} row(s)")

        st.cache_data.clear()

        st.cache_resource.clear()

        st.rerun()



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

        prediction = np.clip(prediction, 0, 1)  # 确保在0-1范围内

        

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
