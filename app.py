import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
import openpyxl
import base64
import requests
import io
import os

# GitHub configuration
GITHUB_USER = "Sharry2025"
GITHUB_REPO = "suspensions"
DATA_FILE = "data.xlsx"
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")  # Get token from environment variables
HEADERS = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# Configure page
st.set_page_config(layout="wide", page_title="Pharmaceutical Formulation Analysis")

# Global variables
TARGET_COL = "F40"
FEATURES = [
    "Sulfadiazine (g)", "Glycerol (ml)", "Tragacanth gum (g)", 
    "CMC-Na (g)", "sodium citrate (g)", "Purified water (ml)"
]

# GitHub API functions
def get_file_sha():
    url = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{DATA_FILE}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code == 200:
        return response.json()["sha"]
    return None

def load_data_from_github():
    url = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{DATA_FILE}"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code == 200:
        content = response.json()["content"]
        decoded = base64.b64decode(content)
        df = pd.read_excel(io.BytesIO(decoded), engine='openpyxl')
        return df.dropna()
    else:
        # If file doesn't exist or can't be accessed, create empty DataFrame with columns
        cols = FEATURES + [TARGET_COL]
        return pd.DataFrame(columns=cols)

def save_data_to_github(df):
    # Convert DataFrame to Excel in memory
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False)
    excel_data = output.getvalue()
    
    # Encode to base64
    encoded = base64.b64encode(excel_data).decode()
    
    # Get current file SHA for update
    sha = get_file_sha()
    
    # Prepare API request
    url = f"https://api.github.com/repos/{GITHUB_USER}/{GITHUB_REPO}/contents/{DATA_FILE}"
    data = {
        "message": "Update data file from Streamlit app",
        "content": encoded,
        "sha": sha if sha else None  # If sha is None, GitHub will create a new file
    }
    
    response = requests.put(url, headers=HEADERS, json=data)
    return response.status_code == 200

# Data loading with caching
@st.cache_data
def load_data():
    return load_data_from_github()

# Model training
@st.cache_resource
def train_model():
    df = load_data()
    if len(df) < 10:  # Don't train model if not enough data
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
    best_model = grid_search.best_estimator_
    
    return best_model

# Interface 1: Data Analysis
def show_data_analysis():
    st.header("📊 Data Analysis")
    df = load_data()
    
    if len(df) < 3:
        st.warning("Not enough data for analysis (minimum 3 records required)")
        return
    
    # Show raw data
    with st.expander("View Raw Data"):
        st.dataframe(df)
    
    # Correlation analysis
    st.subheader("Correlation Analysis")
    corr_matrix = df.corr().round(2)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
    st.pyplot(fig)
    
    # Model training and evaluation
    model = train_model()
    if model:
        st.subheader("Model Information")
        
        # Feature importance
        importance_df = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Feature Importance:**")
            st.dataframe(importance_df)
        
        with col2:
            # Calculate R² score
            predictions = model.predict(df[FEATURES])
            r2 = r2_score(df[TARGET_COL], predictions)
            st.markdown("**Model Performance:**")
            st.metric(label="R² Score", value=f"{r2:.4f}")
            
            # Plot actual vs predicted
            fig, ax = plt.subplots()
            ax.scatter(df[TARGET_COL], predictions)
            ax.plot([0, 1], [0, 1], 'r--')
            ax.set_xlabel("Actual F40")
            ax.set_ylabel("Predicted F40")
            st.pyplot(fig)

# Interface 2: Data Management
def show_data_management():
    st.header("📝 Data Management")
    df = load_data()

    # Add new data
    st.subheader("Add New Data")
    new_data = {}
    cols = st.columns(3)
    
    for i, col in enumerate(FEATURES):
        new_data[col] = cols[i%3].number_input(col, value=0.0, key=f"input_{col}")
    
    new_data[TARGET_COL] = st.number_input(TARGET_COL, value=0.0, key="input_F40")
    
    if st.button("Add Data"):
        new_df = pd.DataFrame([new_data])
        df = pd.concat([df, new_df], ignore_index=True)
        if save_data_to_github(df):
            st.success("Data added successfully!")
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        else:
            st.error("Failed to save data to GitHub. Please check your GitHub token and permissions.")
    
    # Data deletion functionality
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
        if save_data_to_github(df):
            st.success(f"Deleted {len(selected_rows)} row(s)")
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        else:
            st.error("Failed to save data to GitHub. Please check your GitHub token and permissions.")

# Interface 3: F40 Prediction
def show_prediction():
    st.header("🔮 F40 Prediction")
    model = train_model()
    
    if not model:
        st.warning("Model not trained yet (need at least 10 records)")
        return
    
    # Input parameters
    st.subheader("Input Parameters")
    input_data = {}
    cols = st.columns(3)
    
    for i, col in enumerate(FEATURES):
        input_data[col] = cols[i%3].number_input(col, value=0.0, key=f"pred_{col}")
    
    # Prediction
    if st.button("Predict F40"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        prediction = np.clip(prediction, 0, 1)  # Ensure between 0-1
        
        st.subheader("Prediction Result")
        st.metric(label="Predicted F40", value=f"{prediction:.4f}")
        
        # Show feature contributions
        st.subheader("Feature Contributions")
        contributions = model.feature_importances_ * input_df.values[0]
        contrib_df = pd.DataFrame({
            "Feature": FEATURES,
            "Contribution": contributions,
            "Percentage": 100 * np.abs(contributions) / np.sum(np.abs(contributions))
        }).sort_values("Percentage", ascending=False)
        
        st.dataframe(contrib_df)

# Main interface
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
