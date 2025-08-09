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
import base64

# Page configuration
st.set_page_config(layout="wide", page_title="Suspension Formulation Analysis System")

# File paths
DATA_FILE = "data.xlsx"

# Initialize data file with sample data if it doesn't exist
def init_data_file():
    if not os.path.exists(DATA_FILE):
        sample_data = {
            'Student ID': ['S001', 'S002'],
            'Student name': ['Alice', 'Bob'],
            'Sulfadiazine (g)': [0.5, 0.6],
            'Glycerol (ml)': [10, 12],
            'Tragacanth gum (g)': [0.2, 0.3],
            'CMC-Na (g)': [0.1, 0.15],
            'sodium citrate (g)': [0.05, 0.06],
            'Purified water (ml)': [100, 120],
            'F40': [0.8, 0.9]
        }
        pd.DataFrame(sample_data).to_excel(DATA_FILE, index=False)

init_data_file()

# Load data with enhanced error handling
def load_data():
    try:
        df = pd.read_excel(DATA_FILE)
        if df.empty:
            st.warning("Data file is empty. Using sample data.")
            init_data_file()
            return pd.read_excel(DATA_FILE)
        return df
    except Exception as e:
        st.error(f"Failed to load data: {str(e)}")
        init_data_file()
        return pd.read_excel(DATA_FILE)

# Save data with error handling
def save_data(df):
    try:
        df.to_excel(DATA_FILE, index=False)
        return True
    except Exception as e:
        st.error(f"Failed to save data: {e}")
        return False

# Train model function (unchanged)
def train_model(df):
    if df.empty or 'F40' not in df.columns:
        return None, None
    
    X = df[['Sulfadiazine (g)', 'Glycerol (ml)', 'Tragacanth gum (g)',
           'CMC-Na (g)', 'sodium citrate (g)', 'Purified water (ml)']]
    y = df['F40']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    return model, r2

# Visualization functions (unchanged)
def plot_single_factor(df, factor):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        x=df[factor], 
        y=df['F40'],
        marker='o',
        ci=95,
        ax=ax
    )
    ax.set_title(f'F40 vs {factor}', fontsize=14)
    ax.set_xlabel(factor, fontsize=12)
    ax.set_ylabel('F40', fontsize=12)
    st.pyplot(fig)

def plot_two_factors(df, factor1, factor2):
    df_heat = df.copy()
    df_heat[f'{factor1}_bin'] = pd.cut(df_heat[factor1], bins=5)
    df_heat[f'{factor2}_bin'] = pd.cut(df_heat[factor2], bins=5)
    
    heat_data = df_heat.groupby([f'{factor1}_bin', f'{factor2}_bin'])['F40'].mean().unstack()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        heat_data, 
        annot=True, 
        fmt=".2f",
        cmap='viridis',
        cbar_kws={'label': 'Mean F40'},
        ax=ax
    )
    ax.set_title(f'F40 by {factor1} and {factor2}', fontsize=14)
    ax.set_xlabel(factor2, fontsize=12)
    ax.set_ylabel(factor1, fontsize=12)
    st.pyplot(fig)

def plot_three_factors(df, factors):
    df_facet = df.copy()
    for factor in factors:
        df_facet[f'{factor}_tertile'] = pd.qcut(df_facet[factor], q=3, labels=['Low', 'Medium', 'High'])
    
    g = sns.FacetGrid(
        df_facet,
        col=f'{factors[2]}_tertile',
        row=f'{factors[1]}_tertile',
        height=4,
        aspect=1.2,
        margin_titles=True
    )
    g.map_dataframe(
        sns.lineplot,
        x=factors[0],
        y='F40',
        ci=95,
        marker='o'
    )
    g.set_axis_labels(factors[0], 'F40')
    g.set_titles(col_template='{col_name}', row_template='{row_name}')
    g.fig.suptitle(f'F40 by {factors[0]}, {factors[1]} and {factors[2]}', y=1.05)
    st.pyplot(g.fig)

# Find similar formulations (unchanged)
def find_similar_formulations(new_data, existing_df, tolerance=0.1):
    if existing_df.empty:
        return pd.DataFrame()
    
    similar = existing_df[
        (existing_df['Sulfadiazine (g)'].between(new_data['Sulfadiazine (g)']*(1-tolerance), 
                                               new_data['Sulfadiazine (g)']*(1+tolerance))) &
        (existing_df['Glycerol (ml)'].between(new_data['Glycerol (ml)']*(1-tolerance), 
                                            new_data['Glycerol (ml)']*(1+tolerance))) &
        (existing_df['Tragacanth gum (g)'].between(new_data['Tragacanth gum (g)']*(1-tolerance), 
                                                 new_data['Tragacanth gum (g)']*(1+tolerance))) &
        (existing_df['CMC-Na (g)'].between(new_data['CMC-Na (g)']*(1-tolerance), 
                                         new_data['CMC-Na (g)']*(1+tolerance))) &
        (existing_df['sodium citrate (g)'].between(new_data['sodium citrate (g)']*(1-tolerance), 
                                                 new_data['sodium citrate (g)']*(1+tolerance))) &
        (existing_df['Purified water (ml)'].between(new_data['Purified water (ml)']*(1-tolerance), 
                                                  new_data['Purified water (ml)']*(1+tolerance)))
    ]
    return similar

# Data Analysis Page with editable table
def data_analysis():
    st.header("📊 Data Analysis")
    df = load_data()
    
    if df.empty:
        return
    
    # Display editable data table
    st.subheader("Experimental Data")
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        use_container_width=True,
        key="data_editor"
    )
    
    # Save button
    if st.button("Save Changes"):
        if save_data(edited_df):
            st.success("Data saved successfully!")
            st.rerun()
        else:
            st.error("Failed to save data")
    
    # Correlation analysis
    st.subheader("Correlation Analysis")
    numeric_cols = ['Sulfadiazine (g)', 'Glycerol (ml)', 'Tragacanth gum (g)',
                   'CMC-Na (g)', 'sodium citrate (g)', 'Purified water (ml)', 'F40']
    corr = df[numeric_cols].corr().round(2)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=ax)
    st.pyplot(fig)
    
    # Visualization section
    st.subheader("Factor Analysis Visualizations")
    
    # Single factor analysis
    st.markdown("### Single Factor Analysis")
    single_factor = st.selectbox(
        "Select factor for single analysis",
        ['Sulfadiazine (g)', 'Glycerol (ml)', 'Tragacanth gum (g)',
         'CMC-Na (g)', 'sodium citrate (g)', 'Purified water (ml)']
    )
    plot_single_factor(df, single_factor)
    
    # Two factor analysis
    st.markdown("### Two Factor Analysis")
    col1, col2 = st.columns(2)
    with col1:
        factor1 = st.selectbox(
            "Select first factor",
            ['Sulfadiazine (g)', 'Glycerol (ml)', 'Tragacanth gum (g)',
             'CMC-Na (g)', 'sodium citrate (g)', 'Purified water (ml)'],
            key='factor1'
        )
    with col2:
        factor2 = st.selectbox(
            "Select second factor",
            ['Sulfadiazine (g)', 'Glycerol (ml)', 'Tragacanth gum (g)',
             'CMC-Na (g)', 'sodium citrate (g)', 'Purified water (ml)'],
            key='factor2',
            index=1
        )
    
    if factor1 != factor2:
        plot_two_factors(df, factor1, factor2)
    else:
        st.warning("Please select two different factors")
    
    # Three factor analysis
    st.markdown("### Three Factor Analysis")
    factors = st.multiselect(
        "Select three factors (order matters)",
        ['Sulfadiazine (g)', 'Glycerol (ml)', 'Tragacanth gum (g)',
         'CMC-Na (g)', 'sodium citrate (g)', 'Purified water (ml)'],
        default=['Sulfadiazine (g)', 'Glycerol (ml)', 'Tragacanth gum (g)'],
        max_selections=3
    )
    
    if len(factors) == 3:
        plot_three_factors(df, factors)
    else:
        st.warning("Please select exactly 3 factors")
    
    # Model training and evaluation
    st.subheader("Model Evaluation")
    model, r2 = train_model(df)
    
    if model:
        st.metric("Model R² Score", f"{r2:.4f}")
        
        feature_imp = pd.DataFrame({
            'Feature': ['Sulfadiazine', 'Glycerol', 'Tragacanth', 'CMC-Na', 'Sodium Citrate', 'Water'],
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=feature_imp,
            x='Importance',
            y='Feature',
            hue='Feature',
            palette='viridis',
            legend=False
        )
        st.pyplot(fig2)

# New Data Submission Page
def new_data_submission():
    st.header("➕ New Data Submission")
    df = load_data()
    
    with st.form("new_data_form"):
        st.markdown("**Please enter formulation parameters**")
        
        col1, col2 = st.columns(2)
        with col1:
            student_id = st.text_input("Student ID*")
            student_name = st.text_input("Student name* (Chinese characters allowed)")
            sulfadiazine = st.number_input("Sulfadiazine (g)*", min_value=0.0)
            glycerol = st.number_input("Glycerol (ml)*", min_value=0.0)
        with col2:
            tragacanth = st.number_input("Tragacanth gum (g)*", min_value=0.0)
            cmc_na = st.number_input("CMC-Na (g)*", min_value=0.0)
            sodium_citrate = st.number_input("sodium citrate (g)*", min_value=0.0)
            purified_water = st.number_input("Purified water (ml)*", min_value=0.0)
        
        f40 = st.number_input("F40 (0-1)*", min_value=0.0, max_value=1.0, step=0.01)
        
        submitted = st.form_submit_button("Submit Data")
        
        if submitted:
            new_row = {
                'Student ID': student_id,
                'Student name': student_name,
                'Sulfadiazine (g)': sulfadiazine,
                'Glycerol (ml)': glycerol,
                'Tragacanth gum (g)': tragacanth,
                'CMC-Na (g)': cmc_na,
                'sodium citrate (g)': sodium_citrate,
                'Purified water (ml)': purified_water,
                'F40': f40
            }
            
            updated_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            if save_data(updated_df):
                st.success("Data submitted successfully!")
                st.rerun()
            else:
                st.error("Data saving failed")

# F40 Prediction Page (unchanged)
def f40_prediction():
    st.header("🔮 F40 Prediction")
    df = load_data()
    model, _ = train_model(df)
    
    if model is None:
        st.warning("No model available. Please submit data first.")
        return
    
    with st.form("prediction_form"):
        st.markdown("""
        **Prediction Instructions**:  
        1. Enter all formulation parameters below  
        2. Click 'Predict F40' button  
        3. View predicted F40 value (0-1 scale)  
        4. System will show comparison with similar existing formulations  
        
        **Parameters**:  
        - Sulfadiazine (g): Weight in grams  
        - Glycerol (ml): Volume in milliliters  
        - Tragacanth gum (g): Weight in grams  
        - CMC-Na (g): Weight in grams  
        - sodium citrate (g): Weight in grams  
        - Purified water (ml): Volume in milliliters  
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            sulfadiazine = st.number_input("Sulfadiazine (g)", min_value=0.0)
            glycerol = st.number_input("Glycerol (ml)", min_value=0.0)
            tragacanth = st.number_input("Tragacanth gum (g)", min_value=0.0)
        with col2:
            cmc_na = st.number_input("CMC-Na (g)", min_value=0.0)
            sodium_citrate = st.number_input("sodium citrate (g)", min_value=0.0)
            purified_water = st.number_input("Purified water (ml)", min_value=0.0)
        
        if st.form_submit_button("Predict F40"):
            input_data = [[sulfadiazine, glycerol, tragacanth, cmc_na, sodium_citrate, purified_water]]
            prediction = model.predict(input_data)[0]
            
            st.success(f"Predicted F40 Value: {prediction:.3f}")
            
            new_row = {
                'Sulfadiazine (g)': sulfadiazine,
                'Glycerol (ml)': glycerol,
                'Tragacanth gum (g)': tragacanth,
                'CMC-Na (g)': cmc_na,
                'sodium citrate (g)': sodium_citrate,
                'Purified water (ml)': purified_water,
                'F40': prediction
            }
            
            similar = find_similar_formulations(new_row, df)
            if not similar.empty:
                st.subheader("Similar Existing Formulations")
                st.dataframe(similar)
                
                avg_f40 = similar['F40'].mean()
                diff = prediction - avg_f40
                st.info(f"Predicted value is {diff:+.3f} ({diff/avg_f40*100:+.1f}%) {'higher' if diff > 0 else 'lower'} than similar formulations average ({avg_f40:.3f})")
            else:
                st.info("No similar formulations found in database")

# Main app
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
