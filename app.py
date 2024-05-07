import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle
import os
import shutil
import git

# Data Collection and Preprocessing
def data_collection_and_preprocessing():
    st.subheader("Data Collection and Preprocessing")
    # Code for data collection and preprocessing
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write(df.head())
        return df
    else:
        st.warning("Please upload a CSV file.")
        return None

# Exploratory Data Analysis (EDA)
def display_data_summary(df):
    st.subheader("Data Summary")
    st.write(df.describe())

def display_univariate_visualizations(df):
    st.subheader("Univariate Visualizations")
    for column in df.columns:
        fig, ax = plt.subplots()
        ax.hist(df[column], bins=20)
        st.pyplot(fig)

def display_bivariate_visualizations(df):
    st.subheader("Bivariate Visualizations")
    for column1 in df.columns:
        for column2 in df.columns:
            if column1 != column2:
                fig, ax = plt.subplots()
                ax.scatter(df[column1], df[column2])
                ax.set_xlabel(column1)
                ax.set_ylabel(column2)
                st.pyplot(fig)

def display_correlation_plots(df):
    st.subheader("Correlation Plots")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

# Feature Engineering
def feature_selection(df):
    st.subheader("Feature Selection")
    selected_columns = st.multiselect("Select features", df.columns)
    if selected_columns:
        selected_df = df[selected_columns]
        return selected_df
    else:
        st.warning("No features selected.")
        return None

def feature_transformation(df, columns):
    st.subheader("Feature Transformation")
    numeric_columns = st.multiselect("Select numeric columns", columns, default=columns)
    categorical_columns = [col for col in columns if col not in numeric_columns]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_columns),
            ("cat", categorical_transformer, categorical_columns),
        ]
    )

    transformed_data = preprocessor.fit_transform(df)
    transformed_df = pd.DataFrame(transformed_data, columns=preprocessor.get_feature_names_out())
    return transformed_df

def feature_importance(df, target_column, problem_type):
    st.subheader("Feature Importance")
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    if problem_type == "Classification":
        # Code for classification feature importance
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier()
        rf.fit(X, y)
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        st.write("Feature Importances:")
        for f in range(X.shape[1]):
            st.write("%d. %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))
    else:
        # Code for regression feature importance
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor()
        rf.fit(X, y)
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        st.write("Feature Importances:")
        for f in range(X.shape[1]):
            st.write("%d. %s (%f)" % (f + 1, X.columns[indices[f]], importances[indices[f]]))

# Model Selection and Tuning
def model_selection_and_tuning(df, target_column, problem_type):
    st.subheader("Model Selection and Tuning")
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if problem_type == "Classification":
        # Code for classification model selection and tuning
        from sklearn.linear_model import LogisticRegression
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV

        models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Support Vector Machine": SVC()
        }

        model_selection = st.selectbox("Select a classification model", list(models.keys()))
        model = models[model_selection]

        if model_selection == "Logistic Regression":
            param_grid = {"C": [0.1, 1, 10]}
        elif model_selection == "Decision Tree":
            param_grid = {"max_depth": [3, 5, 7]}
        elif model_selection == "Random Forest":
            param_grid = {"n_estimators": [100, 200, 300]}
        else:
            param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}

        grid_search = GridSearchCV(model, param_grid, cv=5, scoring="accuracy")
        grid_search.fit(X_train, y_train)
        st.write(f"Best parameters: {grid_search.best_params_}")
        st.write(f"Best score: {grid_search.best_score_:.2f}")
        model = grid_search.best_estimator_

    else:
        # Code for regression model selection and tuning
        from sklearn.linear_model import LinearRegression
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.svm import SVR
        from sklearn.model_selection import GridSearchCV

        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "Support Vector Regression": SVR()
        }

        model_selection = st.selectbox("Select a regression model", list(models.keys()))
        model = models[model_selection]

        if model_selection == "Linear Regression":
            param_grid = {}
        elif model_selection == "Decision Tree":
            param_grid = {"max_depth": [3, 5, 7]}
        elif model_selection == "Random Forest":
            param_grid = {"n_estimators": [100, 200, 300]}
        else:
            param_grid = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}

        grid_search = GridSearchCV(model, param_grid, cv=5, scoring="r2")
        grid_search.fit(X_train, y_train)
        st.write(f"Best parameters: {grid_search.best_params_}")
        st.write(f"Best score: {grid_search.best_score_:.2f}")
        model = grid_search.best_estimator_

    return model

# Model Deployment
def model_deployment(df, target_column, problem_type, model=None):
    st.subheader("Model Deployment")
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    if model is None:
        st.info("No model loaded. Please train or load a model first.")
    else:
        st.write("Model loaded successfully!")
        
        # Save the model
        save_model = st.button("Save Model")
        if save_model:
            pickle.dump(model, open("model.pkl", "wb"))
            st.success("Model saved successfully!")
        
        # Load a saved model
        load_model = st.file_uploader("Load a Saved Model", type=["pkl"])
        if load_model is not None:
            model = pickle.load(open(load_model, "rb"))
            st.success("Model loaded successfully!")
        
        # Make predictions
        new_data = st.text_area("Enter new data (comma-separated values)")
        if new_data:
            new_data = [float(x) for x in new_data.split(",")]
            new_data = np.array(new_data).reshape(1, -1)
            prediction = model.predict(new_data)
            
            if problem_type == "Classification":
                st.write(f"Prediction: {prediction[0]}")
                
                # Confusion matrix
                y_pred = model.predict(X)
                cm = confusion_matrix(y, y_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                fig, ax = plt.subplots(figsize=(6, 6))
                disp.plot(ax=ax)
                st.pyplot(fig)
            else:
                st.write(f"Predicted value: {prediction[0]:.2f}")
                
                # Scatter plot
                plt.figure(figsize=(8, 6))
                plt.scatter(y, model.predict(X), alpha=0.5)
                plt.xlabel("Actual Values")
                plt.ylabel("Predicted Values")
                plt.title("Actual vs. Predicted Values")
                st.pyplot()

# Documentation and Collaboration
def documentation_and_collaboration():
    st.subheader("Documentation and Collaboration")
    
    # Project documentation
    st.write("### Project Documentation")
    project_doc = st.text_area("Enter your project documentation here (Markdown supported)", height=300)
    st.markdown(project_doc)
    
    # Collaboration features
    st.write("### Collaboration Features")
    
    # Git integration
    repo_url = st.text_input("Enter the Git repository URL")
    if repo_url:
        try:
            repo = git.Repo.clone_from(repo_url, "repo")
            st.success("Repository cloned successfully!")
        except Exception as e:
            st.error(f"Error cloning repository: {e}")
    
    # File uploader/downloader
    uploaded_file = st.file_uploader("Upload a file", type=["py", "ipynb", "txt", "csv", "xlsx"])
    if uploaded_file is not None:
        with open(os.path.join("repo", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")
    
    files = os.listdir("repo")
    selected_file = st.selectbox("Select a file to download", files)
    if selected_file:
        with open(os.path.join("repo", selected_file), "rb") as f:
            bytes_data = f.read()
        st.download_button(
            label=f"Download {selected_file}",
            data=bytes_data,
            file_name=selected_file,
            mime="application/octet-stream",
        )
    
    # Export project
    st.write("### Export Project")
    export_project = st.button("Export Project")
    if export_project:
        shutil.make_archive("project", "zip", "repo")
        with open("project.zip", "rb") as f:
            bytes_data = f.read()
        st.download_button(
            label="Download Project",
            data=bytes_data,
            file_name="project.zip",
            mime="application/zip",
        )
        os.remove("project.zip")

def main():
    st.title("Streamlit Automated Data Science Toolkit")
    sections = ["Data Collection and Preprocessing", "Exploratory Data Analysis (EDA)", "Feature Engineering", "Model Selection and Tuning", "Model Deployment", "Documentation and Collaboration"]
    selected_section = st.sidebar.radio("Select a section", sections)

    df = None
    transformed_df = None
    model = None

    if selected_section == "Data Collection and Preprocessing":
        df = data_collection_and_preprocessing()

    elif selected_section == "Exploratory Data Analysis (EDA)":
        if df is not None:
            display_data_summary(df)
            display_univariate_visualizations(df)
            display_bivariate_visualizations(df)
            display_correlation_plots(df)

    elif selected_section == "Feature Engineering":
        if df is not None:
            selected_df = feature_selection(df)
            if selected_df is not None:
                transformed_df = feature_transformation(selected_df, selected_df.columns)
                target_column = st.selectbox("Select the target column", transformed_df.columns)
                problem_type = st.radio("Select the problem type", ["Classification", "Regression"])
                feature_importance(transformed_df, target_column, problem_type)

    elif selected_section == "Model Selection and Tuning":
        if transformed_df is not None:
            target_column = st.selectbox("Select the target column", transformed_df.columns)
            problem_type = st.radio("Select the problem type", ["Classification", "Regression"])
            model = model_selection_and_tuning(transformed_df, target_column, problem_type)

    elif selected_section == "Model Deployment":
        if transformed_df is not None and model is not None:
            target_column = st.selectbox("Select the target column", transformed_df.columns)
            problem_type = st.radio("Select the problem type", ["Classification", "Regression"])
            model_deployment(transformed_df, target_column, problem_type, model)

    elif selected_section == "Documentation and Collaboration":
        documentation_and_collaboration()

if __name__ == "__main__":
    main()
