#Import required libraries
import os
from dotenv import load_dotenv, find_dotenv
import streamlit as st
import pandas as pd
from langchain.llms import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Set your API key directly
apikey = st.secrets["openai"]["api_key"]

# Title
st.title('AI Assistant for Data Science 🤖')

# Welcoming message
st.write("Hello, 👋 I am your AI Assistant and I am here to help you with your data science projects.")

# Explanation sidebar
with st.sidebar:
    st.write('*Your Data Science Adventure Begins with a CSV File.*')
    st.caption('''**Let's start a Data Science journey, with Stratascratch, shall we?**
    ''')

    st.divider()

    st.caption("<p style ='text-align:center'> Made by StrataScratch</p>", unsafe_allow_html=True)

# Initialise the key in session state
if 'clicked' not in st.session_state:
    st.session_state.clicked = {1: False}
if 'cleaned' not in st.session_state:
    st.session_state.cleaned = False
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None
if 'uploaded' not in st.session_state:
    st.session_state.uploaded = False


# Function to update the value in session state
def clicked(button):
    st.session_state.clicked[button] = True

def cleaned(data):
    st.session_state.cleaned = True
    st.session_state.cleaned_data = data

def uploaded():
    st.session_state.uploaded = True

st.button("Let's get started", on_click=clicked, args=[1])
if st.session_state.clicked[1]:
    user_csv = st.file_uploader("Upload your file here", type="csv", on_change=uploaded)
    if st.session_state.uploaded and user_csv is not None:
        user_csv.seek(0)
        df = pd.read_csv(user_csv, low_memory=False)

        # llm model
        llm = OpenAI(api_key=apikey, temperature=0)

        # Updated code with additional parameters to control prompt size
        pandas_agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            allow_dangerous_code=True,
            include_df_in_prompt=False,  # Do not include the dataframe in the prompt
            number_of_head_rows=5,  # Limit the number of rows included in the prompt if needed
            max_execution_time=400,
            max_iterations=20
        )

        @st.cache_data(show_spinner=False)
        def steps_eda():
            steps_eda = llm('What are the steps of EDA')
            return steps_eda

        # Exploratory Data Analysis Section
        st.header('Exploratory Data Analysis')
        st.subheader('General information about the dataset')

        # Data Exploration
        def data_exploration(data):
            st.subheader('Data Exploration')
            st.write("**Data Overview**")
            st.write("Let's start by taking a quick look at your dataset. We'll examine the first few rows to understand the structure of the data.")
            st.write("The first rows of your dataset look like this:")
            st.write(data.head())
            st.write("Now, let's also look at the last few rows to ensure no surprises at the end.")
            st.write("Last rows of your dataset look like this:")
            st.write(data.tail())
            st.write("Next, we'll look at a statistical dataset summary. This will give us a sense of the dataset's distribution's central tendency, dispersion, and shape.")
            st.write("Statistical Summary:")
            st.write(data.describe())
            return data

        # Data Cleaning
        def data_cleaning(data):
            st.subheader('Data Cleaning')
            st.write(""" Data cleaning involves preparing the dataset for analysis by addressing issues such as missing values, duplicate entries, 
            and inconsistencies. This step ensures the data is accurate, complete, and ready for further analysis. 
            Let's start by examining the column names and cleaning the data if necessary. """)
            columns_df = pandas_agent.run("Explain the column names")
            st.write(columns_df)

            st.write("**Missing Values**")
            st.write("""
            Missing values can significantly impact the analysis and performance of machine learning models. 
            Identifying and handling missing data is a crucial part of data cleaning. 
            Let's check if there are any missing values in our dataset and decide on appropriate strategies to address them.
            """)
            missing_values = data.isnull().sum()
            total_missing = missing_values.sum()
            if total_missing == 0:
                st.write("There are no missing values in this dataset.")
                cleaned(data)
            else:
                st.write(f"There are {total_missing} missing values in this dataset.")
                st.write(missing_values)

                # Handling missing values
                st.write("**Handling Missing Values**")
                missing_value_option = st.selectbox(
                    "Choose a strategy to handle missing values:",
                    ("Remove rows with missing values", "Impute with mean", "Impute with median", "Forward fill", "Backward fill")
                )

                if st.button("Submit", key="missing_values"):
                    if missing_value_option == "Remove rows with missing values":
                        data = data.dropna()
                    elif missing_value_option == "Impute with mean":
                        data = data.fillna(data.mean())
                    elif missing_value_option == "Impute with median":
                        data = data.fillna(data.median())
                    elif missing_value_option == "Forward fill":
                        data = data.fillna(method='ffill')
                    elif missing_value_option == "Backward fill":
                        data = data.fillna(method='bfill')
                    st.write("Missing values handled successfully.")
                    cleaned(data)

            st.write("**Duplicate Values**")
            st.write("""
            Duplicate entries can distort analyses and lead to biased results. 
            It's important to identify and remove duplicate records to ensure the integrity of the dataset. 
            Let's see if there are any duplicate values in our dataset.
            """)
            duplicates = data.duplicated().sum()
            if duplicates == 0:
                st.write("There are no duplicate values in this dataset.")
            else:
                st.write(f"There are {duplicates} duplicate values in this dataset.")
                st.write(data[data.duplicated()])

            return data

        # Data Visualization
        def data_visualization(data):
            st.subheader('Data Visualization')
            st.write("""
            Data visualization helps understand the distribution, trends, and patterns within the dataset. 
            It provides a graphical representation of data, making it easier to identify relationships and outliers. 
            Let's create some visualizations for our numerical columns.
            """)

            # Identify numerical columns
            numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if len(numerical_columns) == 0:
                st.write("No numerical columns available for visualization.")
            else:
                # Visualization 1: Histogram for the first numerical column
                st.write(f"1. Histogram of {numerical_columns[0]}")
                fig, ax = plt.subplots()
                sns.histplot(data[numerical_columns[0]], kde=True, ax=ax)
                st.pyplot(fig)

                # Visualization 2: Box Plot for the first numerical column
                st.write(f"2. Box Plot of {numerical_columns[0]}")
                fig, ax = plt.subplots()
                sns.boxplot(x=data[numerical_columns[0]], ax=ax)
                st.pyplot(fig)

                # Visualization 3: Correlation Heatmap for numerical columns
                if len(numerical_columns) > 1:
                    st.write("3. Correlation Heatmap of numerical columns")
                    fig, ax = plt.subplots()
                    corr = data[numerical_columns].corr()
                    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                    st.pyplot(fig)
                else:
                    st.write("Not enough numerical columns for a correlation heatmap.")

            return data


        def model_selection(data):
            st.subheader('Machine Learning Models')
            st.write("Let's preprocess the data and build some machine learning models. Please follow the steps below:")

            # Initialize session state for selections if not already set
            if 'scaling_option' not in st.session_state:
                st.session_state.scaling_option = "None"
            if 'reduction_option' not in st.session_state:
                st.session_state.reduction_option = "None"
            if 'n_components' not in st.session_state:
                st.session_state.n_components = 2
            if 'model_choice' not in st.session_state:
                st.session_state.model_choice = "Linear Regression"
            if 'target_column' not in st.session_state:
                st.session_state.target_column = data.columns[0] if len(data.columns) > 0 else None

            # Subheader for Label Encoding
            st.subheader("Label Encoding for Categorical Variables")
            st.write("Turning categorical variables into numerical ones is crucial for many machine learning models.")

            # Identify and encode categorical columns
            categorical_columns = data.select_dtypes(include=['object']).columns.tolist()
            if categorical_columns:
                st.write(f"Categorical columns: {', '.join(categorical_columns)}")
                for col in categorical_columns:
                    data[col] = LabelEncoder().fit_transform(data[col])
                st.write("Categorical variables have been encoded into numerical values.")
            else:
                st.write("No categorical columns to encode.")

            # Subheader for Normalization
            numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if numerical_columns:
                st.subheader("Normalization")
                st.write("Scaling numerical features is essential to ensure that all features contribute equally to the model.")

                scaling_option = st.selectbox(
                    "Select scaling method:", 
                    ["None", "Standardization", "Min-Max Scaling"], 
                    index=["None", "Standardization", "Min-Max Scaling"].index(st.session_state.scaling_option)
                )
                st.session_state.scaling_option = scaling_option

                if scaling_option == "Standardization":
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
                    st.write("Numerical features have been standardized.")
                elif scaling_option == "Min-Max Scaling":
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler()
                    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
                    st.write("Numerical features have been scaled using Min-Max scaling.")
                else:
                    st.write("No scaling applied.")
            else:
                st.write("No numerical columns to normalize.")

            # Subheader for Dimensionality Reduction
            if numerical_columns:
                st.subheader("Dimensionality Reduction")
                st.write("Reducing the number of features can help improve model performance and reduce overfitting.")

                reduction_option = st.selectbox(
                    "Select dimensionality reduction method:", 
                    ["None", "PCA"], 
                    index=["None", "PCA"].index(st.session_state.reduction_option)
                )
                st.session_state.reduction_option = reduction_option

                if reduction_option == "PCA":
                    n_components = st.slider(
                        "Select number of components:", 
                        1, len(numerical_columns), 
                        st.session_state.n_components
                    )
                    st.session_state.n_components = n_components

                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=n_components)
                    data_pca = pca.fit_transform(data[numerical_columns])
                    data_pca_df = pd.DataFrame(data_pca, columns=[f'PC{i+1}' for i in range(n_components)])
                    
                    # Preserve original column names after PCA
                    for i, col in enumerate(numerical_columns):
                        if i < n_components:
                            data_pca_df[col] = data_pca_df.pop(f'PC{i+1}')
                            
                    data = pd.concat([data.drop(columns=numerical_columns), data_pca_df], axis=1)
                    st.write(f"PCA applied, reduced to {n_components} components.")
                else:
                    st.write("No dimensionality reduction applied.")

            # Ensure target_column is valid
            if st.session_state.target_column not in data.columns:
                st.session_state.target_column = data.columns[0] if len(data.columns) > 0 else None

            target_column = st.selectbox("Select the target column", data.columns, index=list(data.columns).index(st.session_state.target_column))
            st.session_state.target_column = target_column

            # Subheader for Model Selection
            st.subheader("Model Selection")
            st.write("Let's choose and train a machine learning model based on the preprocessed data.")

            model_choice = st.selectbox(
                "Choose a model",
                ("Linear Regression", "Random Forest Classifier", "Support Vector Machine (SVM)"),
                index=["Linear Regression", "Random Forest Classifier", "Support Vector Machine (SVM)"].index(st.session_state.model_choice)
            )
            st.session_state.model_choice = model_choice

            if model_choice and target_column:
                st.write(f"You selected: {model_choice}")
                st.write(f"Target column: {target_column}")

                if st.button("Apply Model"):
                    st.write("Splitting the data into training and testing sets...")
                    X = data.drop(columns=[target_column])
                    y = data[target_column]

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    # Ensure that target column is appropriate for the selected model
                    if model_choice == "Linear Regression" and not pd.api.types.is_numeric_dtype(y):
                        st.write("Linear Regression cannot be applied to categorical target variables. Please select a numerical target column.")
                        return
                    elif model_choice in ["Random Forest Classifier", "Support Vector Machine (SVM)"] and pd.api.types.is_numeric_dtype(y):
                        st.write(f"{model_choice} is typically used for classification tasks. Please select a categorical target column.")
                        return

                    # Building and evaluating the selected model
                    if model_choice == "Linear Regression":
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        mse = mean_squared_error(y_test, y_pred)
                        accuracy = model.score(X_test, y_test) * 100
                        st.write("Mean Squared Error:", mse)
                        st.write("Accuracy:", accuracy, "%")

                    elif model_choice == "Random Forest Classifier":
                        model = RandomForestClassifier()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        st.write("Classification Report:")
                        st.write(classification_report(y_test, y_pred))
                        accuracy = model.score(X_test, y_test) * 100
                        st.write("Accuracy:", accuracy, "%")

                    elif model_choice == "Support Vector Machine (SVM)":
                        model = SVC()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        st.write("Classification Report:")
                        st.write(classification_report(y_test, y_pred))
                        accuracy = model.score(X_test, y_test) * 100
                        st.write("Accuracy:", accuracy, "%")

                    st.write("Model training and evaluation complete.")






        @st.cache_data(show_spinner=False)
        # Function to analyze a specific variable
        def function_question_variable(data, variable):
            st.subheader("Summary Statistics")
            st.write("""
            Summary statistics provide a quick overview of the dataset, 
            highlighting the central tendency, dispersion, and shape of the data's distribution. 
            These metrics are essential for understanding the overall characteristics of the dataset.
            Let's take a look at the summary statistics for the numerical columns in our dataset.
            """)
            summary_statistics = data[variable].describe()
            st.write(summary_statistics)

            if pd.api.types.is_numeric_dtype(data[variable]):
                st.subheader("Normality Check")
                fig, ax = plt.subplots()
                sns.histplot(data[variable], kde=True, ax=ax)
                st.pyplot(fig)
                normality_test = stats.normaltest(data[variable].dropna())
                st.write(f"Normality test result: Statistic={normality_test.statistic:.2f}, p-value={normality_test.pvalue:.2f}")

                st.subheader("Outliers")
                fig, ax = plt.subplots()
                sns.boxplot(x=data[variable], ax=ax)
                st.pyplot(fig)
                q1 = data[variable].quantile(0.25)
                q3 = data[variable].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = data[(data[variable] < lower_bound) | (data[variable] > upper_bound)]
                st.write(f"Number of outliers: {len(outliers)}")
                st.write(outliers)
            else:
                st.write("Normality Check, Outliers analysis skipped for non-numeric data.")

            st.subheader("Trends, Seasonality, and Cyclic Patterns")
            st.write("""
            Analyzing trends, seasonality, and cyclic patterns in the data can reveal important insights about the underlying processes. 
            Trends show long-term movements in the data, seasonality captures regular fluctuations, 
            and cyclic patterns represent non-seasonal cycles. 
            Let's visualize these aspects for the selected variable """ )
            fig, ax = plt.subplots()
            if pd.api.types.is_numeric_dtype(data[variable]):
                data[variable].plot(ax=ax)
            else:
                sns.countplot(x=data[variable], ax=ax)
            st.pyplot(fig)

            st.subheader("Missing Values")
            missing_values = data[variable].isnull().sum()
            st.write(f"Number of missing values: {missing_values}")

            return
        
        @st.cache_data(show_spinner=False)
        def function_question_dataframe(question):
            dataframe_info = pandas_agent.run(question)
            st.write(dataframe_info)
            return

        # Main
        with st.sidebar:
            with st.expander('What are the steps of EDA'):
                st.write(steps_eda())

        df = data_exploration(df)

        df_cleaned = data_cleaning(df)
        if st.session_state.cleaned:
            df_cleaned = st.session_state.cleaned_data
            df_cleaned = data_visualization(df_cleaned)
            model_selection(df_cleaned)

        st.subheader('Variable of study')
        user_question_variable = st.text_input('What variable are you interested in')
        if user_question_variable is not None and user_question_variable != "":
            function_question_variable(df_cleaned, user_question_variable)

            st.subheader('Further study')

        if user_question_variable:
            user_question_dataframe = st.text_input("Is there anything else you would like to know about your dataframe?")
            if user_question_dataframe is not None and user_question_dataframe not in ("", "no", "No"):
                function_question_dataframe(user_question_dataframe)
            if user_question_dataframe in ("no", "No"):
                st.write("")

