#Import required libraries
from openai import OpenAI
import streamlit as st
import pandas as pd
from langchain_community.llms import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import matplotlib.pyplot as plt
import seaborn as sns

# Set your API key directly

apikey = st.secrets["openai"]["api_key"]

#Title
st.title('AI Assistant for Data Science 🤖')

#Welcoming message
st.write("Hello, 👋 I am your AI Assistant and I am here to help you with your data science projects.")

#Explanation sidebar
with st.sidebar:
    st.write('*Your Data Science Adventure Begins with an CSV File.*')
    st.caption('''**You may already know that every exciting data science journey starts with a dataset.
    That's why I'd love for you to upload a CSV file.
    Once we have your data in hand, we'll dive into understanding it and have some fun exploring it.
    Then, we'll work together to shape your business challenge into a data science framework.
    I'll introduce you to the coolest machine learning models, and we'll use them to tackle your problem. Sounds fun right?**
    ''')

    st.divider()

    st.caption("<p style ='text-align:center'> Made by StrataScratch</p>",unsafe_allow_html=True )

#Initialise the key in session state
if 'clicked' not in st.session_state:
    st.session_state.clicked ={1:False}

#Function to udpate the value in session state
def clicked(button):
    st.session_state.clicked[button]= True
st.button("Let's get started", on_click = clicked, args=[1])
if st.session_state.clicked[1]:
    user_csv = st.file_uploader("Upload your file here", type="csv")
    if user_csv is not None:
        user_csv.seek(0)
        df = pd.read_csv(user_csv, low_memory=False)

        #llm model
        llm = OpenAI(api_key=apikey, temperature=0)


        #Function sidebar
        @st.cache_data
        def steps_eda():
            steps_eda = llm('What are the steps of EDA')
            return steps_eda

        # Updated code with additional parameters to control prompt size
        pandas_agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            allow_dangerous_code=True,
            include_df_in_prompt=False,  # Do not include the dataframe in the prompt
            number_of_head_rows=5  # Limit the number of rows included in the prompt if needed
        )

        #Functions main
        @st.cache_data
        def function_agent():
            st.write("**Data Overview**")
            st.write("The first rows of your dataset look like this:")
            st.write(df.head())
            st.write("Last rows of your dataset look like this:")
            st.write(df.tail())
            st.write("Statistical Summary:")
            st.write(df.describe())
            st.write("**Data Cleaning**")
            columns_df = pandas_agent.run("Explain the column names")
            st.write(columns_df)
            st.write("**Visualization**")
            # Identify numerical columns
            numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if len(numerical_columns) == 0:
                st.write("No numerical columns available for visualization.")
            else:
                # Visualization 1: Histogram for the first numerical column
                st.write(f"1. Histogram of {numerical_columns[0]}")
                fig, ax = plt.subplots()
                sns.histplot(df[numerical_columns[0]], kde=True, ax=ax)
                st.pyplot(fig)

                # Visualization 2: Box Plot for the first numerical column
                st.write(f"2. Box Plot of {numerical_columns[0]}")
                fig, ax = plt.subplots()
                sns.boxplot(x=df[numerical_columns[0]], ax=ax)
                st.pyplot(fig)

                # Visualization 3: Correlation Heatmap for numerical columns
                if len(numerical_columns) > 1:
                    st.write("3. Correlation Heatmap of numerical columns")
                    fig, ax = plt.subplots()
                    corr = df[numerical_columns].corr()
                    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                    st.pyplot(fig)
                else:
                    st.write("Not enough numerical columns for a correlation heatmap.")
            st.write("**Missing Values**")
            missing_values = pandas_agent.run("Are there any missing values in this dataset?Start with There are'")
            st.write(missing_values)
            duplicates = pandas_agent.run("Are there any duplicates?")
            st.write(duplicates)
            correlation_analysis = pandas_agent.run("Calculate correlations between numerical variables to identify potential relationships.")
            st.write(correlation_analysis)
            outliers = pandas_agent.run("Identify outliers in the data. Start with there are:")
            st.write(outliers)
            new_features = pandas_agent.run("What new features would be interesting to create?.")
            st.write(new_features)
            return

        @st.cache_data
        def function_question_variable():
            st.line_chart(df, y =[user_question_variable])
            summary_statistics = pandas_agent.run(f"Give me a summary of the statistics of {user_question_variable}")
            st.write(summary_statistics)
            normality = pandas_agent.run(f"Check for normality or specific distribution shapes of {user_question_variable}")
            st.write(normality)
            outliers = pandas_agent.run(f"Assess the presence of outliers of {user_question_variable}")
            st.write(outliers)
            trends = pandas_agent.run(f"Analyse trends, seasonality, and cyclic patterns of {user_question_variable}")
            st.write(trends)
            missing_values = pandas_agent.run(f"Determine the extent of missing values of {user_question_variable}")
            st.write(missing_values)
            return
        
        @st.cache_data
        def function_question_dataframe():
            dataframe_info = pandas_agent.run(user_question_dataframe)
            st.write(dataframe_info)
            return

        #Main

        st.header('Exploratory data analysis')
        st.subheader('General information about the dataset')

        with st.sidebar:
            with st.expander('What are the steps of EDA'):
                st.write(steps_eda())

        function_agent()

        st.subheader('Variable of study')
        user_question_variable = st.text_input('What variable are you interested in')
        if user_question_variable is not None and user_question_variable !="":
            function_question_variable()

            st.subheader('Further study')

        if user_question_variable:
            user_question_dataframe = st.text_input( "Is there anything else you would like to know about your dataframe?")
            if user_question_dataframe is not None and user_question_dataframe not in ("","no","No"):
                function_question_dataframe()
            if user_question_dataframe in ("no", "No"):
                st.write("")
