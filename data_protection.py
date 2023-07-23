#!/usr/bin/env python
# coding: utf-8

# In[476]:

# from main_code import main
import pandas as pd
import ast
from tabulate import tabulate
import csv
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
import statistics
from IPython.display import display, HTML
import seaborn as sns
import streamlit as st
import re
from prettytable import PrettyTable
from termcolor import colored
import requests

from streamlit_jupyter import StreamlitPatcher, tqdm

StreamlitPatcher().jupyter()  # register streamlit with jupyter-compatible wrappers

widget_counter = 0
# for saving unique names of users 
students = set()
# for saving "user: avarage score" dictionary
student_average_scores = {}

tests_passed = {}
# for saving percentage of each test passed by user
percentage_of_success_values = []
parent_chapter_pids = []
success_scores = []

def call_api():
    url = "https://lrs-tst.fernuni.ch:8089/score/?organisation=6319b81fd6d577048b550c26"
    headers = {
      'Authorization': 'Basic Mzk5OTBjYzA0N2QyYmI5ZWYxNWJiOTk5M2I2N2I3ZjY3NzM4YTMyZjozMjg2ZmFmZDM3MDJkNDU1NzIzMDk2MjhhODRjNTdiM2Vh'
    }
    response = requests.get(url, headers=headers, verify=False)  # Ignorer la vérification du certificat
    data = response.json()
    return data

# def get_passed_data(data):
#     # List to store the extracted data
#     extracted_data = []

#     # Loop through each statement in the data
#     for statement in data['statements']:
#         # Extract relevant fields from the statement
#         actor = statement['statement']['actor']['name']
#         timestamp = statement['statement']['timestamp']
#         result = statement['statement']['verb']['display']['en']
#         extension_data = statement['statement']['object']['definition']['extensions']['http://lrs&46;learninglocker&46;net/define/extensions/kairos']

#         result_details =  extension_data['results']
#         # Extract specific extension fields
#         uuid = extension_data['uuid']
#         parent_chapter_pid = extension_data['parent_chapter_pid']
        
#         # Check if the 'percentage_of_success' key exists in the extensions dictionary
#         percentage_of_success = extension_data.get('percentage_of_success')
#         # Set a default value if 'percentage_of_success' key does not exist
#         if percentage_of_success is None:
#             percentage_of_success = -1  # You can set any default value here

#         # Append the extracted data to the list
#         extracted_data.append([actor, timestamp, result, uuid, parent_chapter_pid, percentage_of_success])

#     # Create a DataFrame from the extracted data
#     df = pd.DataFrame(extracted_data, columns=["actor", "timestamp", "result", 'result_details' "uuid", "parent_chapter_pid", "percentage_of_success"])

#     # Filter the DataFrame to keep only the "passed" statements
#     passed_results = df[df['result'] == 'passed']
#     passed_results = passed_results.drop_duplicates(subset=["actor", "parent_chapter_pid"])

#     return passed_results

# # Call the function to process the passed data from the API data
# passed_data = get_passed_data(api_data)
def get_passed_data(data):
    # List to store the extracted data
    extracted_data = []

    # Loop through each statement in the data
    for statement in data['statements']:
        # Extract relevant fields from the statement
        actor = statement['statement']['actor']['name']
        timestamp = statement['statement']['timestamp']
        result = statement['statement']['verb']['display']['en']
        extension_data = statement['statement']['object']['definition']['extensions']['http://lrs&46;learninglocker&46;net/define/extensions/kairos']
        
        if 'results' in extension_data:
            result_details = extension_data['results']
        else:
            result_details = None

        # Extract specific extension fields
        uuid = extension_data['uuid']
        parent_chapter_pid = extension_data['parent_chapter_pid']
    
        percentage_of_success = extension_data.get('percentage_of_success')
        if percentage_of_success is None:
            percentage_of_success = -1  # You can set any default value here

        # Append the extracted data to the list
        extracted_data.append([actor, timestamp, result, uuid, parent_chapter_pid, percentage_of_success, result_details])

    # Create a DataFrame from the extracted data
    df = pd.DataFrame(extracted_data, columns=["actor", "timestamp", "result", "uuid", "parent_chapter_pid", "percentage_of_success", "result_details"])

    # Filter the DataFrame to keep only the "passed" statements
    passed_results = df[df['result'] == 'passed']
    passed_results = passed_results.drop_duplicates(subset=["actor", "parent_chapter_pid"])

    return passed_results

# Call the function to process the passed data from the API data
# passed_data = get_passed_data(api_data)
# st.dataframe(passed_data)

def extract_comments_and_links(data):
    rows = []
    for entry in data['statements']:
        statement = entry.get('statement')
        if statement:
            verb = statement.get('verb', {}).get('display', {}).get('en', '')
            if verb == 'commented':
                comment_type = statement['object']['definition']['extensions']['http://lrs&46;learninglocker&46;net/define/extensions/kairos']['comment_type']
                comment = statement['object']['definition']['extensions']['http://lrs&46;learninglocker&46;net/define/extensions/kairos']['comment']
                question_link = statement['object']['id']
                rows.append([comment_type, comment, question_link, statement['actor']['name']])
    return rows

# df = pd.DataFrame(api_data)

# Call the function to extract comments and links
# commented = extract_comments_and_links(df)

def print_comment(df):
    # Convert the result into a DataFrame
    df = pd.DataFrame(commented, columns=['Comment Type', 'Comment', 'Question Link', 'User Name'])

    # Display the table using Streamlit
    st.table(df)

# Function to replace "parent_chapter_pid" field with a string value
def replace_parent_chapter_pid(row):
    parent_chapter_pid = row["parent_chapter_pid"]
    if parent_chapter_pid == 333:
        return "Partie I"  
    elif parent_chapter_pid == 334:
        return "Partie II" 
    elif parent_chapter_pid == 336:
        return "Partie III" 
    elif parent_chapter_pid == 337:
        return "Partie IV"
    else: 
        return "Partie V"

def create_table_results(df):
    df["parent_chapter_pid"] = df.apply(replace_parent_chapter_pid, axis=1)
    extracted_data = pd.DataFrame(df.apply(lambda x: [
        # convert_email(x["actor"]),
        x["actor"],
        str(round(x["percentage_of_success"])) + "%",
        x["parent_chapter_pid"],
        "Passed" if x["percentage_of_success"] >= 70 else "Failed"
    ], axis=1).tolist(), columns=["actor", "object", "parent_chapter_pid", "Result"])

    # Create a list of headers for the grid
    headers = ["Name of Student", "Percentage of Success", "Chapter", "Pass/Fail"]

    # print(tabulate(extracted_data, headers=headers, colalign=('left', 'left', 'right', 'left', 'left'), tablefmt="fancy_grid"))
    # st.table(extracted_data)
    
def convert_email(name):
    name = str(name)
    match = re.search(r"([a-zA-Z]+)\.([a-zA-Z]+)@", name)
    if match:
        edited_name = f"{match.group(1).capitalize()} {match.group(2).capitalize()}"
        name = edited_name
    return name

def convert_date_format(date_str):
    pattern = r'^(\d{4})-(\d{2})-(\d{2})T.*$'
    match = re.match(pattern, date_str)
    if match:
        year, month, day = match.groups()
        # Format the date as "dd.mm.yyyy"
        formatted_date = f"{day}.{month}.{year}"
        return formatted_date

    return formatted_date

# create_table_results(passed_data)

def create_table_drill_results(df):
    st.markdown('<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-4CkC4QsNp3DxhUGpSrfL3p6B/bYqo4bMMZw5kb1+e7zF0zqH5Bz2sDc3o27kAiS0" crossorigin="anonymous">', unsafe_allow_html=True)
    global widget_counter
    # Prepare table headers
    headers = ["Name", "Result", "Status"]
    # Prepare table rows
    rows = []

    for _, row in df.iterrows():
        name = row["actor"]
        name = str(name)
        # name = convert_email(name)
        students.add(name)
        percentage_of_success = row["percentage_of_success"]

        if percentage_of_success is not None:
            percentage_of_success = int(percentage_of_success)
            rows.append([name, percentage_of_success])
            percentage_of_success_values.append(percentage_of_success)

    # Generate per-student average percentage of success
    for student in students:
        student_scores = [score for name, score in rows if name == student]
        average_score = statistics.mean(student_scores)
        student_average_scores[student] = average_score

    # Add rows to the table
    table_rows = []
    for student in students:
        average_percentage = student_average_scores[student]
        verification_status = "Validated" if average_percentage >= 70 else "Not Validated"
        table_rows.append([student, str(round(average_percentage)) + "%", verification_status])

# Create a DataFrame from the table_rows
    table_df = pd.DataFrame(table_rows, columns=headers)
    
    # Apply styles to the Status column
    def apply_status_style(status):
        return "background-color: lightgreen" if status == "Validated" else "background-color: pink"

    # Apply styles to the DataFrame
    styled_df = table_df.style.applymap(lambda value: "background-color: lightgreen" if "%" in value and float(value[:-1]) >= 70 else "", subset=["Result"]) \
                               .applymap(apply_status_style, subset=["Status"])

    # # Display the styled DataFrame using Streamlit
    # st.dataframe(styled_df)
    # # st.table(table_rows)
    # # Increment the counter for each call to this function
    # Display the styled DataFrame using Streamlit
    st.markdown('<h3>Student Performance Summary</h3>', unsafe_allow_html=True)
    st.dataframe(styled_df)

    # Generate a unique key for the selectbox widget
    selectbox_key = f"student_selector_{widget_counter}"

    # Display the table using Streamlit
    selected_student = st.selectbox("Select a student:", list(students), key=selectbox_key)
    widget_counter += 1

    # Filter the data based on the selected student
    filtered_data = df[df["actor"] == selected_student]

    # Prepare table headers for the detailed information
    headers_details = ["Chapter", "Score", "Time"]

    # Prepare table rows for the detailed information
    rows_details = []
    for _, row in filtered_data.iterrows():
        # chapter = replace_parent_chapter_pid(row)timestamp
        timestamp = convert_date_format(row['timestamp'])
        chapter = row["parent_chapter_pid"]
        score = row["percentage_of_success"]
        rows_details.append([chapter, str(round(score)) + "%", timestamp])

 # Display the detailed information table
    # st.write("Detailed Information:")
    # st.dataframe(details_df)
    # Display the detailed information table
    details_df = pd.DataFrame(rows_details, columns=headers_details)
    st.markdown('<h3>Detailed Information</h3>', unsafe_allow_html=True)
    st.dataframe(details_df)

# create_table_drill_results(passed_data)

def avarage_success_per_chapter(df):

    # Loop through each statement in the data
    for _, row in df.iterrows():
        # Check if the 'percentage_of_success' key exists in the extensions dictionary
        percentage_of_success = row["percentage_of_success"]
        parent_chapter_pid = row["parent_chapter_pid"]
        # Set a default value if 'percentage_of_success' key does not exist
        if percentage_of_success is not None and parent_chapter_pid is not None:
            parent_chapter_pids.append(str(parent_chapter_pid))
            success_scores.append(percentage_of_success)
            
    # Calculate average success score per parent_chapter_pid
    unique_pids = list(set(parent_chapter_pids))
    average_scores = []
    for pid in unique_pids:
        scores = [score for score, p in zip(success_scores, parent_chapter_pids) if p == pid]
        average_score = np.mean(scores)
        average_scores.append(average_score)

    # Sort unique_pids and average_scores based on unique_pids values
    unique_pids, average_scores = zip(*sorted(zip(unique_pids, average_scores)))
    
#     unique_pids_edit = [replace_parent_chapter_pid(pid) for pid in unique_pids]

    # Define a color palette for the bars
    colors = plt.cm.get_cmap('tab10', len(unique_pids))

    # Plot the graph
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.bar(unique_pids, average_scores, color=colors(range(len(unique_pids))))
    ax.set_xlabel("Parent Chapter PID")
    ax.set_ylabel("Average Success Score")
    ax.set_title("Average Success Score per Parent Chapter PID")

    # Add a legend for the colors
    ax.legend(bars, unique_pids)

    # Add percentage labels on each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        percentage = f"{average_scores[i]:.2f}%"
        ax.text(bar.get_x() + bar.get_width() / 2, height, percentage, ha='center', va='bottom')

    # Display the plot using Streamlit
    st.pyplot(fig)
    display(fig)

# avarage_success_per_chapter(passed_data)


def avarage_passing_percentages_per_chapter(df):
    # Group the DataFrame by parent_chapter_pid and calculate the average passing percentage for each chapter
    grouped_df = df.groupby('parent_chapter_pid')['percentage_of_success'].mean()

    # Filter out the chapters with NaN values
    grouped_df = grouped_df.dropna()

    # Sort the chapters based on their average passing percentages
    grouped_df = grouped_df.sort_values(ascending=False)

    # Get the parent_chapter_pid and passing percentages from the grouped DataFrame
    unique_pids = grouped_df.index
    passing_percentages = grouped_df.values

    # Define a color palette for the bars
    colors = plt.cm.get_cmap('tab10', len(unique_pids))

    # Plot the graph with a smaller figure size
    fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the figure size as per your preference
    bars = ax.bar(unique_pids, passing_percentages, color=colors(range(len(unique_pids))))
    ax.set_xlabel("Parent Chapter PID")
    ax.set_ylabel("Percentage of passing (>70%)")
    ax.set_title("Passing percentage per Parent Chapter PID")

    # Add a legend for the colors
    ax.legend(bars, unique_pids)

    # Add percentage labels on each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        percentage =  f"{passing_percentages[i]:.2f}%"
        ax.text(i, height, percentage, ha='center', va='bottom')

    # Display the plot using Streamlit
    st.pyplot(fig)

# Call the function in your Streamlit app
# st.markdown('<h3>Average Passing Percentages per Parent Chapter PID</h3>', unsafe_allow_html=True)
# avarage_passing_percentages_per_chapter(passed_data)



def pass_and_fail_percentages_per_chapter():
    # Calculate passing and failing percentages per parent_chapter_pid
    unique_pids = list(set(parent_chapter_pids))

    passing_percentages = []
    failing_percentages = []
    for pid in unique_pids:
        scores = [score for score, p in zip(success_scores, parent_chapter_pids) if p == pid]
        passing_count = sum(score >= 70 for score in scores)
        failing_count = len(scores) - passing_count
        passing_percentage = (passing_count / len(scores)) * 100
        failing_percentage = (failing_count / len(scores)) * 100
        passing_percentages.append(passing_percentage)
        failing_percentages.append(failing_percentage)

    # Sort unique_pids and passing_percentages based on passing_percentages values
    unique_pids, passing_percentages = zip(*sorted(zip(unique_pids, passing_percentages)))

    # Plot the diagram
    plt.figure(figsize=(12, 8))
    plt.bar(unique_pids, passing_percentages, label='Passed', color='green')
    plt.bar(unique_pids, failing_percentages, bottom=passing_percentages, label='Failed', color='red')
    plt.xlabel("Parent Chapter PID")
    plt.ylabel("Percentage")
    plt.title("Pass/Fail Percentage per Parent Chapter PID")
    plt.legend()

    for i in range(len(unique_pids)):
        passing_percentage = passing_percentages[i]
        failing_percentage = failing_percentages[i]
        plt.text(unique_pids[i], passing_percentage / 2, f"{passing_percentage:.2f}%", ha='center', va='center')
        plt.text(unique_pids[i], passing_percentage + failing_percentage / 2, f"{failing_percentage:.2f}%", ha='center', va='center')

    plt.show()
    st.pyplot(plt)

# Call the function in your Streamlit app
st.markdown('<h3>Pass/Fail Percentages per Parent Chapter PID</h3>', unsafe_allow_html=True)
# pass_and_fail_percentages_per_chapter()
    
def student_list(df):
    # Prepare table headers
    headers = ["Name", "Average Percentage of Success (%)"]
    # Prepare table rows
    rows = []

    for _, row in df.iterrows():
        name = row["actor"]
        name = str(name)
        match = re.search(r"([a-zA-Z]+)\.([a-zA-Z]+)@", name)
        if match:
            # Get the edited name from the regular expression match
            edited_name = f"{match.group(1).capitalize()} {match.group(2).capitalize()}"
            name = edited_name
        students.add(name)
        percentage_of_success = row["percentage_of_success"]
        if percentage_of_success is not None:
            percentage_of_success = int(percentage_of_success)
            rows.append([name, percentage_of_success])
            percentage_of_success_values.append(percentage_of_success)


    # Generate per-student average percentage of success
    for student in students:
        student_scores = [score for name, score in rows if name == student]
        average_score = statistics.mean(student_scores)
        student_average_scores[student] = average_score

    # Add rows to the table
    table_rows = []
    for student in students:
        average_percentage = student_average_scores[student]
        table_rows.append([student, round(average_percentage)])

    st.table(table_rows)
    table = tabulate(table_rows, headers, tablefmt="fancy_grid")
    print(table)


def line_graph():
    # Line Graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(students)), list(student_average_scores.values()), marker='o')
    plt.xticks(range(len(students)), students, rotation=90)
    plt.xlabel("Student")
    plt.ylabel("Average Percentage of Success")
    plt.title("Average Percentage of Success for Each Student (Line Graph)")
    plt.show()

    st.pyplot(plt)


def bar_chart():
    # Create the bar chart using Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(list(students), list(student_average_scores.values()))
    ax.set_xlabel("Student")
    ax.set_ylabel("Average Percentage of Success")
    ax.set_title("Scores of Individual Students")
    ax.tick_params(axis='x', rotation=90)  # Rotate x-axis labels for better readability

    # Display the chart using Streamlit
    st.pyplot(fig)
    plt.show()


# # Plot the number of students passed each test

def table_success_students(df):
    headers = ["Name", "Average Percentage of Success (%)"]
    tests = [key for key in tests_passed.keys() if key != 335]
    pass_counts = [tests_passed[key] for key in tests]

    # Create a list of rows
    table_rows = list(zip(tests, pass_counts))

    # Create a pandas DataFrame
    df = pd.DataFrame(table_rows, columns=headers)

    # Display the table using st.table
    st.table(df)
    
def summary_statistics():
    # Compute summary statistics
    mean = statistics.mean(percentage_of_success_values)
    median = statistics.median(percentage_of_success_values)
    mode = statistics.mode(percentage_of_success_values)
    minimum = min(percentage_of_success_values)
    maximum = max(percentage_of_success_values)
    standard_deviation = statistics.stdev(percentage_of_success_values)

    # Print the summary statistics
    st.markdown("### Summary Statistics for percentage_of_success:")
    st.markdown(f"- Mean: {mean}")
    st.markdown(f"- Median: {median}")
    st.markdown(f"- Mode: {mode}")
    st.markdown(f"- Minimum: {minimum}")
    st.markdown(f"- Maximum: {maximum}")
    st.markdown(f"- Standard Deviation: {standard_deviation}")

def plot_the_distribution(): 
    # Plot the distribution of percentage_of_success
    plt.figure(figsize=(8, 6))
    sns.histplot(percentage_of_success_values, kde=True)
    plt.xlabel("Percentage of Success")
    plt.ylabel("Frequency")
    plt.title("Distribution of Percentage of Success")
    st.pyplot(plt)
    plt.show()


# Initialize the students dictionary
def individual_student_analysis(df):
    students = {}

    # Prepare table rows
    rows = []

    for _, row in df.iterrows():
        actor_data = eval(row["actor"])
        name = actor_data["name"]
        students.setdefault(name, [])  # Initialize an empty list for each student if not already present
        object_data = eval(row["object"])
        extensions = object_data["definition"]["extensions"]["http://lrs&46;learninglocker&46;net/define/extensions/kairos"]
        percentage_of_success = extensions["percentage_of_success"]
        if percentage_of_success is not None:
            percentage_of_success_values.append(int(percentage_of_success))
            students[name].append(int(percentage_of_success))

    # Set title
    st.title("Student Analysis")

    # Create sidebar
    st.sidebar.title("Options")
    # Perform individual student analysis
    for student, scores in students.items():
        student_mean = statistics.mean(scores)
        student_median = statistics.median(scores)
        student_mode = statistics.mode(scores)
        student_minimum = min(scores)
        student_maximum = max(scores)

        if len(scores) >= 2:
            student_standard_deviation = statistics.stdev(scores)
        else:
            student_standard_deviation = 0  # Set standard deviation to 0 if there is only one data point

        # Display individual analysis
        st.write(f"\nIndividual Analysis for Student: {student}")
        st.write("Mean:", student_mean)
        st.write("Median:", student_median)
        st.write("Mode:", student_mode)
        st.write("Minimum:", student_minimum)
        st.write("Maximum:", student_maximum)
        st.write("Standard Deviation:", student_standard_deviation)

        # Generate histogram plot
        plt.figure(figsize=(8, 6))
        sns.histplot(scores, kde=True)
        plt.xlabel("Percentage of Success")
        plt.ylabel("Frequency")
        plt.title(f"Distribution of Percentage of Success for Student: {student}")
        st.pyplot(plt)

    
def main():
    api_data = call_api()

    passed_data = get_passed_data(api_data)


    st.dataframe(passed_data)

    create_table_results(passed_data)

    create_table_drill_results(passed_data)

    avarage_success_per_chapter(passed_data)

    pass_and_fail_percentages_per_chapter()

if __name__ == "__main__":
    main()




