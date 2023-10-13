import pathlib
import fastbook
import kaggle
import shutil

def get_kaggle_dataset(comp: str) -> pathlib.Path:
    """
    Download and unzip the dataset for the provided competition name (comp).
    Return the local path to the dataset.
    
    If the local path already exists, return the local path.
    """
    kaggle_api_credentials = pathlib.Path('~/.kaggle/kaggle.json').expanduser().read_text()
    path = fastbook.URLs.path(comp)
    if path.exists():
        # print(path, "already exists.")
        return path
    path.mkdir(parents=True)
    kaggle.api.competition_download_cli(comp, path=path)
    shutil.unpack_archive(str(path/f'{comp}.zip'), str(path))
    return path

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_df(df: pd.DataFrame):
    # Set the aesthetic style of the plots
    sns.set_style("whitegrid")
    
    # For numerical features, you can use histograms or box plots:
    for column in df.select_dtypes(include=['number']).columns:
        plt.figure(figsize=(10, 5))
        sns.histplot(df[column].dropna(), kde=True)
        plt.title(f'Distribution of {column}')
        plt.show()
    
    # For categorical features, you can use bar plots:
    for column in df.select_dtypes(include=['object']).columns:
        plt.figure(figsize=(10, 5))
        sns.countplot(y=df[column].dropna())
        plt.title(f'Distribution of {column}')
        plt.show()
    
    # For date features, you can plot the count of records over time:
    for column in df.select_dtypes(include=['datetime']).columns:
        df[column] = pd.to_datetime(df[column])  # Ensure the column is datetime type
        plt.figure(figsize=(10, 5))
        df[column].dropna().dt.year.value_counts().sort_index().plot(kind='bar')
        plt.title(f'Distribution of {column} by Year')
        plt.show()

import subprocess
def submit_to_kaggle(competition_name: str, file_path: str, comment: str):
    """
    Submit a file to a specified Kaggle competition.
    
    Parameters:
    - competition_name (str): The name of the Kaggle competition.
    - file_path (str): The path to the file to be submitted.
    - comment (str): A comment describing the submission.
    
    Returns:
    - None
    """
    kaggle_executable = '/home/john/.local/bin/kaggle'
    command = [
        kaggle_executable,
        'competitions',
        'submit',
        '-c',
        competition_name,
        '-f',
        file_path,
        '-m',
        f'"{comment}"'
    ]
    
    try:
        # Execute the command
        subprocess.run(command, check=True)
        print(f'Submission successful for competition: {competition_name}')
    except subprocess.CalledProcessError as e:
        print(f'An error occurred: {e}')
        print(f'Command executed: {e.cmd}')
        print(f'Return code: {e.returncode}')



# from IPython.display import display, HTML
# 
# def notify(n, m):
#     html_str = """
#     <audio controls autoplay>
#       <source src="703380__sonically_sound__short-notification.wav" type="audio/wav">
#       Your browser does not support the audio element.
#     </audio>
#     """
# 
#     for _ in range(m):
#         for _ in range(n):
#             display(HTML(html_str))
#             !sleep 0.25
#         !sleep 2