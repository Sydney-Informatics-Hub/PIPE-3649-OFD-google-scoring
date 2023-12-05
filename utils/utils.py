import glob
import os
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import inflect
import csv
import pandas as pd
import statistics
p = inflect.engine()

def clean_google_categories(categories):
    categories = str(categories).lower()
    categories = categories.replace("take away", "takeout")
    categories = categories.replace("takeaway", "takeout")
    categories = categories.replace("&", ",")
    categories = categories.replace(" and ", ",")
    categories = categories.split(",")
    final_categories = set()
    for c in categories:
        if c != "" and c != "nan":
            final_categories.add(clean_single_category(c))
    return ",".join(list(final_categories))

def clean_single_category(category):
    category = category.strip().lower()
    category = category.replace("-", " ")
    category = category.replace("/", " ")
    # replace multiple white spaces with single white space
    category = " ".join(category.split())  # replace multiple white spaces with single white space

    # convert plural forms to singular
    category = p.singular_noun(category) or category

    # words to remove
    remove_list = ['restaurant', 'shop', 'store']
    category_list =category.split()
    if len(category_list) > 1:
        category = ' '.join([c for c in category_list if c not in remove_list])
        category = category.replace('barbecue', 'bbq')
        category = category.replace('hamburger', 'burger')
    return category


# function that checks if two category strings contain the same words
def same_categories(categories_old, categories_new):
    """Checks if two category strings contain the same words"""
    words_old = set([clean_google_categories(c) for c in categories_old.split(",") if c != ""])
    words_new = set([clean_google_categories(c) for c in categories_new.split(",") if c != ""])

    # Check if the sets of words are equal
    # print("words_old:", words_old, "words_new:", words_new, words_old == words_new)
    return words_old == words_new

def train_score(train:pd.DataFrame, column_to_train_on:str) -> tuple[dict, dict]:
    google_train_columns = {}
    for index, row in train.iterrows():
        google_categories_list = str(row[column_to_train_on]).split(',')
        for category in google_categories_list:
            if category in google_train_columns:
                google_train_columns[category].append(float(row['average_health_score']))
            else:
                google_train_columns[category] = [float(row['average_health_score'])]

    google_categories_score = {}
    for k, v in google_train_columns.items():
        # print(k, v)
        if k == "nan":
            continue
        if len(v) > 0:
            try:
                google_categories_score[k] = (sum(v) / len(v), statistics.stdev(v))
            except statistics.StatisticsError:
                google_categories_score[k] = (sum(v) / len(v), 0)

    return google_categories_score, google_train_columns

def test_score(test:pd.DataFrame, google_categories_score:dict, column_to_train_on:str):

    total_diff = []
    # iterate through each row in test data frame
    for index, row in test.iterrows():
        categories = str(row[column_to_train_on]).split(',')
        try:
            score = calculate_score(google_categories_score, categories)
            diff = (float(row['average_health_score']), score)
            total_diff.append(diff)
        except ZeroDivisionError:
            pass
    return total_diff

def calculate_score(google_categories_score, categories) -> float:
    score_with_stdev = sum([google_categories_score[category][0] * (1/google_categories_score[category][1]) for category in categories if category in google_categories_score])/sum([(1/google_categories_score[category][1]) for category in categories if category in google_categories_score])
    score_with_stdev = sum([google_categories_score[category][0] * (1/google_categories_score[category][1]**4) for category in categories if category in google_categories_score])/sum([(1/google_categories_score[category][1]**4) for category in categories if category in google_categories_score])
    score_just_avg = sum([google_categories_score[category][0] for category in categories if category in google_categories_score])/sum([1 for category in categories if category in google_categories_score])

    return score_with_stdev
    return score_just_avg

def run_k_fold(dataframe:pd.DataFrame, column_to_train_on:str, plot:bool=False):
    """
    Run k-fold cross-validation on the given dataframe using the specified column to train on.

    Args:
        dataframe (pd.DataFrame): The dataframe containing the data to be used for k-fold cross-validation.
        column_to_train_on (str): The name of the column to train the model on.

    Returns:
        None
    """
    k_folds = 10
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    print("==========================================================================")
    print(column_to_train_on)
    i = 0
    for train_index, test_index in kf.split(dataframe):
        train, test = dataframe.iloc[train_index], dataframe.iloc[test_index]
        google_categories_score, google_train_columns = train_score(train, column_to_train_on)
        # save google_categories_score to a text file
        # with open(f'google_categories_score{i}.csv', 'w') as f:
        #     for k, v in google_categories_score.items():
        #         f.write(f"{k},{v}\n")
        total_diff = test_score(test, google_categories_score, "google_categories")
        diff = [abs(diff[0] - diff[1]) for diff in total_diff]
        average_diff = sum(diff) / len(total_diff)
        diff_stdv = pd.Series(diff).std()

        print(f"Average difference: {average_diff}; Standard deviation: {diff_stdv}; Number of outlets: {len(total_diff)}; Total difference: {sum(diff)}")
        i += 1
        if plot:
            plot_scores(total_diff)


def plot_scores(total_diff):

    # Extract real scores and estimate scores from total_diff
    real_scores = [diff[0] for diff in total_diff]
    estimate_scores = [diff[1] for diff in total_diff]

    # Set the figure size
    fig, ax = plt.subplots(figsize=(6, 6))  # Adjust the width and height as desired
    # Plot the scatter plot
    ax.scatter(real_scores, estimate_scores, s=2, alpha=0.5)
    
    m, b = np.polyfit(estimate_scores, real_scores, 1)  # Compute the slope and intercept of the best fit line
    estimate_scores_line = np.linspace(min(estimate_scores), max(estimate_scores), 100)
    real_scores_line = m * estimate_scores_line + b
    ax.plot(estimate_scores_line, real_scores_line, color='blue', linestyle='--')
    ax.plot([-8, 5], [-8, 5], color='red', linestyle='-')

    ax.set_ylabel("Estimate Scores")
    ax.set_xlabel("Real Scores")
    ax.set_title("Estimate Scores vs Real Scores")

    plt.tight_layout()
    plt.show()