"""DEPRECATED"""
# import clean_category from utils.py
import sys
sys.path.append('/Users/xluo3503/Library/CloudStorage/OneDrive-TheUniversityofSydney(Staff)/work/PIPE-3649-Food Delivery Modelling/utils/')
from utils import clean_google_categories
import csv
import pandas as pd
from sklearn.model_selection import KFold


# read in google_processed.csv
# google_processed_cleaned = pd.read_csv("google_processed.csv", dtype=str)

# Clean the Category column in google_processed using a loop
# for index, row in google_processed_cleaned.iterrows():
#     google_processed_cleaned.at[index, 'google_Category'] = clean_google_categories(row['google_Category'])

# # output google_processed_cleaned to a csv
# google_processed_cleaned = pd.read_csv("google_processed_cleaned.csv", dtype=str)

# matched = pd.read_csv("../matched.csv", dtype=str)
# uber_2023_score_using_new_food_mapping = pd.read_csv("../uber_2023_score_using_new_food_mapping.csv", dtype=str)

# # inner merge matched and uber_2023_score_using_new_food_mapping based on column uber_id
# merged_df = pd.merge(matched, uber_2023_score_using_new_food_mapping, on='uber_id', how='inner')

# # then inner merge the result with google_processed_cleaned based on column google_cid
# matched_with_score = pd.merge(merged_df, google_processed_cleaned, on='google_cid', how='inner')

# # print(matched_with_score.columns.tolist())
# # only keep columns uber_Name_x, google_Name_x, average_health_score, google_cid, uber_id, categories, google_Category
# matched_with_score = matched_with_score[['uber_Name_x', 'google_Name_x', 'average_health_score', 'google_cid', 'uber_id', 'categories', 'google_Category']]

# # rename uber_Name_x to uber_Name
# matched_with_score = matched_with_score.rename(columns={'uber_Name_x': 'uber_Name', 'google_Name_x': 'google_Name'})

# # output matched_with_score to csv
# matched_with_score.to_csv("matched_with_score.csv", index=False)
matched_with_score = pd.read_csv("matched_tages_with_score.csv", dtype=str)


def train_score(train, column_to_train_on):
    google_categories = {}
    for index, row in train.iterrows():
        google_categories_list = str(row[column_to_train_on]).split(',')
        for category in google_categories_list:
            if category in google_categories:
                google_categories[category][0] += float(row['average_health_score'])
                google_categories[category][1] += 1
            else:
                google_categories[category] = [float(row['average_health_score']), 1]

    google_categories_score = {}
    for k, v in google_categories.items():
        google_categories_score[k] = v[0] / v[1]
        # print(k, v, v[0] / v[1])

    return google_categories_score

def test_score(test, google_categories_score:dict, column_to_train_on:str):
    
    count_outlet = 0
    total_diff = 0
    # iterate through each row in test data frame
    for index, row in test.iterrows():
        categories = str(row[column_to_train_on]).split(',')
        categories_count = len(categories)

        score = sum([google_categories_score[category] for category in categories if category in google_categories_score])/categories_count

        if score:
            diff = abs(float(row['average_health_score']) - score)
            # print(row['google_Name'], float(row['average_health_score']), score, diff)
            total_diff += diff
            count_outlet += 1
        else:
            print(row['google_Name'], row[column_to_train_on])
            pass
    return total_diff, count_outlet


k_folds = 10
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
i = 0
for train_index, test_index in kf.split(matched_with_score):
    train, test = matched_with_score.iloc[train_index], matched_with_score.iloc[test_index]
    google_categories_score = train_score(train, "google_categories")
    # save google_categories_score to a text file
    with open(f'google_categories_score{i}.csv', 'w') as f:
        for k, v in google_categories_score.items():
            f.write(f"{k},{v}\n")
    
    total_diff, count_outlet = (test_score(test, google_categories_score, "google_categories"))
    average_diff = total_diff / count_outlet

    print(f"Average difference: {average_diff}; Number of outlets: {count_outlet}; Total difference: {total_diff}")
    i += 1
