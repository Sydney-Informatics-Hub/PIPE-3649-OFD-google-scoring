"""DEPRECATED"""
import glob
import os
import pandas as pd
import sys
from sklearn.model_selection import KFold
from utils.utils import *

# create empty dataframe df
df = pd.DataFrame()
# read in all files that starts with CID and ends with .csv in before2211 
for file in glob.glob("datasets/*.csv"):
    # for files that starts with CID and ends with .csv, combine them into 1 dataframe variable
    filename = os.path.basename(file)
    if filename.startswith("CID") and filename.endswith(".csv"):
        # print(file)
        current_df = pd.read_csv(file, dtype=str)
        # drop duplicate rows with the same cid and keep first
        current_df = current_df.drop_duplicates(subset=['cid'], keep='first')
        df = pd.concat([df, current_df], ignore_index=True)

# keep only columns cid and columns that ends with "title" and columns that starts with "categories"
df['google_tags'] = df[[col for col in df.columns if col.endswith('title') and col != 'title']].apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)
df['google_categories'] = df[[col for col in df.columns if col.startswith('categories')]].apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)
# apply clean_google_categories function to google_categories column
df['google_categories'] = df['google_categories'].apply(clean_google_categories)
df = df[['cid', 'google_tags', 'google_categories']]
# add string "a" to front of all cids
df['cid'] = "a" + df['cid']

# print(df.columns.tolist())

# join columns that ends with "title" by ",", and save into a new column called tags

# output df to a csv
# df.to_csv("google_combined_tags.csv", index=False)


# read matched_with_score.csv as a dataframe
matched_with_score = pd.read_csv("matched_with_score.csv", dtype=str)
# rename google_cid to cid
matched_with_score = matched_with_score.rename(columns={'google_cid': 'cid'})

# merge matched_with_score and df based on column cid
matched_tages_with_score = pd.merge(matched_with_score, df, on='cid', how='inner')
# print(matched_tages_with_score.columns.tolist())

# for each row, join columns google_categories and google_Category by ",", and save into a new column called google_cate
matched_tages_with_score['google_cate'] = matched_tages_with_score['google_categories'] + ',' + matched_tages_with_score['google_Category']

# remove columns google_categories and google_Category
matched_tages_with_score = matched_tages_with_score.drop(columns=['google_categories', 'google_Category'])
# rename columns google_cate to google_categories
matched_tages_with_score = matched_tages_with_score.rename(columns={'google_cate': 'google_categories'})
matched_tages_with_score['google_categories'] = matched_tages_with_score['google_categories'].apply(clean_google_categories)
matched_tages_with_score['google_tags'] = matched_tages_with_score['google_tags'].apply(clean_google_categories)

# print(matched_tages_with_score.columns.tolist())
matched_tages_with_score.to_csv("matched_tages_with_score.csv", index=False)



# k_folds = 10
# kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
# print("==========================================================================")
# print("google_tags")
# i = 0
# for train_index, test_index in kf.split(matched_tages_with_score):
#     train, test = matched_tages_with_score.iloc[train_index], matched_tages_with_score.iloc[test_index]
#     google_categories_score = train_score(train, "google_tags")
#     # save google_categories_score to a text file
#     with open(f'google_tags_score{i}.csv', 'w') as f:
#         for k, v in google_categories_score.items():
#             f.write(f"{k},{v}\n")
    
#     total_diff, count_outlet = (test_score(test, google_categories_score, "google_tags"))
#     average_diff = total_diff / count_outlet

#     print(f"Average difference: {average_diff}; Number of outlets: {count_outlet}; Total difference: {total_diff}")
#     i += 1



# print("==========================================================================")
# print("google_categories")
# i = 0
# for train_index, test_index in kf.split(matched_tages_with_score):
#     train, test = matched_tages_with_score.iloc[train_index], matched_tages_with_score.iloc[test_index]
#     google_categories_score = train_score(train, "google_categories")
#     # save google_categories_score to a text file
#     with open(f'google_categories_score{i}.csv', 'w') as f:
#         for k, v in google_categories_score.items():
#             f.write(f"{k},{v}\n")
    
#     total_diff, count_outlet = (test_score(test, google_categories_score, "google_categories"))
#     average_diff = total_diff / count_outlet

#     print(f"Average difference: {average_diff}; Number of outlets: {count_outlet}; Total difference: {total_diff}")
#     i += 1
