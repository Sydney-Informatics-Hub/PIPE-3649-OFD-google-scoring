import inflect
import csv
import pandas as pd


p = inflect.engine()
def clean_category(category):
    category = category.lower()
    category = category.replace("-", " ")
    category = category.replace("/", " ")
    category = category.replace("&", " and ")
    # replace multiple white spaces with single white space
    category = " ".join(category.split())  # replace multiple white spaces with single white space

    # convert plural forms to singular
    category = p.singular_noun(category) or category
    return category


# function that checks if two category strings contain the same words
def same_categories(categories_old, categories_new):
    """Checks if two category strings contain the same words"""
    words_old = set([clean_category(c) for c in categories_old.split(",") if c != ""])
    words_new = set([clean_category(c) for c in categories_new.split(",") if c != ""])

    # Check if the sets of words are equal
    # print("words_old:", words_old, "words_new:", words_new, words_old == words_new)
    return words_old == words_new


food_mapping = {}
food_mapping_count = {}

with open('FoodMapping.csv', newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        food_mapping_count[row['\ufeffCategoryTrain']] = (float(row['weightedAvgFoodScore']), int(row['number']))


# unify food_mapping for different cases and plural form etc.
with open('FoodMapping.csv', newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # print("before:", row['\ufeffCategoryTrain'])
        category = clean_category(row['\ufeffCategoryTrain'])
        # print("after:", category)
        if category in food_mapping.keys():
            total_score = 0
            count = 0
            for k, v in food_mapping_count.items():
                if clean_category(k) == category:
                    total_score += v[0] * v[1]
                    count += v[1]

            total_score += float(row['weightedAvgFoodScore']) * int(row['number'])
            count += int(row['number'])

            food_mapping[category] = total_score / count
        else:
            food_mapping[category] = float(row['weightedAvgFoodScore'])


# save food_mapping to file
with open('food_mapping.txt', 'w') as file:
    for key, value in food_mapping.items():
        file.write(f"{key}: {value}\n")

class UberOutlet:
    unmapped_food = set()
    def __init__(self, name, uber_id, categories, food_mapping):
        self.name = name
        self.uber_id = uber_id
        self.categories = [clean_category(c) for c in categories if c != ""]
        self.food_mapping = food_mapping
        self.score = self.calculate_average_food_score()

    def __str__(self):
        return f"UberOutlet - Name: {self.name}, Uber ID: {self.uber_id}, Categories: {', '.join(self.categories)}, Score: {self.score}"

    # use food_mapping to calculate the average food score
    def calculate_average_food_score(self):
        total_score = 0
        count = 0
        for category in self.categories:
            try:
                total_score += self.food_mapping[category]
                count += 1
            except KeyError:
                UberOutlet.unmapped_food.add(category)
                continue
        if count > 0:
            average_score = total_score / count
        else:
            average_score = 0
        return average_score


# Load the full_uber_2023.csv file
full_uber_2023 = []

with open('full_uber_2023.csv', newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # print(row['uber_Name'], row['uber_id'], row['uber_Categories'])
        outlet = UberOutlet(row['uber_Name'], row['uber_id'], list(set(row['uber_Categories'].split(','))), food_mapping)
        full_uber_2023.append(outlet)
        # print(outlet)

# create a dataframe from full_uber_2023 with columns uber_Name, uber_id, average_health_score
df = []
for outlet in full_uber_2023:
    df.append([outlet.name, outlet.uber_id, outlet.score, ",".join(outlet.categories)])
uber_2023_score_using_food_mapping = pd.DataFrame(df, columns=['uber_Name', 'uber_id', 'average_health_score', 'categories'])
uber_2023_score_using_food_mapping.to_csv('uber_2023_score_using_old_food_mapping.csv', index=False)

# read in full_uber_2021.csv and join with uber_2023_score_using_food_mapping by uber_id column
full_uber_2021 = pd.read_csv('full_uber_2021.csv', dtype=str)
# Clean the Category column in full_uber_2021 using a loop
for index, row in full_uber_2021.iterrows():
    # print(row['uber_id'], row['Category'])
    categories = row['Category'].split("; ")
    cleaned_category = [clean_category(c) for c in categories if c != ""]
    full_uber_2021.at[index, 'Category'] = ",".join(cleaned_category)


# Join with uber_2023_score_using_food_mapping by uber_id column
merged_df = pd.merge(full_uber_2021, uber_2023_score_using_food_mapping, on='uber_id')

# print(merged_df.columns.tolist())
merged_df = merged_df[['uber_id', 'wAvg', 'average_health_score', 'uber_Name_x', 'uber_Name_y', 'Category', 'categories']]

merged_df.to_csv("merged_old_new_uber_scores.csv", index=False)

# count the number of rows with average_health_score same as wAvg
count_same_scores = merged_df[merged_df['average_health_score'] == merged_df['wAvg']].shape[0]
print(count_same_scores)

# Extract rows that have average_health_score same as wAvg and same categories using a loop
same_scores_same_categories_df = []
diff_scores_same_categories_df = []
diff_categories_df = []
for index, row in merged_df.iterrows():
    # if rounded to 3dp and the numbers are same
    if round(float(row['average_health_score']), 5) == round(float(row['wAvg']), 5) and same_categories(row['Category'], row['categories']):
        same_scores_same_categories_df.append(row)
    elif row['average_health_score'] != row['wAvg'] and same_categories(row['Category'], row['categories']):
        diff_scores_same_categories_df.append(row)
    elif not same_categories(row['Category'], row['categories']):
        diff_categories_df.append(row)

same_scores_same_categories_df = pd.DataFrame(same_scores_same_categories_df)
diff_scores_same_categories_df = pd.DataFrame(diff_scores_same_categories_df)
diff_categories_df = pd.DataFrame(diff_categories_df)

# output the 3 df to csv
same_scores_same_categories_df.to_csv("same_scores_same_categories_df.csv", index=False)
diff_scores_same_categories_df.to_csv("diff_scores_same_categories_df.csv", index=False)
diff_categories_df.to_csv("diff_categories_df.csv", index=False)



"""create mapping for initially unmapped food categories"""

# output the unmapped food categories to a file
with open('unmapped_food_categories.txt', 'w') as file:
    print(UberOutlet.unmapped_food, len(UberOutlet.unmapped_food))
    for category in UberOutlet.unmapped_food:
        file.write(category + '\n')

# create a dictionary for the UberOutlet.unmapped_food categories
unmapped_food_dict = {}
for category in UberOutlet.unmapped_food:
    unmapped_food_dict[category] = [0, 0]  # [sum_score, count]

# iterate through each row in the uber_2023_score_using_food_mapping
for index, row in uber_2023_score_using_food_mapping.iterrows():
    outlet_categories = row['categories'].split(',')
    # print(outlet_categories)
    for k, v in unmapped_food_dict.items():
        if k in outlet_categories:
            v[0] += row['average_health_score']
            v[1] += 1

unmapped_food_score = {}
for k, v in unmapped_food_dict.items():
    if v[1] > 0:
        unmapped_food_score[k] = v[0] / v[1]
    else:
        unmapped_food_score[k] = 0

# output unmapped_food_score to a file
with open('unmapped_food_score.txt', 'w') as file:
    for category, score in unmapped_food_score.items():
        file.write(f"{category}: {score}\n")
        
        
# combine food_mapping and unmapped_food_score into a single dictionary
combined_food_score = {**food_mapping, **unmapped_food_score}
print(combined_food_score)

# output combined_dict to a file
with open('combined_food_score.txt', 'w') as file:
    for key, value in combined_food_score.items():
        file.write(f"{key}: {value}\n")
        
full_uber_2023 = []

with open('full_uber_2023.csv', newline='') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # print(row['uber_Name'], row['uber_id'], row['uber_Categories'])
        outlet = UberOutlet(row['uber_Name'], row['uber_id'], list(set(row['uber_Categories'].split(','))), combined_food_score)
        full_uber_2023.append(outlet)
        # print(outlet)

# create a dataframe from full_uber_2023 with columns uber_Name, uber_id, average_health_score
df = []
for outlet in full_uber_2023:
    df.append([outlet.name, outlet.uber_id, outlet.score, ",".join(outlet.categories)])
uber_2023_score_using_food_mapping = pd.DataFrame(df, columns=['uber_Name', 'uber_id', 'average_health_score', 'categories'])
uber_2023_score_using_food_mapping.to_csv('uber_2023_score_using_new_food_mapping.csv', index=False)
