import spacy
import pandas as pd

df_ingredients = pd.read_csv("Ingredients.csv")

# Preprocess ingredients: lowercase, strip spaces, and remove duplicates
df_ingredients["Ingredient"] = df_ingredients["Ingredient"].str.lower().str.strip()

ingredients = set(df_ingredients["Ingredient"].value_counts().head(1000).index)
nutritions = ["calories", "fat", "saturated fat", "cholesterol", "carbohydrates", "fiber", "sugar", "protein"]
for nutrition in nutritions:
    ingredients.add(nutrition)

nlp = spacy.load("en_core_web_lg")


def token_check(review, ingredients):
    doc = nlp(review)
    found = {token.text.lower() for token in doc if token.text.lower() in ingredients}  # Use set for unique values only
    return list(found)  # Convert back to list


# Example usage
review = "I love having butter in my food"
print(token_check(review, ingredients))

df_reviews = pd.read_csv("Reviews.csv", quotechar='"')
df_reviews = df_reviews.head(5000)

rows_list=[]

from tqdm import tqdm

for i, row in tqdm(df_reviews.iterrows()):
    mentions = token_check(row["Review"], ingredients)
    for mention in mentions:
        rows_list.append({"ReviewId": row["ReviewId"], "mentions": mention})

df_mentions=pd.DataFrame(rows_list)
df_mentions.to_csv("Review_Mentions.csv", index=False)
