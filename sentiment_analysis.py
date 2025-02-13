import pandas as pd

# First try to read in naively
df = pd.read_csv(
    "Reviews.txt",
    sep="\t",
    on_bad_lines="skip"
)
# Check for NaN for all required cols and drop them
required_cols = [
    "ReviewId",  # should be numeric
    "RecipeId",  # should be numeric
    "AuthorId",  # should be numeric
    "AuthorName",  # text
    "Review",  # text
    "DateSubmitted",  # datetime
    "DateModified"  # datetime
]

df.dropna(subset=required_cols, inplace=True)

# Convert Id's to numeric value and if conversion fails drop them
for col in ["ReviewId", "RecipeId", "AuthorId"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.dropna(subset=["ReviewId", "RecipeId", "AuthorId"], inplace=True)

# Convert dates to datetime and if conversion fails drop them
for col in ["DateSubmitted", "DateModified"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")

df.dropna(subset=["DateSubmitted", "DateModified"], inplace=True)

# Convert the rest to the correct datatype
df["ReviewId"] = df["ReviewId"].astype(int)
df["RecipeId"] = df["RecipeId"].astype(int)
df["AuthorId"] = df["AuthorId"].astype(int)

df["AuthorName"] = df["AuthorName"].astype(str)
df["Review"] = df["Review"].astype(str)

print(df.info())
print(df.head())

# Create a sentiment analysis using Meta's Llama-3.2-3B-Instruct model

system_prompt = """
You are a sentinent analysis assistant. You will receive a review for a recipe online and will determine a sentiment for that review. A sentinent either be positive, neutral or negative. You will also give a confidence score between 0 and 1 for that sentinent.
Please stick to the following JSON format:

{
    "sentiment": <postive, neutral, negative>,
    "confidence": <float between 0 and 1>     
}

"""

import json

def validate_sentiment_output(output_str):
    try:
        data = json.loads(output_str)
    except json.JSONDecodeError:
        raise ValueError("Output is not valid JSON.")

    # Check for required keys
    if 'sentiment' not in data or 'confidence' not in data:
        raise ValueError("JSON must contain 'sentiment' and 'confidence' keys.")

    sentiment = data['sentiment']
    if sentiment not in ["positive", "neutral", "negative"]:
        raise ValueError(f"Invalid sentiment value: '{sentiment}'. Must be 'positive', 'neutral', or 'negative'.")

    # Validate confidence
    confidence = data['confidence']
    if not isinstance(confidence, (int, float)):
        raise ValueError("Confidence must be a number (int or float).")
    if not (0 <= confidence <= 1):
        raise ValueError("Confidence must be between 0 and 1, inclusive.")

    return data


from LLM_Pipeline import ModelPipeline
# Initialize modelpipeline
model = ModelPipeline("Llama-3.2-3B-Instruct", max_length=256, temperature=0.3)
# Only take the first 500 rows ignore the rest
df=df.head(500).copy()
# Create new columns for sentiment and confidence
df["sentiment"] = ""
df["confidence"] = 0.0

limit = 500

from tqdm import tqdm

for i in tqdm(range(limit)):
    review_text = df.iloc[i]["Review"]
    print("Review Text: ", review_text)

    # Call the model
    model_output = model.generate(prompt=review_text, system_prompt=system_prompt)
    print("Model Output: ", model_output)

    # Validate and parse the output
    try:
        result = validate_sentiment_output(model_output)
        df.at[i, "sentiment"] = result["sentiment"]
        df.at[i, "confidence"] = result["confidence"]
    except ValueError as ve:
        print(f"Row {i} - Validation error: {ve}")
        df.at[i, "sentiment"] = "error"
        df.at[i, "confidence"] = 0.0
# Save only the Reviews with working sentiment analysis
df = df[df["confidence"] != 0.0]
# Save to a new CSV file
df.to_csv("Reviews_with_Sentiment.csv", index=False)
print("Sentiment analysis complete. Output saved to 'Reviews_with_Sentiment.csv'.")