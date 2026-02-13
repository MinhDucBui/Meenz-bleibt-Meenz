import pandas as pd


df = pd.read_csv("/p/project/westai0073/MainzDictionary/output/extractor/gpt-oss-120b_part2.csv")
df["raw_extraction"] = df["response"].str.split("assistantfinal").str[-1]
sample = df.sample(n=100, random_state=42)
df["cleaned_extraction"] = df["response_cleaner"]

# Print selected columns for evaluation
sample[["Word", "Definition", "raw_extraction", "response_cleaner"]].to_csv("/p/project/westai0073/MainzDictionary/output/human_annotation/evaluate_extractor.csv")


df = pd.read_csv("/p/project/westai0073/MainzDictionary/output/extractor/gpt-oss-120b_part2.csv")
df["model"] = df["response"].str.split("assistantfinal").str[-1]
sample = df.sample(n=100, random_state=42)

# Print selected columns for evaluation
import pandas as pd

# Load data
df = pd.read_csv("/p/project/westai0073/MainzDictionary/output/generator/Llama-3.3-70B-Instruct.csv")

# Clean evaluator column
df["response_evaluator"] = df["response_evaluator"].str.split("assistantfinal").str[-1].str.strip()

df = df[df["definition_final"] != "Keine Definition"]
df = df[~df["definition_final"].str.contains("SIEHE", na=False)]


# Subsample 50 "GLEICH" and 50 "UNTERSCHIEDLICH" if available
gleich_sample = df[df["response_evaluator"] == "GLEICH"].sample(n=50, random_state=42)
unterschiedlich_sample = df[df["response_evaluator"] == "UNTERSCHIEDLICH"].sample(n=50, random_state=42)

# Combine
sample = pd.concat([gleich_sample, unterschiedlich_sample]).sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
output_path = "/p/project/westai0073/MainzDictionary/output/human_annotation/evaluate_evaluator.csv"
#sample[["Word", "definition_final", "response_generator", "response_evaluator"]].to_csv(output_path, index=False)

print(f"Saved balanced sample to {output_path}")
print(sample["response_evaluator"].value_counts())
