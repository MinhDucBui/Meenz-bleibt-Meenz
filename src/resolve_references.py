import pandas as pd
import re

files = [
    "/p/project/westai0073/MainzDictionary/output/extractor/gpt-oss-120b.csv"
]

for file in files:

    df = pd.read_csv(file)

    # Suppose your columns are 'word' and 'response_cleaner'
    # Adjust if they differ
    def resolve_reference(row, df, column="response_cleaner"):
        text = row["response_cleaner"]
        text = text.strip()

        # check for reference pattern like "[SIEHE] Apfel"
        match = re.search(r"\[SIEHE\]\s*(.+)", text)
        if match:
            ref_word = match.group(1).strip()

            # find the referenced word’s definition
            ref_row = df.loc[df["Word"].str.lower().str.strip() == ref_word.lower().strip()]
            if not ref_row.empty:
                return ref_row.iloc[0][column]
            else:
                return f"[SIEHE: {ref_word}]"
        return text


    # Apply the resolver
    df["definition_final"] = df.apply(lambda row: resolve_reference(row, df), axis=1)
    df["definition_final"] = df.apply(lambda row: resolve_reference(row, df, "definition_final"), axis=1)

    unresolved = df["definition_final"].str.contains(r"\[SIEHE\]", na=False)
    num_unresolved = unresolved.sum()
    print(num_unresolved)

    df.to_csv(file)