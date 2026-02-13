import sys, os
import pandas as pd
from rapidfuzz.distance import Levenshtein  # faster than python-Levenshtein
from textdistance import jaccard
import random
random.seed(42)

RULES = """
## Practical Application Rules

### Step-by-Step Mapping Process

1. **Identify Core Root**
   - Extract the main semantic element from the dialect word
   - Example: `Klebberschulde` → focus on `Klebb` and `Schuld`

2. **Apply Sound Transformations**
   - Apply systematic sound changes based on patterns above
   - Example: `Klebb` → `Kleb` (consonant simplification)

3. **Handle Suffixes Systematically**
   - Apply standard German suffix equivalents
   - Example: `-schulde` → `-schulden` (noun pluralization)

4. **Consider Semantic Context**
   - Use definition to guide final word choice
   - Ensure the mapped word fits the semantic field

### Quick Reference Guide

| Dialect Pattern | Standard German Equivalent | Example |
|----------------|---------------------------|---------|
| `-che` | `-chen` | `Bobbelche` → `Bobby` |
| `-ele` | `-eln` | `knerchele` → `knirschen` |
| `aa` | `ei` | `Gaawer` → `Geifer` |
| Final `-e` | Often added to nouns | `Grobbe` → `Grob` |
| Compound words | Preserve first element | `Klebberschulde` → `Kleberschulden` |
| Professional `-er` | Usually preserved | `Stinkerd` → `Stinker` |

"""

def replace_word(row):
    word = row["Word"]
    definition = row["definition_final"]
    # Remove quoted version
    cleaned = definition.replace(f'"{word}" ', "<Dialektwort> ")
    # Remove the word in any casing (lower, upper, capitalized)
    for variant in [word, word.lower(), word.upper(), word.capitalize(), word + "s"]:
        for end in ["s", ""]:
            cleaned = cleaned.replace(variant + end, "<Dialektwort>")
    cleaned = cleaned.strip()
    return cleaned


def load_data_def_generation(language):

    if language == "english":
        SYSTEM_PROMPT = (
            "Du bist ein präziser und zuverlässiger linguistischer Assistent. "
            "Deine Aufgabe ist es, Wörterbuchdefinitionen zu erstellen. "
            "Gib ausschließlich eine einzige, kurze und prägnante Bedeutung des angefragten Wortes an. "
            "Nutze dabei die im English gebräuchliche Bedeutung des Wortes, "
            "formuliere die Definition auf English."
        )
        TEMPLATE = (
            "Erstelle genau eine kurze Wörterbuchdefinition für das Wort '{Word}'. "
            "Verwende dabei die im English gebräuchliche Bedeutung, "
            "ohne Zusatzinformationen, Beispiele oder alternative Bedeutungen anzugeben."
        )
        df = pd.read_csv("data/english/subsample.txt", sep="\t", header=None, names=["Word", "Definition"], encoding="utf-8")
    else:
        SYSTEM_PROMPT = (
            "Du bist ein präziser und zuverlässiger linguistischer Assistent. "
            "Deine Aufgabe ist es, Wörterbuchdefinitionen zu erstellen. "
            "Gib ausschließlich eine einzige, kurze und prägnante Bedeutung des angefragten Wortes an. "
            "Nutze dabei die im Mainzer Dialekt gebräuchliche Bedeutung des Wortes, "
            "formuliere die Definition jedoch auf Hochdeutsch."
        )
        TEMPLATE = (
            "Erstelle genau eine kurze Wörterbuchdefinition für das Wort '{Word}'. "
            "Verwende dabei die im Mainzer Dialekt gebräuchliche Bedeutung, "
            "ohne Zusatzinformationen, Beispiele oder alternative Bedeutungen anzugeben."
        )
        df = pd.read_csv("data/final_wb.txt", sep="\t", header=None, names=["Word", "Definition"], encoding="utf-8")
    df = df.iloc[1:]

    df["raw_prompts"] = df.apply(lambda row: TEMPLATE.format(**row), axis=1)
    return df, SYSTEM_PROMPT



def load_data_def_generation_rulebased(language):

    SYSTEM_PROMPT = (
        "Du bist ein präziser und zuverlässiger linguistischer Assistent. "
        "Deine Aufgabe ist es, Wörterbuchdefinitionen zu erstellen. "
        "Gib ausschließlich eine einzige, kurze und prägnante Bedeutung des angefragten Wortes an. "
        "Nutze dabei die im Mainzer Dialekt gebräuchliche Bedeutung des Wortes, "
        "formuliere die Definition jedoch auf Hochdeutsch.\n"
    )
    TEMPLATE = (
        "Verwende die folgenden vom Modell extrahierten Regeln, um Definitionen zu erstellen:\n" + RULES + "\n\n"
        "Erstelle genau eine kurze Wörterbuchdefinition für das Wort '{Word}'. "
        "Verwende dabei die im Mainzer Dialekt gebräuchliche Bedeutung, "
        "ohne Zusatzinformationen, Beispiele oder alternative Bedeutungen anzugeben."
    )
    df = pd.read_csv("data/final_wb.txt", sep="\t", header=None, names=["Word", "Definition"], encoding="utf-8")
    df = df.iloc[1:]

    df["raw_prompts"] = df.apply(lambda row: TEMPLATE.format(**row), axis=1)
    return df, SYSTEM_PROMPT


def load_data_reversed_generate_def(mode, language):

    if language == "english":
        SYSTEM_PROMPT = (
            "Du bist ein präziser und zuverlässiger linguistischer Assistent. "
            "Deine Aufgabe ist es, zu einem gegebenen Bedeutungsinhalt das passende Wort "
            "im Englischen zu finden. "
            "Gib ausschließlich ein einziges Wort aus, das diese Bedeutung im Englischen ausdrückt. "
            "Gib keine Erklärungen, Übersetzungen oder Zusatzinformationen."
        )

        TEMPLATE = (
            "Finde das Englische Wort, das die folgende Bedeutung ausdrückt:\n\n"
            "'{definition_final_generation}'\n\n"
            "Antworte nur mit dem Englischen Wort."
        )
        df = pd.read_csv("data/english/extractor.csv")
    else:
        SYSTEM_PROMPT = (
            "Du bist ein präziser und zuverlässiger linguistischer Assistent. "
            "Deine Aufgabe ist es, zu einem gegebenen Bedeutungsinhalt das passende Wort "
            "im Mainzer Dialekt zu finden. "
            "Gib ausschließlich ein einziges Wort aus, das diese Bedeutung im Mainzer Dialekt ausdrückt. "
            "Gib keine Erklärungen, Übersetzungen oder Zusatzinformationen."
        )
        TEMPLATE = (
            "Finde das Mainzer Dialektwort, das die folgende Bedeutung ausdrückt:\n\n"
            "'{definition_final_generation}'\n\n"
            "Antworte nur mit dem Dialektwort."
        )
        df = pd.read_csv("output/extractor/gpt-oss-120b.csv")

    df["definition_final_generation"] = df.apply(lambda row: replace_word(row), axis=1)
    df["definition_final_generation"] = df["definition_final_generation"].apply(lambda x: x[:500])
    print(df["definition_final_generation"].iloc[0])
    print(df["definition_final_generation"].iloc[1])

    df["raw_prompts_" + mode] = df.apply(lambda row: TEMPLATE.format(**row), axis=1)
    return df, SYSTEM_PROMPT


def load_data_reversed_generate_def_rulebased(mode, language):
    def replace_word(row):
        word = row["Word"]
        definition = row["definition_final"]
        # Remove quoted version
        cleaned = definition.replace(f'"{word}" ', "<Dialektwort> ")
        # Remove the word in any casing (lower, upper, capitalized)
        for variant in [word, word.lower(), word.upper(), word.capitalize(), word + "s"]:
            for end in ["s", ""]:
                cleaned = cleaned.replace(variant + end, "<Dialektwort>")
        cleaned = cleaned.strip()
        return cleaned

    SYSTEM_PROMPT = (
        "Du bist ein präziser und zuverlässiger linguistischer Assistent. "
        "Deine Aufgabe ist es, zu einem gegebenen Bedeutungsinhalt das passende Wort "
        "im Mainzer Dialekt zu finden. "
        "Gib ausschließlich ein einziges Wort aus, das diese Bedeutung im Mainzer Dialekt ausdrückt. "
        "Gib keine Erklärungen, Übersetzungen oder Zusatzinformationen.\n"

    )
    TEMPLATE = (
        "Verwende die folgenden vom Modell extrahierten Regeln, um das Dialektwort zu erstellen:\n" + RULES + "\n\n"
        "Finde das Mainzer Dialektwort, das die folgende Bedeutung ausdrückt:\n\n"
        "'{definition_final_generation}'\n\n"
        "Antworte nur mit dem Dialektwort."
    )
    df = pd.read_csv("output/extractor/gpt-oss-120b.csv")

    df["definition_final_generation"] = df.apply(lambda row: replace_word(row), axis=1)
    df["definition_final_generation"] = df["definition_final_generation"].apply(lambda x: x[:500])
    print(df["definition_final_generation"].iloc[0])
    print(df["definition_final_generation"].iloc[1])

    df["raw_prompts_" + mode] = df.apply(lambda row: TEMPLATE.format(**row), axis=1)
    return df, SYSTEM_PROMPT


def load_data_def_evaluator(input_file, mode):

    SYSTEM_PROMPT = (
        "Du bist ein präziser und neutraler Evaluator für Bedeutungsähnlichkeit von "
        "Wörterbuchdefinitionen. "
        "Deine Aufgabe ist es, zu beurteilen, ob zwei Definitionen inhaltlich dieselbe Bedeutung ausdrücken, "
        "auch wenn sie unterschiedlich formuliert sind. "
        "Berücksichtige Synonyme, Umformulierungen oder stilistische Unterschiede. "
        "Antworte ausschließlich mit 'GLEICH', wenn die beiden Definitionen denselben Inhalt wiedergeben, "
        "oder mit 'UNTERSCHIEDLICH', wenn sich ihre Bedeutung wesentlich unterscheidet. "
        "Wörterbucheinträge können mehrere Definitionen enthalten. Wenn mindestens eine dieser Definitionen mit der vom LLM erzeugten übereinstimmt, gilt das Ergebnis als 'GLEICH'.“"
        "Gib keine weiteren Erklärungen oder Begründungen."
    )

    TEMPLATE = (
        "Vergleiche die folgenden zwei Definitionen. "
        "Beurteile, ob sie inhaltlich gleich sind.\n\n"
        "Definition A (LLM): '{response_generator}'\n\n"
        "Definition B (Wörterbuch): '{definition_final}'"
    )


    # Load model outputs
    df_model = pd.read_csv(input_file)
    col = [c for c in df_model.columns if c.startswith("response_generator")][0]
    df_model["response_generator"] = df_model[col].str.split("assistantfinal").str[-1]
    df_model["response_generator"] = df_model["response_generator"].str.split("think>").str[-1]
    print(df_model["response_generator"].iloc[0])
    df_model = df_model.drop(columns=[c for c in ["definition_final"] if c in df_model.columns])

    # Load ground truth definitions
    if "/english/" in input_file:
        df_gt_definition = pd.read_csv("data/english/extractor.csv")
    else:
        df_gt_definition = pd.read_csv("output/extractor/gpt-oss-120b.csv")

    # Merge the two DataFrames on 'Word'
    df_merged = pd.merge(df_model, df_gt_definition[["Word", "definition_final"]], on="Word", how="inner")

    # Generate prompts
    df_merged["raw_prompts_" + mode] = df_merged.apply(lambda row: TEMPLATE.format(**row), axis=1)

    return df_merged, SYSTEM_PROMPT



def load_data_def_cleaner(input_file, mode):

    SYSTEM_PROMPT = (
        "Du bist ein sorgfältiger linguistischer Assistent. "
        "Deine Aufgabe ist es, Wörterbuchdefinitionen zu bereinigen, "
        "ohne deren Bedeutung zu verändern. Entferne ausschließlich unnötige Sonderzeichen "
        "wie Bindestriche, doppelte Leerzeichen oder ähnliche OCR-Artefakte. "
        "Lasse Verweise in der Form [SIEHE] unverändert und Nummerierungen von Definitionen."
    )

    TEMPLATE = (
        "Hier ist die Definition für das Wort '{Word}':\n\n"
        "{cleaned}\n\n"
        "Bereinige die Definition gemäß den Anweisungen. "
        "Wenn sie bereits sauber ist, gib sie unverändert zurück. "
        "Gebe nur die bereinigte Definition wieder"
    )

    df = pd.read_csv(input_file)

    df["cleaned"] = df["response_extractor"].str.split("assistantfinal").str[-1]

    df["raw_prompts_" + mode] = df.apply(lambda row: TEMPLATE.format(**row), axis=1)
    return df, SYSTEM_PROMPT


def load_data_extract_definition():

    SYSTEM_PROMPT = (
        "Du bist ein präziser und zuverlässiger Assistent zur Extraktion linguistischer Daten. "
        "Gib ausschließlich die angeforderte Information aus, ohne Inhalte zu verändern, zu kürzen oder hinzuzufügen."
    )

    TEMPLATE = (
        "Du erhältst eine unstrukturierte oder fehlerhafte Definition des Wortes '{Word}' "
        "aus einem alten Wörterbuch. Deine Aufgabe ist es, nur die eigentliche Bedeutung "
        "des Wortes aus dem Text zu extrahieren, ohne Kommentare, Reformulierungen oder Erklärungen.\n\n"

        "Regeln:\n"
        "1. Wenn mehrere Bedeutungen vorhanden sind, nummeriere sie fortlaufend (1., 2., 3., …) und trenne sie jeweils mit einem Zeilenumbruch.\n"
        "2. Wenn die Definition ausschließlich auf ein anderes Wort verweist, gib '[SIEHE] <Wort>' aus.\n"
        "3. Verändere den Originaltext nicht, sondern gib ausschließlich den relevanten Ausschnitt wieder.\n\n"
        "4. Wenn es keine Definition im Text gibt, gebe 'Keine Definition' wieder.\n\n"

        "Beispiele:\n\n"

        "Wort: Dambnudel\n"
        "Text: 'die, die Dambnudele, kurzes a tontragend, gewöhnlich und gehoben für: "
        "eine Hefeteigspeise, die in Milch, Öl oder Fett – im Krobbe (s. d.) – gedämpft und gebacken wird.'\n"
        "Output: 'eine Hefeteigspeise, die in Milch, Öl oder Fett – im Krobbe (s. d.) – gedämpft und gebacken wird'\n\n"

        "Wort: Kaut\n"
        "Text: 'die, die Kaute, ohne Betonungseigenheit, allgemein für: 1. flache Grube, – "
        "2. Bett. „ich geh in moi Kaut“ = ich gehe in mein Bett. – "
        "3. Vertiefung, in die beim „Klickern“ gespielt wird. s. möbsele. "
        "Bei Duden: Kaute, Kute: Vertiefung im Boden, Grube, Loch.'\n"
        "Output: '1. flache Grube\n2. Bett\n3. Vertiefung, in die beim „Klickern“ gespielt wird'\n\n"

        "Wort: uffsteije\n"
        "Text: 'uffgestieche, u betont, ch ist ich-Laut, gewöhnlich für: "
        "aufstehen, vom Bett erheben – allgemein für: anspruchsvoll auftreten, mehr scheinen wollen, als man ist. "
        "„die steiht uff wie e Gräfin“ = die gibt sich so, als wäre sie eine Gräfin.'\n"
        "Output: '1. aufstehen, vom Bett erheben\n2. anspruchsvoll auftreten, mehr scheinen wollen, als man ist'\n\n"

        "Wort: baffsatt\n"
        "Text: 's. babbsatt.'\n"
        "Output: '[SIEHE] babbsatt'\n\n"

        "Wort: Arschgrün\n"
        "Text: 'Arschgrün s. Aaschgrie.'\n"
        "Output: '[SIEHE] Aaschgrie'\n\n"

        "Wort: {Word}\n"
        "Text: '{Definition}'\n"
        "Output:"
    )

    df = pd.read_csv("data/final_wb.txt", sep="\t", header=None, names=["Word", "Definition"], encoding="utf-8")
    df = df.iloc[1:]
    df["Definition"] = df["Definition"].apply(lambda x: x[:1000])
    df["raw_prompts"] = df.apply(lambda row: TEMPLATE.format(**row), axis=1)
    return df, SYSTEM_PROMPT



def load_data_def_generation_fewshot(language, shots, fewshot_mode):
    def few_shot_random(row, df_train, n=5):
        # Randomly sample n examples from the training set
        few_shot_examples = df_train.sample(n)
        few_shot_text = "\n".join(
            [f"Wort: {ex['Word']}\nDefinition: {ex['Definition']}" for _, ex in few_shot_examples.iterrows()]
        )
        # Construct the full few-shot prompt
        FEW_SHOT_PROMPT = f"Beispiele:\n{few_shot_text}\nWort: {row['Word']}\nDefinition:"
        return FEW_SHOT_PROMPT
            
    def few_shot_lowest_edit_distance(row, df_train, n=5):
        # Compute Levenshtein distance between the target word and all training words
        df_train["distance"] = df_train["Word"].apply(lambda w: Levenshtein.distance(row["Word"], w))
        # Select n examples with smallest edit distance
        few_shot_examples = df_train.nsmallest(n, "distance")
        # Build few-shot examples text
        few_shot_text = "\n".join(
            [f"Wort: {ex['Word']}\nDefinition: {ex['Definition']}" for _, ex in few_shot_examples.iterrows()]
        )
        # Construct the few-shot prompt
        FEW_SHOT_PROMPT = f"Beispiele:\n{few_shot_text}\nWort: {row['Word']}\nDefinition:"
        return FEW_SHOT_PROMPT

    def few_shot_ngram(row, df_train, n=5, ngram_size=3):
        def ngram_similarity(a, b, n=3):
            a_grams = {a[i:i+n] for i in range(len(a)-n+1)}
            b_grams = {b[i:i+n] for i in range(len(b)-n+1)}
            return jaccard(a_grams, b_grams)
        word = row["Word"].lower()
        df_train = df_train.assign(
            ngram_sim=df_train["Word"].apply(lambda w: ngram_similarity(word, w.lower(), n=ngram_size))
        )
        few_shot_examples = df_train.nlargest(n, "ngram_sim")
        few_shot_text = "\n".join(
            [f"Wort: {ex['Word']}\nDefinition: {ex['Definition']}" for _, ex in few_shot_examples.iterrows()]
        )
        return f"Beispiele:\n{few_shot_text}\nWort: {row['Word']}\nDefinition:"

    SYSTEM_PROMPT = (
        "Du bist ein präziser und zuverlässiger linguistischer Assistent. "
        "Deine Aufgabe ist es, Wörterbuchdefinitionen zu erstellen. "
        "Gib ausschließlich eine einzige, kurze und prägnante Bedeutung des angefragten Wortes an. "
        "Nutze dabei die im Mainzer Dialekt gebräuchliche Bedeutung des Wortes, "
        "formuliere die Definition jedoch auf Hochdeutsch."
    )
    TEMPLATE = (
        "Erstelle genau eine kurze Wörterbuchdefinition für das Wort '{Word}'. "
        "Verwende dabei die im Mainzer Dialekt gebräuchliche Bedeutung, "
        "ohne Zusatzinformationen, Beispiele oder alternative Bedeutungen anzugeben.\n\n{raw_prompts_fewshot}"
    )

    # Load the few-shot examples
    df_train = pd.read_csv("output/extractor/train.csv")
    df_train["Definition"] = df_train["definition_final"].str.split("\n").str[0].str.replace("1. ", "")

    # Read DataFrame
    df = pd.concat([pd.read_csv("output/extractor/dev.csv"), pd.read_csv("output/extractor/test.csv")])
    df.rename(columns={"definition_final": "Definition"}, inplace=True)
    if fewshot_mode == "random":
        df["raw_prompts_fewshot"] = df.apply(lambda row: few_shot_random(row, df_train, n=shots), axis=1)
    elif fewshot_mode == "editdistance":
        df["raw_prompts_fewshot"] = df.apply(lambda row: few_shot_lowest_edit_distance(row, df_train, n=shots), axis=1)
    df["raw_prompts"] = df.apply(lambda row: TEMPLATE.format(**row), axis=1)
    print(df["raw_prompts"].iloc[0])
    return df, SYSTEM_PROMPT


def load_data_reverse_generation_def_fewshot(language, shots, fewshot_mode):
    def few_shot_random(row, df_train, n=5):
        # Randomly sample n examples from the training set
        few_shot_examples = df_train.sample(n)
        few_shot_text = "\n".join(
            [f"Definition: {ex['Definition']}\nWort: {ex['Word']}" for _, ex in few_shot_examples.iterrows()]
        )
        # Construct the full few-shot prompt
        FEW_SHOT_PROMPT = f"Beispiele:\n{few_shot_text}\nDefinition: {row['definition_final_generation']}\nWord:"
        return FEW_SHOT_PROMPT
            
    def few_shot_lowest_edit_distance(row, df_train, n=5):
        # Compute Levenshtein distance between the target word and all training words
        df_train["distance"] = df_train["Word"].apply(lambda w: Levenshtein.distance(row["Word"], w))
        # Select n examples with smallest edit distance
        few_shot_examples = df_train.nsmallest(n, "distance")
        # Build few-shot examples text
        few_shot_text = "\n".join(
            [f"Definition: {ex['Definition']}\nWort: {ex['Word']}" for _, ex in few_shot_examples.iterrows()]
        )
        # Construct the few-shot prompt
        FEW_SHOT_PROMPT = f"Beispiele:\n{few_shot_text}\nDefinition: {row['definition_final_generation']}\nWord:"
        return FEW_SHOT_PROMPT


    SYSTEM_PROMPT = (
        "Du bist ein präziser und zuverlässiger linguistischer Assistent. "
        "Deine Aufgabe ist es, zu einem gegebenen Bedeutungsinhalt das passende Wort "
        "im Mainzer Dialekt zu finden. "
        "Gib ausschließlich ein einziges Wort aus, das diese Bedeutung im Mainzer Dialekt ausdrückt. "
        "Gib keine Erklärungen, Übersetzungen oder Zusatzinformationen."
    )
    TEMPLATE = (
        "Finde das Mainzer Dialektwort, das die folgende Bedeutung ausdrückt:\n\n"
        "'{definition_final_generation}'\n\n"
        "Antworte nur mit dem Dialektwort.\n\n{raw_prompts_fewshot}"
    )
    df = pd.read_csv("output/extractor/gpt-oss-120b.csv")
    df["definition_final_generation"] = df.apply(lambda row: replace_word(row), axis=1)
    df["definition_final_generation"] = df["definition_final_generation"].apply(lambda x: x[:500])

    df_train = pd.read_csv("output/extractor/train.csv")
    df_train["Definition"] = df_train["definition_final"]
    if fewshot_mode == "random":
        df["raw_prompts_fewshot"] = df.apply(lambda row: few_shot_random(row, df_train, n=shots), axis=1)
    elif fewshot_mode == "editdistance":
        df["raw_prompts_fewshot"] = df.apply(lambda row: few_shot_lowest_edit_distance(row, df_train, n=shots), axis=1)

    df["raw_prompts"] = df.apply(lambda row: TEMPLATE.format(**row), axis=1)
    print(df["raw_prompts"].iloc[0])
    print(df["raw_prompts"].iloc[1])
    return df, SYSTEM_PROMPT
