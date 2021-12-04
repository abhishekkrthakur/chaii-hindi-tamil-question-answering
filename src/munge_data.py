import json
import pandas as pd
import ast


def prepare_squad_translated():
    hi_df = pd.read_csv("../input/squad_hi.csv")
    ta_df = pd.read_csv("../input/squad_ta.csv")

    data = []
    for _, row in hi_df.iterrows():
        context = row["context"]
        question = row["question"]
        answers = ast.literal_eval(row["answers"])
        for ans in answers:
            data_row = (context, question, ans["text"], ans["answer_start"], "hindi")
            data.append(data_row)

    for _, row in ta_df.iterrows():
        context = row["context"]
        question = row["question"]
        answers = ast.literal_eval(row["answers"])
        for ans in answers:
            data_row = (context, question, ans["text"], ans["answer_start"], "tamil")
            data.append(data_row)

    new_df = pd.DataFrame(data, columns=["context", "question", "answer_text", "answer_start", "language"])
    new_df = new_df.sample(frac=1).reset_index(drop=True)
    new_df = new_df[new_df.answer_start >= 0].reset_index(drop=True)
    new_df.to_csv("../input/squad_translated.csv", index=False)


def prepare_squadv2(filepath):
    data = []
    with open(filepath, encoding="utf-8") as f:
        squad = json.load(f)
        for example in squad["data"]:
            for paragraph in example["paragraphs"]:
                context = paragraph["context"]  # do not strip leading blank spaces GH-2585
                for qa in paragraph["qas"]:
                    question = qa["question"]
                    id_ = qa["id"]

                    answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                    answers = [answer["text"] for answer in qa["answers"]]

                    # Features currently used are "context", "question", and "answers".
                    # Others are extracted here for the ease of future expansions.
                    if len(answer_starts) != 0:
                        data.append(
                            {
                                "id": id_,
                                "context": context,
                                "question": question,
                                "answer_start": answer_starts[0],
                                "answer_text": answers[0],
                                "language": "english",
                                "kfold": -1,
                            }
                        )
    return data


def prepare_tydiqa(filepath):
    data_list = []
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
        for article in data["data"]:
            title = article.get("title", "").strip()
            for paragraph in article["paragraphs"]:
                context = paragraph["context"].strip()
                for qa in paragraph["qas"]:
                    question = qa["question"].strip()
                    id_ = qa["id"]

                    answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                    answers = [answer["text"].strip() for answer in qa["answers"]]

                    # Features currently used are "context", "question", and "answers".
                    # Others are extracted here for the ease of future expansions.
                    data_list.append(
                        {
                            "id": id_,
                            "context": context,
                            "question": question,
                            "answer_text": answers[0],
                            "answer_start": answer_starts[0],
                            "language": id_.split("-")[0],
                            "kfold": -1,
                        }
                    )
    return data_list


if __name__ == "__main__":

    # generate squadv2.csv
    data = prepare_squadv2("../input/train-v2.0.json")
    data += prepare_squadv2("../input/dev-v2.0.json")
    data = pd.DataFrame.from_records(data)
    data.to_csv("../input/squadv2.csv", index=False)

    # generate tydiqa.csv
    data = prepare_tydiqa("../input/tydiqa-goldp-v1.1-train.json")
    data += prepare_tydiqa("../input/tydiqa-goldp-v1.1-dev.json")
    data = pd.DataFrame.from_records(data)
    data.to_csv("../input/tydiqa.csv", index=False)

    # generate squad_translated.csv
    prepare_squad_translated()
