from pathlib import Path
import pandas as pd

data_file = Path("./data-consolidated.json")
data_json = pd.read_json(data_file)

drop_columns = ["context", "document_id", "lang"]
data_json = data_json.drop(columns=drop_columns, axis=1)

data_json.to_csv("./data-consolidated.csv", index=False, header=["question", "language_code", "answer"], sep=",")
print("Done.")
