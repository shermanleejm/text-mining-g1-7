import pandas as pd 
import json
df = pd.read_csv("covid19_articles_20200914.csv")
with open("named_entities.json") as fp:
    ne_json = json.load(fp)

result = {}
## 380 381 385
ne_json_cleaned = ne_json["380"] + ne_json["385"]
ne_json_cleaned = set(ne_json_cleaned)

for i, row in df.iterrows():
    for ne in ne_json_cleaned:
        if ne in row["content"].lower():
            if result.get(ne, False) == False:
                result[ne] = 0
            result[ne] += 1

with open("results/ne_occurence_py.json", "w+") as fp:
    json.dump(result, fp, indent=2)