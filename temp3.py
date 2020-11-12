import pandas as pd 
import re
import json
from nltk.tag import pos_tag
from nltk.chunk import conlltags2tree, tree2conlltags
import nltk
import math
stopwords = nltk.corpus.stopwords.words('english')

sample = pd.read_csv("covid19_articles_20200914.csv", index_col=[0])

import spacy 
from spacy import displacy 
## python -m spacy download en_core_web_sm 
## theres md and lg 
## en_core_web_sm is a CNN for token vectors, POS tags
## pretrained statistical models for English
import en_core_web_sm
nlp = en_core_web_sm.load()

## only concerned with the named entities that involve human names
SIGNIGICANT_ENTITY_LABELS = [385, 381, 380]

ne_count = {}
for line in sample.iterrows(): 
    content = re.sub("[^a-zA-Z\&]+", " ",line[1]['content'])
    content = nlp(content)
    for ent in content.ents:
        if ent.label in SIGNIGICANT_ENTITY_LABELS:
            ent_text = ent.text.lower()
            if ne_count.get(ent_text, False) == False:
                ne_count[ent_text] = 0
            ne_count[ent_text] += 1

with open("named_entities.json", "w+") as fp:
    json.dump(ne_count, fp, indent=2)