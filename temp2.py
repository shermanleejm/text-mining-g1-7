import pandas as pd
import networkx as nx
import networkx.algorithms.community as nx_comm
from networkx.algorithms.centrality import betweenness_centrality
import json
import csv
import nltk
import gensim
import re
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import spacy
from nltk.stem.snowball import SnowballStemmer
from datetime import datetime

stemmer = SnowballStemmer(language="english")

## for more comprehensive use the large one
## python -m spacy download en_core_web_lg
sp = spacy.load('en_core_web_sm')
spacy_stopwords = sp.Defaults.stop_words

SAVED_STATE_FOLDER = "/Users/shrmnl/Github/text-mining-g1-7/saved_states/"
SAVED_FIGURES = "/Users/shrmnl/Github/text-mining-g1-7/plt_figures/"

df = pd.read_csv("./sample.csv")

def get_bias(text, index, visualise=False, filename="graph.png", folder=False):
    G = nx.Graph()
    ## convert lowercase
    # first_article = df.head().loc[row_num]["content"].lower()
    first_article = text.lower()
    ## remove html tags from improper scraping
    first_article = re.sub("\<(.)+\>", " ", first_article)
    ## replace non alpha-numeric + . + & + \s with \s
    ## kept the apostraphe ’ chr(8217)
    special_apostraphe = chr(8217) 
    first_article = re.sub(f"[^0-9a-zA-Z\.\&\s{special_apostraphe}]+", " ", first_article)
    ## remove random new lines
    first_article = re.sub(f"\n", " ", first_article)

    ## loop through the sentences
    for sentence in first_article.split(". "):
        ## ignore empty strings
        if sentence.strip() != "":
            cleaned_sentence = [stemmer.stem(word) for word in sentence.split(" ") if word not in spacy_stopwords and word.strip() != ""]
            for i in range(len(cleaned_sentence) - 1):
                word = cleaned_sentence[i]
                for other_word in cleaned_sentence[i + 1:]:
                    if G.get_edge_data(word, other_word) == None:
                        G.add_edge(word, other_word, weight=0)
                    old_weight = G.get_edge_data(word, other_word)["weight"]
                    new_weight = old_weight + 1
                    G.add_edge(word, other_word, weight=new_weight)

    ## remove edges that have very small weights
    # to_remove = [(a,b) for a, b in G.edges if G[a][b]["weight"] == 1]
    # G.remove_edges_from(to_remove)
    # G.remove_nodes_from(list(nx.isolates(G)))

    ## remove nodes that are in isolated pairs and triplets
    for island in list(nx.connected_components(G)):
        if len(island) < 11:
            for node in island:
                G.remove_node(node)

    ## get list of betweenness_centrality scores to find most influential words
    top_words_with_scores = {k: v for k, v in sorted(betweenness_centrality(G).items(), key=lambda item: item[1], reverse=True)[:5] if v > 0} 
    # print (f"The top 5 key words are ", end="")
    # for word in top_5_words_with_scores.keys():
    #     print (word, end=", ")
    # print ()

    ## determining modularity
    try:
        modularity_score = nx_comm.modularity(G, nx_comm.label_propagation_communities(G))
    except:
        modularity_score = 0
    # print(f"The modularity score is {modularity_score}")


    if folder:
        filename = folder + str(index) + " - " + str(modularity_score) + ".png"

    ## for visualisation
    plt.figure(figsize=(100,100))
    pos = nx.spring_layout(G)
    pos_higher = {}
    y_off = 10 ## offset value
    for k, v in pos.items():
        pos_higher[k] = (v[0], v[1]+y_off)
    nx.draw(G, pos_higher, with_labels=True, node_size=60)
    plt.axis("off")

    if folder:
        plt.savefig(f"{SAVED_FIGURES}{filename}")
        
    if visualise:
        plt.show()

    plt.close()

    return modularity_score, list(top_words_with_scores.keys())

for i, row in df.iterrows():
    modularity_score, top_words = get_bias(text=row["content"], index=i)
    df.loc[i, "bias"] = modularity_score
    try:
        top_words_string = top_words[0]
        for word in top_words[1:]:
            top_words_string += "," + word
    except:
        top_words_string = ""
    df.loc[i, "top_words"] = top_words_string

for domain in df.domain.unique().tolist():
    NEWS_AGENCY = domain

    cnn_df = df[df["domain"] == NEWS_AGENCY]

    for i in range(0, 100, 10):
        i = i / 100
        data_range = cnn_df[(cnn_df["bias"] > i) & (cnn_df["bias"] < (i + 0.1))]
        for i, row in data_range.sample(5).iterrows():
            try:
                modularity_score, top_words = get_bias(text=row["content"], folder=f"{NEWS_AGENCY}_figures/", index=i)
            except:
                pass