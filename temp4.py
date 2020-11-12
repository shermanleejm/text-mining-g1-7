import matplotlib.pyplot as plt
import json

topic_name_id = {
    0: "Medical",
    1: "Airlines",
    2: "Schools",
    3: "Farming",
    4: "Medical",
    5: "Hollywood",
    6: "Finance",
    7: "India",
    8: "Businesses",
    9: "Football",
    10: "Europe",
    11: "US Politics",
    12: "Finance",
    13: "Automotive",
}

root_colors = [
    "green",
    "red",
    "purple",
    "brown",
    "pink",
    "grey",
    "darkorange",
    "blue",
    "plum",
    "crimson",
    "sandybrown",
    "lightseagreen",
    "goldenrod",
    "magenta",
]

topic_color = {}
for k, v in topic_name_id.items():
    topic_color[v] = root_colors[k]

with open("results/news_agency_topic_distribution.json") as fp:
    json_data = json.load(fp)
    for publication, topic_distribution in json_data.items():
        # Data to plot
        labels = topic_distribution.keys()
        sizes = topic_distribution.values()
        colors = [topic_color[label] for label in labels]

        # Plot
        plt.pie(sizes, labels=labels, colors=colors)
        plt.axis("on")
        plt.savefig(f"results/news_agency_topic_distribution/{publication}.png")
        plt.close()

