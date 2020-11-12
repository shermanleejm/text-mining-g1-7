import matplotlib.pyplot as plt
import json

with open("results/news_agency_topic_distribution.json") as fp:
    json_data = json.load(fp)
    for publication, topic_distribution in json_data.items():
        # Data to plot
        labels = topic_distribution.keys()
        sizes = topic_distribution.values()
        colors = [
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
        ]

        # Plot
        plt.pie(sizes, labels=labels)
        plt.axis("on")
        plt.savefig(f"results/news_agency_topic_distribution/{publication}.png")
        plt.close()

