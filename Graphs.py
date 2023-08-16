import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import main 
df = pd.read_csv("values.csv")
metrics = ["relevance", "diversity", "novelty", "popularity","calibration","rank"]
df["user_profile_count"] = round(df["user_profile_count"]/50)*50
for i in [0,3,4]:
    g = sns.relplot(data=df,\
                x=f"user_profile_count", y=f"support_{i}", kind="line")
    g.set_axis_labels("user_profile_count", metrics[i])
    g.set(title=metrics[i])
    plt.show()
for diversity_type in ["intra_list_diversity","maximal_diversity","binomial_diversity"]:
    g = sns.relplot(data=df[df["diversity_type"]==diversity_type],\
            x=f"user_profile_count", y=f"support_1", kind="line")
    g.set_axis_labels("user_profile_count", metrics[1])
    g.set(title=f"{metrics[i]}\ndiversity : {diversity_type}")
    plt.show()
for novelty_type in ["popularity_complement","maximal_distance_based_novelty","intra_list_distance_based_novelty"]:
    g = sns.relplot(data=df[df["novelty_type"]==novelty_type],\
                x=f"user_profile_count", y=f"support_2", kind="line")
    g.set_axis_labels("user_profile_count", metrics[2])
    g.set(title=f"{metrics[i]}\ndiversity : {diversity_type}, novelty: {novelty_type}")
    plt.show()


def correlation(df):
    for diversity_type in ["intra_list_diversity","maximal_diversity","binomial_diversity"]:
        for novelty_type in ["popularity_complement","maximal_distance_based_novelty","intra_list_distance_based_novelty"]:
            part_df = df[(df["diversity_type"]==diversity_type) & (df["novelty_type"]==novelty_type)]
            part_df = part_df[[f"support_{i}" for i in range(5)] + ["rank"]]
            corr = part_df.corr()
            g = sns.heatmap(corr,  cmap='coolwarm',
                        xticklabels=metrics + ["rank"],
                        yticklabels=metrics + ["rank"])
            g.set(title=f"diversity : {diversity_type}, novelty: {novelty_type}")
            plt.show()

