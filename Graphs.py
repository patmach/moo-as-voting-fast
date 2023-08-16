import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import main 
import os
df = pd.read_csv("values.csv")
metrics = ["relevance", "diversity", "novelty", "popularity","calibration"]
matplotlib.rcParams['figure.figsize'] = 15, 12
def print_line_graphs(df, x_axis, x_axis_postfix = False, to_round = 50, folder="NormalizationGraphs"):
    df["user_profile_count"] = round(df["user_profile_count"]/to_round)*to_round
    for i in [0,3,4]:
        x_axis_column = x_axis
        if x_axis_postfix:
            x_axis_column = f"{x_axis}_{i}"
        g = sns.relplot(data=df,\
                    x=x_axis_column, y=f"support_{i}", kind="line")
        g.set_axis_labels(x_axis, metrics[i])
        g.set(title=metrics[i])
        g.set(ylim=(0, 100))
        plt.tight_layout()
        plt.savefig(os.path.join(folder,f"line_{metrics[i]}_{x_axis}.png"), bbox_inches='tight')
    for diversity_type in ["intra_list_diversity","maximal_diversity","binomial_diversity"]:
        x_axis_column = x_axis
        if x_axis_postfix:
            x_axis_column = f"{x_axis}_1"
        g = sns.relplot(data=df[df["diversity_type"]==diversity_type],\
                x=x_axis_column, y=f"support_1", kind="line")
        g.set_axis_labels(x_axis, metrics[1])
        g.set(title=f"diversity : {diversity_type}")
        g.set(ylim=(0, 100))
        plt.tight_layout()
        plt.savefig(os.path.join(folder,f"line_{diversity_type}_{x_axis}.png"), bbox_inches='tight')
    for novelty_type in ["popularity_complement","maximal_distance_based_novelty","intra_list_distance_based_novelty"]:
        x_axis_column = x_axis
        if x_axis_postfix:
            x_axis_column = f"{x_axis}_2"
        g = sns.relplot(data=df[df["novelty_type"]==novelty_type],\
                    x=x_axis_column, y=f"support_2", kind="line")
        g.set(ylim=(0, 100))
        g.set_axis_labels(x_axis, metrics[2])
        g.set(title=f"novelty: {novelty_type}")
        plt.tight_layout()
        plt.savefig(os.path.join(folder,f"line_{novelty_type}_{x_axis}.png"), bbox_inches='tight')


def correlation(df, folder="NormalizationGraphs"):
    for diversity_type in ["intra_list_diversity","maximal_diversity","binomial_diversity"]:
        for novelty_type in ["popularity_complement","maximal_distance_based_novelty","intra_list_distance_based_novelty"]:
            part_df = df[(df["diversity_type"]==diversity_type) & (df["novelty_type"]==novelty_type)]
            part_df = part_df[[f"support_{i}" for i in range(5)] + ["rank","user_profile_count"]]
            corr = part_df.corr()
            g = None
            g = sns.heatmap(corr,  cmap='coolwarm',
                        xticklabels=metrics + ["rank", "number of rated"],
                        yticklabels=metrics + ["rank", "number of rated"])
            
            g.set(title=f"Correlation\ndiversity : {diversity_type}\nnovelty: {novelty_type}")
            plt.savefig(os.path.join(folder,f"corr_{diversity_type}_{novelty_type}.png"), bbox_inches='tight')
            plt.clf()
            


print_line_graphs(df, "importance", True, 1)
print_line_graphs(df, "rank", False, 1)
print_line_graphs(df, "user_profile_count", False, 25)
correlation(df)

print("DONE")


