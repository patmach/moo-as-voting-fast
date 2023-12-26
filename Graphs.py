import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import main 
import os

variant_code_to_czech = {
    "binomial_diversity" : "binomická diverzita"
    ,"intra_list_diversity": "intra-list diverzita"
    ,"maximal_diversity":"diverzita na základě maximální podobnosti"
    ,"popularity_complement":"očekáváná inverzní popularita"
    ,"maximal_distance_based_novelty":"maximální distance-based novelty"
    ,"intra_list_distance_based_novelty":"intra-list distance-based novelty"
    ,"only_positive":"EASE_POS"
     , "also_negative":"EASE_NEG"
    ,"num_of_ratings":"Popularita dle známosti"
    ,"avg_ratings":"Popularita na základě hodnocení"
    ,"rank":"Pozice v seznamu doporučení"
    ,"importance":"Váha kritéria"
    ,"user_profile_count":"Počet ohodnocených položek uživatelem"
}

df = pd.read_csv("values_21_12_old.csv")
metrics = ["relevance", "diverzita", "novelty", "popularita","kalibrace"]
matplotlib.rcParams['figure.figsize'] = 15, 12
def print_line_graphs(df, x_axis, x_axis_postfix = False, to_round = 50, folder="OldNormalizationGraphs2"):
    df["user_profile_count"] = round(df["user_profile_count"]/to_round)*to_round
    for i in [4]:
        x_axis_column = x_axis
        if x_axis_postfix:
            x_axis_column = f"{x_axis}_{i}"
        g = sns.relplot(data=df,\
                    x=x_axis_column, y=f"support_{i}", kind="line")
        g.set_axis_labels(variant_code_to_czech[x_axis], metrics[i])
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
        g.set_axis_labels(variant_code_to_czech[x_axis], metrics[1])
        g.set(title=variant_code_to_czech[diversity_type])
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
        g.set_axis_labels(variant_code_to_czech[x_axis], metrics[2])
        g.set(title=variant_code_to_czech[novelty_type])
        plt.tight_layout()
        plt.savefig(os.path.join(folder,f"line_{novelty_type}_{x_axis}.png"), bbox_inches='tight')
    for popularity_type in ["num_of_ratings","avg_ratings"]:
        x_axis_column = x_axis
        if x_axis_postfix:
            x_axis_column = f"{x_axis}_3"
        g = sns.relplot(data=df[df["popularity_type"]==popularity_type],\
                    x=x_axis_column, y=f"support_3", kind="line")
        g.set(ylim=(0, 100))
        g.set_axis_labels(variant_code_to_czech[x_axis], metrics[3])
        g.set(title=variant_code_to_czech[popularity_type])
        plt.tight_layout()
        plt.savefig(os.path.join(folder,f"line_{popularity_type}_{x_axis}.png"), bbox_inches='tight')
    for relevance_type in ["only_positive", "also_negative"]:
        x_axis_column = x_axis
        if x_axis_postfix:
            x_axis_column = f"{x_axis}_0"
        g = sns.relplot(data=df[df["relevance_type"]==relevance_type],\
                    x=x_axis_column, y=f"support_0", kind="line")
        g.set(ylim=(0, 100))
        g.set_axis_labels(variant_code_to_czech[x_axis], metrics[0])
        g.set(title=variant_code_to_czech[relevance_type])
        plt.tight_layout()
        plt.savefig(os.path.join(folder,f"line_{relevance_type}_{x_axis}.png"), bbox_inches='tight')


def correlation(df, folder="OldNormalizationGraphs2"):
    for diversity_type in ["intra_list_diversity","maximal_diversity","binomial_diversity"]:
        for novelty_type in ["popularity_complement","maximal_distance_based_novelty","intra_list_distance_based_novelty"]:
            for popularity_type in ["num_of_ratings","avg_ratings"]:
                    for relevance_type in ["only_positive", "also_negative"]:
                        part_df = df[(df["diversity_type"]==diversity_type) & (df["novelty_type"]==novelty_type)\
                                     &(df["popularity_type"]==popularity_type) & (df["relevance_type"]==relevance_type)]
                        part_df = part_df[[f"support_{i}" for i in range(5)] + ["rank","user_profile_count"]]
                        corr = part_df.corr()
                        g = None
                        g = sns.heatmap(corr,  cmap='coolwarm',
                                    xticklabels=metrics + ["rank", "number of rated"],
                                    yticklabels=metrics + ["rank", "number of rated"])
                        
                        g.set(title=f"Korelace\nrelevance: {relevance_type}\ndiverzita : {diversity_type}\nnovelty: {novelty_type}\npopularita : {popularity_type}")
                        plt.savefig(os.path.join(folder,f"corr_{relevance_type}_{diversity_type}_{novelty_type}_{popularity_type}.png"), bbox_inches='tight')
                        plt.clf()
            
if __name__ == "__main__":
    correlation(df)
    print_line_graphs(df, "importance", True, 1)
    print_line_graphs(df, "rank", False, 1)
    print_line_graphs(df, "user_profile_count", False, 25)


    print("DONE")


