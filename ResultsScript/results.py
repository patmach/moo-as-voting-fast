from datetime import datetime, timedelta
import pypyodbc as odbc
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
from textwrap import wrap
import textwrap

date_format= "%d-%m-%Y %H:%M:%S.%f"
number_of_recommendations = 15
folder_with_graphs = "Results"
ActTypeToQuestionSection = {
    "Demographics":[],
    "Information about movies":[],
    "Explanation":["Preview Explanation", "Score Explanation", "Metrics Explanation"],
    "Relevance":["Relevance"],
    "Diversity":["Diversity"],
    "Novelty":["Novelty"],
    "Popularity":["Popularity"],
    "Calibration":[],
    "Objectives overall":["Diversity", "Novelty", "Popularity", "Relevance"],
    "Types of objectives filter":["Tweak Mechanism"],
    "Overall":["Diversity", "Novelty", "Popularity", "Relevance"],
    "Additional":[]
}

QuestionSectionsDependency = {
    "Demographics":[],
    "Information about movies":[],
    "Explanation":[],
    "Relevance":[],
    "Diversity":[],
    "Novelty":[],
    "Popularity":[],
    "Calibration":[],
    "Objectives overall":["Diversity", "Novelty", "Popularity", "Relevance"],
    "Types of objectives filter":[],
    "Overall":["Diversity", "Novelty", "Popularity", "Relevance", "Objectives overall"],
    "Additional":[]
}
LikertScale = {"Strongly agree" : 1, 
               "Agree" : 0.5,
               "Neutral / Don't Know" : 0,
               "Disagree" : -0.5,
               "Strongly disagree" : -1}


def wrap_labels(ax, width_x, break_long_words=True, width_y = 60):
    """
    wraps labels by newlines
    
    Parameters
    ----------
    ax : _type_
        graph or its axis
    width : int
        maximum width for one line of text
    break_long_words : bool, optional
        If true breaks long words, by default False
    """
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text().replace('_',' ')
        labels.append(textwrap.fill(text, width=width_x,
                      break_long_words=break_long_words))
    ax.set_xticks(ax.get_xticks().tolist())
    ax.set_xticklabels(labels, rotation=0)
    labels = []
    for label in ax.get_yticklabels():
        text = label.get_text().replace('_',' ')
        labels.append(textwrap.fill(text, width=width_y,
                      break_long_words=break_long_words))
    ax.set_yticks(ax.get_yticks().tolist())
    ax.set_yticklabels(labels, rotation=0)


def get_connection_string():
    """
        Connects to the database
    """
    DriverName = "SQL Server"
    ServerName =  "np:\\\\.\\pipe\LOCALDB#6E0416EF\\tsql\\query"
    ServerName = "localhost,1401"
    DatabaseName = "aspnet-53bc9b9d-9d6a-45d4-8429-2a2761773502"
    Username = 'RS'
    file = open('pswd.txt',mode='r')    
    Password = file.read()
    file.close()
    connectionstring=f"""DRIVER={{{DriverName}}};
        SERVER={ServerName};
        DATABASE={DatabaseName};
        UID={Username};
        PWD={Password};
        TrustServerCertificate=yes;
    """
    return connectionstring

def get_table(table_name):
    """

    Parameters
    ----------
    table_name : str
        name of the table in database

    Returns
    -------
    pd.DataFrame
        Content of the database table
    """
    conn = odbc.connect(get_connection_string())
    df = pd.read_sql_query('SELECT  * FROM ' + table_name, conn)
    df.columns = df.columns.str.lower()
    conn.close()
    return df

def not_null_lists(f,s):
    """

    Parameters
    ----------
    s : list
        first list
    d : list
        second list

    Returns
    -------
    list
        list where value of each index corresponds to value in s with the same index. 
        If the value in s with the same index is null the the value in d with the same index is inserted.
    """
    list = []
    for i in range(len(f)):
        list.append(f[i] if not pd.isna(f[i]) else s[i])
    return list

def get_userAnswers():
    """

    Returns
    -------
    pd.DataFrame
        DataFrame with all needed information about users answers
    """
    userAnswers = get_table("UserAnswers")
    questions = get_table("Questions").rename(columns={"id": "questionid", "text":"questiontext"})
    answers = get_table("Answers").rename(columns={"id": "answerid", "text":"answertext"}).drop(columns=["questionid"])
    questionSections = get_table("QuestionSections").rename(columns={"id": "questionsectionid", "name": "sectionname"})
    questions = pd.merge_ordered(questions, questionSections, how="left", on="questionsectionid")
    userAnswers = pd.merge_ordered(userAnswers, questions, how="inner", on="questionid")
    userAnswers = pd.merge_ordered(userAnswers, answers, how="left", on="answerid")
    LikertScaleTexts = list(LikertScale.keys())
    userAnswers["valueanswer"] = [LikertScaleTexts[int(x)] if not np.isnan(x) else x for x in list(userAnswers["value"])]
    userAnswers["answer"] = not_null_lists(list(userAnswers["valueanswer"]), list(userAnswers["answertext"]))
    userAnswers["answerindex"] = not_null_lists(list(userAnswers["answerid"]), list(userAnswers["value"]))
    userAnswers.sort_values(by="answerindex", inplace=True)
    userAnswers = userAnswers.drop(columns=["value","text","answerid","valueanswer"])
    return userAnswers

def get_first_user_acts():
    """

    Returns
    -------
    pd.DataFrame
        DataFrame with all users first acts from each group of acts
    """
    acts = get_table("Acts").rename(columns={"id": "actid", "code":"actcode"})
    colnames=["userid","actid","date"] 
    userActs = pd.read_csv("Logs/UserActs.txt", sep=';', names=colnames)
    userActs["index"] = userActs.index
    userActs = pd.merge_ordered(userActs, acts, how="inner",on="actid")
    userActs.sort_values(by="index", inplace=True)
    userActs = userActs.drop_duplicates(subset=["userid","typeofact"])    

    return userActs

def get_most_used_in_recommender_queries():
    """

    Returns
    -------
    pd.DataFrame
        DataFrame with all users first acts from each group of acts
    """
    recommenderQueries = get_recommender_queries_from_file()
    recommenderQueries["tweak mechanism"] = ["PlusMinusButtons" if x == "Buttons" else "DragAndDrop" if x=="Drag and drop"
                                             else x for x in list(recommenderQueries["tweak mechanism"])]
    grouped_dfs = []
    acts = get_table("Acts").rename(columns={"id": "actid", "code":"actcode"})
    for column in ["relevance type","diversity type","novelty type","popularity type", "tweak mechanism"]:
        df = recommenderQueries.groupby(['userid',column])['date'].count()
        df = df.reset_index()
        df = df.rename(columns={"date": "count", column:"actcode"})
        df= pd.merge(df, acts[["actcode","typeofact"]], on="actcode")
        grouped_dfs.append(df)            
    recommenderQueries = pd.concat(grouped_dfs)
    recommenderQueries.sort_values("count", ascending=False, inplace=True)
    recommenderQueries = recommenderQueries.drop_duplicates(subset=["userid","typeofact"])
    
    return recommenderQueries

def get_recommender_queries_from_file():
    colnames=["relevance type","diversity type","novelty type","popularity type","calibration type",
              "relevance","diversity","novelty","popularity","calibration","tweak mechanism", "userid", "date"] 
    return pd.read_csv("Logs/RecommenderQueries.txt", sep=';', names=colnames)

def get_recommender_queries_by_metric():
    """
    
    Returns
    -------
    pd.DataFrame
        All queries to the recommender called by user
    """
    recommenderqueries = get_recommender_queries_from_file()
    bymetrics = []
    for metric in ["relevance","diversity","novelty","popularity","calibration"]:
        bymetric = pd.DataFrame()
        bymetric["Metric"] = [metric] * len(recommenderqueries)
        bymetric["Metric importance"] = recommenderqueries[metric]
        bymetric["Metric variant"] = recommenderqueries[metric+" type"]
        bymetric["Metric variant"] = [s if pd.isna(s) else s.replace("_"," ")
                                      for s in list(bymetric["Metric variant"])]
        bymetric["Tweak mechanism"] = recommenderqueries["tweak mechanism"]
        bymetric["userid"] = recommenderqueries["userid"]
        bymetric["Date"] = recommenderqueries["date"]
        bymetrics.append(bymetric)
    bymetrics = pd.concat(bymetrics)
    return bymetrics

def get_recommender_queries_rating_interaction():
    recommenderqueries = get_recommender_queries_from_file()
    recommenderqueries = recommenderqueries[~recommenderqueries["userid"].isin(author_users)]
    colnames=["userid","itemid","ratingscore","date"] 
    ratings = pd.read_csv("Logs/Ratings.txt", sep=';', names=colnames)
    colnames=["userid","itemid","type","date"] 
    interactions = pd.read_csv("Logs/Interactions.txt", sep=';', names=colnames)
    recommenderqueries_new = []
    recommenderqueries["date"] = pd.to_datetime(recommenderqueries["date"], format=date_format)
    ratings["date"] = pd.to_datetime(ratings["date"], format=date_format)
    interactions["date"] = pd.to_datetime(interactions["date"], format=date_format)
    for userid in recommenderqueries["userid"].unique():
        recommenderqueries_new.append(process_user_interactions(userid, ratings, interactions, recommenderqueries))
    recommenderqueries_new = pd.concat(recommenderqueries_new)
    return recommenderqueries_new

def process_user_interactions(userid, ratings, interactions, recommenderqueries):
    """
    Enrich recommender query dataset of users seen, clicks, ratings and positive ratings 
    
    Parameters
    ----------
    userid : int
        ID of user
    ratings : pd.DataFrame
        dataframe with all given ratings
    interactions : pd.DataFrame
        dataframe with all interactions
    recommenderqueries : pd.DataFrame
        dataframe with all recommender queries

    Returns
    -------
    pd.DataFrame
        Enriched recommender query dataset of users seen, clicks, ratings and positive ratings 
    """
    u_ratings = ratings[ratings["userid"]==userid]
    u_interactions = interactions[interactions["userid"]==userid]
    u_seens = u_interactions[u_interactions["type"]=="Seen"]
    u_clicks = u_interactions[u_interactions["type"]=="Click"]
    u_recommenderqueries = recommenderqueries[recommenderqueries["userid"]==userid]
    num_seens = []
    num_clicks = []
    num_positive_ratings = []
    num_ratings = []
    upper_bound_date = recommenderqueries["date"].max() + timedelta(days=1)
    for i in range(len(u_recommenderqueries)):
        min_date = u_recommenderqueries.iloc[i]["date"]
        max_date = upper_bound_date
        if (len(u_recommenderqueries) > i+1):
            max_date = u_recommenderqueries.iloc[i + 1]["date"]
        cur_seens = u_seens[(u_seens["date"] > min_date) & (u_seens["date"] < max_date)].head(number_of_recommendations)
        cur_clicks = u_clicks[(u_clicks["date"] > min_date) & (u_clicks["date"] < max_date)]
        cur_clicks = pd.merge(cur_clicks, cur_seens["itemid"], on=["itemid"]).drop_duplicates(subset=["itemid"])
        cur_ratings = u_ratings[(u_ratings["date"] > min_date) & (u_ratings["date"] < max_date)]
        cur_ratings = pd.merge(cur_ratings, cur_seens["itemid"], on=["itemid"]).drop_duplicates(subset=["itemid"], keep="last")
        cur_positive_ratings = cur_ratings[cur_ratings["ratingscore"] > 5]
        num_seens.append(len(cur_seens))
        num_clicks.append(len(cur_clicks))
        num_ratings.append(len(cur_ratings))
        num_positive_ratings.append(len(cur_positive_ratings))
    u_recommenderqueries["seens"] = num_seens
    u_recommenderqueries["clicks"] = num_clicks
    u_recommenderqueries["positive_ratings"] = num_positive_ratings
    u_recommenderqueries["ratings"] = num_ratings
    u_recommenderqueries["clicks_per_seen"] =  u_recommenderqueries["clicks"] / u_recommenderqueries["seens"] 
    u_recommenderqueries["ratings_per_seen"] =  u_recommenderqueries["ratings"] / u_recommenderqueries["seens"] 
    u_recommenderqueries["positive_ratings_per_seen"] =  u_recommenderqueries["positive_ratings"] / u_recommenderqueries["seens"] 
    u_recommenderqueries["positive_ratings_per_rating"] = u_recommenderqueries["positive_ratings"] / u_recommenderqueries["ratings"]
    u_recommenderqueries["rank"] = list(range(len(u_recommenderqueries)))
    corr = u_recommenderqueries[["relevance","diversity","novelty","popularity","calibration", "rank",\
                                 "seens", "clicks_per_seen", "positive_ratings_per_seen","ratings_per_seen"]] .corr()
    g = sns.heatmap(corr,  cmap='coolwarm')    
    g.set(title=f"Korelace - požadavek na RS")
    plt.savefig(os.path.join(folder_with_graphs,f"corr_recommender_queries.png"), bbox_inches='tight')
    plt.clf()
    plt.cla()
    return u_recommenderqueries

def process_metrics():
    """
    Compute all stats objectives weights in recommender queries and save them as graphs
    """
    global author_users
    recommenderqueries = get_recommender_queries_by_metric()
    recommenderqueries = recommenderqueries[~recommenderqueries["userid"].isin(author_users)]
    cm = sns.color_palette("plasma",len(recommenderqueries["Metric"].unique()))
    g = sns.violinplot(data = recommenderqueries, x= "Metric", y= "Metric importance",
               palette=cm)
    g.set(title = f"Metrics weights specified by user")
    wrap_labels(g, 12)
    plt.savefig(os.path.join(folder_with_graphs,f"Metrics_importances.png"), bbox_inches='tight')
    plt.close('all')
    plt.clf()
    plt.cla()
    g = sns.violinplot(data = recommenderqueries, x= "Metric variant", y= "Metric importance",
               palette=cm)
    g.set(title = f"Metrics variants weights specified by user")
    wrap_labels(g, 6)
    plt.savefig(os.path.join(folder_with_graphs,f"Metrics_variants_importances.png"), bbox_inches='tight')
    plt.close('all')
    plt.clf()
    plt.cla()
    process_metrics_per_variant_and_per_mechanism(recommenderqueries, cm)
    

def process_metrics_per_variant_and_per_mechanism(recommenderqueries, cm):
    """
    Compute all stats objectives weights per used metric variant and used mechanism

    Parameters
    ----------
    recommenderqueries : pd.DataFrame
        All queries to the recommender called by user
    cm : _RGBColorPalette
        Color palette used
    """
    for metric in recommenderqueries["Metric"].unique():
        metric_recommenderqueries = recommenderqueries[recommenderqueries["Metric"] == metric]
        if(len(metric_recommenderqueries["Metric variant"].unique()) > 1):
            g = sns.violinplot(data = metric_recommenderqueries, x= "Metric variant", y= "Metric importance",
               palette=cm)
            g.set(title = f"{metric} weight specified by user per metric variant")
            wrap_labels(g, 12)
            plt.savefig(os.path.join(folder_with_graphs,f"variants_of_{metric}_importances.png"), bbox_inches='tight')
            plt.close('all')
            plt.clf()
            plt.cla()
            g = sns.violinplot(data = metric_recommenderqueries, x= "Tweak mechanism", y= "Metric importance",
               split=True, palette=cm)
            g.set(title = f"{metric} weight specified by user per tweak mechanism")
            wrap_labels(g, 12)
            plt.savefig(os.path.join(folder_with_graphs,f"by_tweak_mechanism_{metric}_importances.png"), bbox_inches='tight')
            plt.close('all')
            plt.clf()
            plt.cla() 

discarded_users = []
author_users = []
users_without_questionnaire = []

def set_discarded_users():
    """
    Discards users of authors and users that haven't answered attention checks right
    """
    global discarded_users, author_users, users_without_questionnaire
    users = get_table("Users")
    author_users.extend(list(users[(users["username"]=="log_master2") | (users["username"]=="lp")]["id"]))
    userAnswers = get_userAnswers()
    attentionChecks = userAnswers[userAnswers["questiontext"].str.contains("attention check", case=False)]
    attentionChecks["expected_answer"] = [s.split('"')[1] for s in attentionChecks["questiontext"]]
    userWrongAnswersToAttentionCheck = attentionChecks[attentionChecks["expected_answer"].str.lower() \
                                                       != attentionChecks["answer"].str.lower()]
    discarded_users.extend(list(userWrongAnswersToAttentionCheck["userid"].unique()))
    discarded_users = list(set(discarded_users))
    users_without_questionnaire = list(set(users[~users["id"].isin(userAnswers["userid"])]["id"]))

def process_questions():
    """
    Compute all results from user answers and save them as graphs
    """
    global discarded_users, author_users
    userAnswers = get_userAnswers()
    firstUserActs = get_first_user_acts()
    recommendedQueries = get_most_used_in_recommender_queries()
    best_peformances_by_ratings = process_interactions_and_ratings()
    questions = get_table("Questions")
    userAnswers = userAnswers[~userAnswers["userid"].isin(discarded_users)]
    userAnswers = userAnswers[~userAnswers["userid"].isin(author_users)]
    firstUserActs = firstUserActs[~firstUserActs["userid"].isin(author_users)]
    recommendedQueries = recommendedQueries[~recommendedQueries["userid"].isin(author_users)]
    likertScaleQuestionsMeanAndStd = []
    likertScaleAnswers = []
    for questionid in questions["id"]:        
        q_userAnswers = userAnswers[userAnswers["questionid"] == questionid]   
        first = q_userAnswers.iloc[0]
        if (first["answertype"] == 0):
            mean_std_df, q_userAnswers = get_Likert_Scale_int_value_mean_and_std_dfs(q_userAnswers)
            likertScaleQuestionsMeanAndStd.append(mean_std_df)
            likertScaleAnswers.append(q_userAnswers)
        process_one_question(questionid,userAnswers, q_userAnswers, first, firstUserActs, recommendedQueries, best_peformances_by_ratings)
    process_Likert_Scale_Questions(likertScaleQuestionsMeanAndStd, likertScaleAnswers)

def process_Likert_Scale_Questions(likertScaleQuestionsMeanAndStd, likertScaleAnswers):
    """
    Compute all results from user answers to questions with possible answers from Likert scale converted to number representation
      and save them as graphs

    Parameters
    ----------
    likertScaleQuestionsMeanAndStd : pd.DataFrame
        DataFrame containing mean and std of users answers to questions with possible answers from Likert scale
    likertScaleAnswers : _type_
        DataFrame containing answers to questions with possible answers from Likert scale converted to number representation
    """
    likertScaleQuestionsMeanAndStd = pd.concat(likertScaleQuestionsMeanAndStd)
    likertScaleQuestionsMeanAndStd = likertScaleQuestionsMeanAndStd.reset_index()
    likertScaleQuestionsMeanAndStd.to_csv((os.path.join(folder_with_graphs,f"LikertScaleQuestionsAnswers.csv")))
    likertScaleAnswers = pd.concat(likertScaleAnswers)
    for section in likertScaleAnswers["sectionname"].unique():
        section_LikertScaleAnswers = likertScaleAnswers[likertScaleAnswers["sectionname"] == section]
        section_likertScaleQuestionsMeanAndStd = likertScaleQuestionsMeanAndStd[\
        likertScaleQuestionsMeanAndStd["question"].isin(section_LikertScaleAnswers["questiontext"])]
        g = sns.barplot(section_likertScaleQuestionsMeanAndStd, x="mean",y="question")
        y_coords = [p.get_y() + 0.5*p.get_height() for p in g.patches]
        x_coords = [p.get_width() for p in g.patches]        
        g.set_xlim(-1.1,1.1)        
        g.errorbar(x=x_coords, y=y_coords, xerr=section_likertScaleQuestionsMeanAndStd["std"], fmt="none", c= "k")
        wrap_labels(g, 20)
        plt.savefig(os.path.join(folder_with_graphs,f"{section.replace(' ', '_')}LikertScaleQuestionsMeanAndStd.png"), bbox_inches='tight')
        plt.close('all')
        plt.clf()
        plt.cla()

def get_Likert_Scale_int_value_mean_and_std_dfs(q_userAnswers):
    """_summary_

    Parameters
    ----------
    q_userAnswers : pd.DataFrame
        Dataset with user answers to the question

    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        DataFrame containing mean and std of users answers to questions with possible answers from Likert scale
        Dataset with user answers to the question with new column representing number representation of answer
    """
    q_userAnswers["numvalue"] = [LikertScale[answer] for answer in q_userAnswers["answer"]]
    stats = q_userAnswers.groupby(["questiontext"]).agg({"numvalue": ["mean","std"]})
    stats = stats.reset_index()
    mean_std_df = pd.DataFrame({
            "question":stats["questiontext"],
            "mean":stats["numvalue","mean"],
            "std":stats["numvalue","std"]
    })
    return  mean_std_df , q_userAnswers

def get_possible_answers_to_question(questionid, questionType):
    """
    
    Parameters
    ----------
    questionid : int
        ID of the question
    questionType : int
        Type of the question
    
    Returns
    -------
    list
        Possible answers to the question
    """
    allAnswers = get_table("Answers")

    if (questionType == 0):
        answersToQuestion = list(LikertScale.keys())
        answersToQuestion.reverse()
    elif (questionType == 1):
        answersToQuestion = list(allAnswers[allAnswers["questionid"] == questionid]["text"].unique())
    return answersToQuestion

def process_one_question(questionid, userAnswers, q_userAnswers, first, firstUserActs, recommendedQueries, best_peformances_by_ratings):
    """
    Compute results from user answers to one question and save them as graphs

    Parameters
    ----------
    questionid : int
        ID of questions
    userAnswers : pd.DataFrame
        Dataset with user answers
    firstUserActs : pd.DataFrame
        DataFrame with all users first acts from each group of acts
    """
    questiontext = first["questiontext"]
    sectionname = first["sectionname"]
    questionType = first["answertype"]    
    number_of_users_with_answers = len(userAnswers["userid"].unique())
    answersToQuestion = get_possible_answers_to_question(questionid, questionType)
    process_one_question_all_answers(questionid, q_userAnswers, questiontext, answersToQuestion, number_of_users_with_answers)
    process_one_question_first_type_of_act(questionid, sectionname, firstUserActs, q_userAnswers, questiontext, answersToQuestion,\
                                            number_of_users_with_answers)
    process_one_question_most_used_act_in_query(questionid, sectionname, recommendedQueries, q_userAnswers, questiontext,\
                                                 answersToQuestion, number_of_users_with_answers)
    process_one_question_best_performances_by_ratings_in_query(questionid, sectionname, best_peformances_by_ratings, q_userAnswers, questiontext,\
                                                 answersToQuestion, number_of_users_with_answers)
    process_one_question_other_question(questionid, sectionname,userAnswers, q_userAnswers, questiontext, answersToQuestion,\
                                        number_of_users_with_answers)

def process_one_question_all_answers(questionid, q_userAnswers, questiontext, answersToQuestion, y_max):
    """
    Saves graph of user answers to the question
    
    Parameters
    ----------
    questionid : int
        ID of the question
    q_userAnswers : pd.DataFrame
        Dataset with user answers to the question
    questiontext : str
        Text of the question
    answersToQuestion : list
        Possible answers to the question
    """
    counts = []
    for possibleAnswer in answersToQuestion:
        counts.append(len(q_userAnswers[q_userAnswers["answer"] == possibleAnswer]))
    question_userAnswers = pd.DataFrame({
        "answer" : answersToQuestion,
        "count" : counts
    })
    cm = sns.color_palette("plasma",len(question_userAnswers["answer"].unique()))
    g = sns.barplot(data=question_userAnswers, x="answer",y="count")
    for bin_,i in zip(g.patches, cm):
        bin_.set_facecolor(i)
    g.set(title=("\n".join(wrap(questiontext, 60))))
    g.set_ylim(0,y_max+2)
    wrap_labels(g, 12)
    plt.savefig(os.path.join(folder_with_graphs,f"{questionid}_answers.png"), bbox_inches='tight')
    plt.close('all')
    plt.clf()
    plt.cla()
    


def process_one_question_first_type_of_act(questionid, sectionname, firstUserActs, q_userAnswers, questiontext,\
                                           answersToQuestion,y_max):
    """
    Saves graph of user answers to the question based on first act from group of acts

    Parameters
    ----------
    questionid : int
        ID of the question
    sectionname : str
        name of the questions section where this question belongs
    firstUserActs : pd.DataFrame
        DataFrame with all users first acts from each group of acts
    q_userAnswers : pd.DataFrame
        Dataset with user answers to the question
    questiontext : str
        Text of the question
    answersToQuestion : list
        Possible answers to the question
    """
    for typeOfAct in ActTypeToQuestionSection[sectionname]:
        type_userActs = firstUserActs[firstUserActs["typeofact"] == typeOfAct]
        type_userAnswers = pd.merge(q_userAnswers, type_userActs, how="left", on=["userid"])
        actName = f"First variant of {typeOfAct}"
        type_userAnswers.sort_values("actcode", inplace=True)
        type_userAnswers.rename(columns={"actcode" : actName}, inplace=True)
        data = []
        for possibleAnswer in answersToQuestion:
            type_userAnswers_by_answer = type_userAnswers[type_userAnswers["answer"] == possibleAnswer]
            for nameOfAct in type_userAnswers[actName].unique():
                type_userAnswers_by_act = type_userAnswers_by_answer[type_userAnswers_by_answer[actName] == nameOfAct]
                data.append(pd.DataFrame({
                    "answer" : [possibleAnswer],
                    "count" : [len(type_userAnswers_by_act)],                    
                    actName: ["\n".join(wrap(nameOfAct, 25))]
                }))
        type_userAnswers = pd.concat(data)
        cm = sns.color_palette("plasma",len(type_userAnswers[actName].unique()))
        g = sns.barplot(data=type_userAnswers, x="answer", y="count", hue = actName, palette=cm)
        g.set(title=("\n".join(wrap(questiontext, 40))))
        g.set_ylim(0,y_max / len(type_userAnswers[actName].unique()) + 2)
        wrap_labels(g, 12)
        sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
        plt.savefig(os.path.join(folder_with_graphs,f"by_{typeOfAct}_{questionid}_answers.png"), bbox_inches='tight')
        plt.close('all')
        plt.clf()
        plt.cla()
        #plt.show()

def process_one_question_best_performances_by_ratings_in_query(questionid, sectionname, best_peformances_by_ratings, q_userAnswers, questiontext,\
                                                 answersToQuestion, y_max):
    """
    Saves graph of user answers to the question based on first act from group of acts

    Parameters
    ----------
    questionid : int
        ID of the question
    sectionname : str
        name of the questions section where this question belongs
    best_peformances_by_ratings : pd.DataFrame
        DataFrame with variants of tweak mechanism, metrics,... used in recommender query that had most positive ratings 
    q_userAnswers : pd.DataFrame
        Dataset with user answers to the question
    questiontext : str
        Text of the question
    answersToQuestion : list
        Possible answers to the question
    """
    for typeOfAct in ActTypeToQuestionSection[sectionname]:
        type_userActs = best_peformances_by_ratings[best_peformances_by_ratings["typeofact"] == typeOfAct]
        if (len(type_userActs)) == 0:
            continue
        type_userAnswers = pd.merge(q_userAnswers, type_userActs,  on=["userid"])
        actName = "\n".join(wrap(f"Most given positive ratings per seen when used variant of {typeOfAct} in recommender queries", 25))
        type_userAnswers.sort_values("actcode", inplace=True)
        type_userAnswers.rename(columns={"actcode" : actName}, inplace=True)
        data = []
        for possibleAnswer in answersToQuestion:
            type_userAnswers_by_answer = type_userAnswers[type_userAnswers["answer"] == possibleAnswer]
            for nameOfAct in type_userAnswers[actName].unique():
                type_userAnswers_by_act = type_userAnswers_by_answer[type_userAnswers_by_answer[actName] == nameOfAct]
                
                data.append(pd.DataFrame({
                    "answer" : [possibleAnswer],
                    "count" : [len(type_userAnswers_by_act)],
                    actName: ["\n".join(wrap(nameOfAct, 25))]
                }))
        type_userAnswers = pd.concat(data)
        cm = sns.color_palette("plasma",len(type_userAnswers[actName].unique()))
        g = sns.barplot(data=type_userAnswers, x="answer", y="count", hue = actName, palette=cm)
        g.set(title=("\n".join(wrap(questiontext, 40))))
        g.set_ylim(0,y_max / len(type_userAnswers[actName].unique()) + 2)
        wrap_labels(g, 12)
        sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
        plt.savefig(os.path.join(folder_with_graphs,f"by_best_performed_on_ratings_{typeOfAct}_variant_{questionid}_answers.png"), bbox_inches='tight')
        plt.close('all')
        plt.clf()
        plt.cla()




def process_one_question_most_used_act_in_query(questionid, sectionname, recommenderQueries, q_userAnswers, questiontext,\
                                                answersToQuestion, y_max):
    """
    Saves graph of user answers to the question based on first act from group of acts

    Parameters
    ----------
    questionid : int
        ID of the question
    sectionname : str
        name of the questions section where this question belongs
    recommenderQueries : pd.DataFrame
        DataFrame with most used actions for recommenderquery (tweak emchanism, metric variants)
    q_userAnswers : pd.DataFrame
        Dataset with user answers to the question
    questiontext : str
        Text of the question
    answersToQuestion : list
        Possible answers to the question
    """
    for typeOfAct in ActTypeToQuestionSection[sectionname]:
        type_userActs = recommenderQueries[recommenderQueries["typeofact"] == typeOfAct]
        if (len(type_userActs)) == 0:
            continue
        type_userAnswers = pd.merge(q_userAnswers, type_userActs,  on=["userid"])
        actName = f"Most used variant of {typeOfAct} in recommender queries"
        type_userAnswers.sort_values("actcode", inplace=True)
        type_userAnswers.rename(columns={"actcode" : actName}, inplace=True)
        data = []
        for possibleAnswer in answersToQuestion:
            type_userAnswers_by_answer = type_userAnswers[type_userAnswers["answer"] == possibleAnswer]
            for nameOfAct in type_userAnswers[actName].unique():
                type_userAnswers_by_act = type_userAnswers_by_answer[type_userAnswers_by_answer[actName] == nameOfAct]
                
                data.append(pd.DataFrame({
                    "answer" : [possibleAnswer],
                    "count" : [len(type_userAnswers_by_act)],
                    actName: ["\n".join(wrap(nameOfAct, 25))]
                }))
        type_userAnswers = pd.concat(data)
        cm = sns.color_palette("plasma",len(type_userAnswers[actName].unique()))
        g = sns.barplot(data=type_userAnswers, x="answer", y="count", hue = actName, palette=cm)
        g.set(title=("\n".join(wrap(questiontext, 40))))
        g.set_ylim(0,y_max / len(type_userAnswers[actName].unique()) + 2)
        wrap_labels(g, 12)
        sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
        plt.savefig(os.path.join(folder_with_graphs,f"by_recommender_query_{typeOfAct}_{questionid}_answers.png"), bbox_inches='tight')
        plt.close('all')
        plt.clf()
        plt.cla()
        #plt.show()

def process_one_question_other_question(questionid, sectionname,userAnswers, q_userAnswers, questiontext,
                                         answersToQuestion, y_max):
    """

    Parameters
    ----------
    questionid : int
        ID of the question
    userAnswers : pd.DataFrame
        Dataset with user answers
    sectionname : str
        name of the questions section where this question belongs
    firstUserActs : pd.DataFrame
        DataFrame with all users first acts from each group of acts
    q_userAnswers : pd.DataFrame
        Dataset with user answers to the question
    questiontext : str
        Text of the question
    answersToQuestion : list
        Possible answers to the question
    """
    setOfDependentQuestions = set(userAnswers[(userAnswers["sectionname"] == sectionname)]["questionid"].unique())
    setOfDependentQuestions = setOfDependentQuestions |\
          set(userAnswers[(userAnswers["sectionname"] == "Demographics")]["questionid"].unique())
    for sname in QuestionSectionsDependency[sectionname]:
        setOfDependentQuestions = setOfDependentQuestions |\
          set(userAnswers[(userAnswers["sectionname"] == sname)]["questionid"].unique())
    setOfDependentQuestions.remove(questionid)
    for qid in setOfDependentQuestions:
        dependent_userAnswers = userAnswers[userAnswers["questionid"] == qid]
        sname = dependent_userAnswers.iloc[0]["sectionname"]
        qtype = dependent_userAnswers.iloc[0]["answertype"]
        qtext = dependent_userAnswers.iloc[0]["questiontext"]
        dependent_userAnswers = dependent_userAnswers.add_suffix(f'_{qid}')
        dependent_userAnswers = dependent_userAnswers.rename(columns={f"userid_{qid}":"userid"})
        dependent_userAnswers = pd.merge(q_userAnswers, dependent_userAnswers, on="userid")
        dependent_answerColumn = "\n".join(wrap(qtext, 40))
        dependent_userAnswers.rename(columns={f"answer_{qid}" : dependent_answerColumn}, inplace=True)
        answersToDependentQuestion = get_possible_answers_to_question(qid,qtype)
        data = []
        for possibleAnswer in answersToQuestion:
            dependent_userAnswers_by_answer = dependent_userAnswers[dependent_userAnswers["answer"] == possibleAnswer]
            for possibleDependentAnswer in answersToDependentQuestion:
                dependent_userAnswers_by_dependent = dependent_userAnswers_by_answer\
                    [dependent_userAnswers_by_answer[dependent_answerColumn] == possibleDependentAnswer]
                data.append(pd.DataFrame({
                    "answer" : [possibleAnswer],
                    "count" : [len(dependent_userAnswers_by_dependent)],
                    dependent_answerColumn: ["\n".join(wrap(possibleDependentAnswer, 25))]
                }))
        dependent_userAnswers = pd.concat(data)
        cm = sns.color_palette("plasma",len(dependent_userAnswers[dependent_answerColumn].unique()))
        g = sns.barplot(data=dependent_userAnswers, x="answer", y="count", hue = dependent_answerColumn, palette=cm)
        g.set(title=("\n".join(wrap(questiontext, 60))))
        g.set_ylim(0, max(y_max / 2, dependent_userAnswers["count"].max()))
        wrap_labels(g, 12)
        sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
        plt.savefig(os.path.join(folder_with_graphs,f"by_{qid}_answers_{questionid}.png"), bbox_inches='tight')
        plt.close('all')
        plt.clf()
        plt.cla()
        #plt.show()

def number_of_users_made_act(withFirstUserActs):
    """
    Computes and saves dataset containing how many users have completed each act

    Parameters
    ----------
    withFirstUserActs : bool
        count users that has the action assigned as default
    """
    global discarded_users, author_users
    userActs = get_table("UserActs")
    acts = get_table("Acts")
    userActs = pd.merge(userActs, acts, left_on="actid", right_on="id")
    userActs = userActs[~userActs["userid"].isin(author_users)]
    userActs = userActs[~userActs["userid"].isin(discarded_users)]
    if (not withFirstUserActs):
        firstUserActs = get_first_user_acts()[["userid","actid","priority"]]
        userActs = pd.merge(userActs, firstUserActs, how="left", on=["userid","actid"])
        userActs = userActs[userActs["priority_y"].isna()]
    groupedUserActs = userActs.groupby(["code","typeofact"])["id_x"].count()
    groupedUserActs = groupedUserActs.reset_index()    
    groupedUserActs = groupedUserActs.rename(columns={"id_x": "count"})
    groupedUserActs["from_all"] = groupedUserActs["count"] / len(userActs["userid"].unique())
    groupedUserActs[["code","count","from_all","typeofact"]]\
        .to_csv(os.path.join(folder_with_graphs,f"number_of_users_made_act_{str(withFirstUserActs)}.csv"))

def number_of_finished_groups_of_acts_by_priority(usersThatAnswered):
    """
    Computes and saves dataset containing how many groups of acts user averagely finished for each priority

    Parameters
    ----------
    usersThatAnswered : bool
        count only users that answered atleast one question
    """
    global author_users, users_without_questionnaire
    userActs = get_table("UserActs")
    acts = get_table("Acts")
    acts["groupsize"] = [len(acts[acts["typeofact"] == typeOfAct]) for typeOfAct in acts["typeofact"]]
    userActs = pd.merge(userActs, acts, left_on="actid", right_on="id")
    userActs = userActs[~userActs["userid"].isin(author_users)]
    if (usersThatAnswered):
        userActs = userActs[~userActs["userid"].isin(users_without_questionnaire)]
    userActs["priority"] = userActs["priority"].astype('category')
    userActs = userActs.groupby(["userid","typeofact", "groupsize", "priority"])["id_x"].count()
    userActs = userActs.reset_index()
    userActs = userActs.rename(columns={"id_x": "count"})
    userActs["completed_group"] = userActs["groupsize"] == userActs["count"]
    userActs = userActs[userActs["completed_group"] == True]
    userActs = userActs.groupby(["userid", "priority"])["groupsize"].count()
    userActs = userActs.reset_index()
    userActs = userActs.rename(columns={"groupsize": "count"})
    userActs["priority_acts"] = [len(acts[acts["priority"] == priority]["typeofact"].unique()) 
                                 for priority in userActs["priority"]]
    userActs["count"] = userActs["count"] / userActs["priority_acts"]
    output = userActs.groupby(['priority'], as_index=False).agg({'count':['mean','std']})
    output = output.reset_index()
    output.to_csv(os.path.join(folder_with_graphs,f"number_of_finished_groups_of_acts_by_priority_{str(usersThatAnswered)}.csv"))

def number_of_recommended_queries(usersThatAnswered):    
    """
    Computes mean and std of number of recommended queries made by user
    
    Parameters
    ----------
    usersThatAnswered : bool
        count only users that answered atleast one question
    """
    colnames=["relevance type","diversity type","novelty type","popularity type","calibration type",
              "relevance","diversity","novelty","popularity","calibration","tweak mechanism", "userid", "date"] 
    recommenderqueries = pd.read_csv("Logs/RecommenderQueries.txt", sep=';', names=colnames)
    if (usersThatAnswered):
        recommenderqueries = recommenderqueries[~recommenderqueries["userid"].isin(users_without_questionnaire)]
    recommenderqueries = recommenderqueries.groupby(["userid"])["date"].count()
    recommenderqueries = recommenderqueries.reset_index()
    recommenderqueries = recommenderqueries.rename(columns={"date": "count"})
    recommenderqueriescount = recommenderqueries.agg({'count':['mean','std']})
    recommenderqueriescount.to_csv(os.path.join(folder_with_graphs,f"recommenderqueriescount_{str(usersThatAnswered)}.csv"))

def time_in_user_study(two_hours_max):
    """
    Computes time spent in user study 
        : from first recommender query to first answer to question
        : from first recommender query to last answer to question

    Parameters
    ----------
    two_hours_max : bool
        count only with users that spend less than 2 hours in the user study (performed study at once)
    """
    users = get_table("Users")
    users = users[~users["firstrecommendationtime"].isna()]
    #format = cnv_csharp_date_fmt("dd-MM-yyyy HH:mm:ss.f")
    users["firstrecommendationtime"] = pd.to_datetime(users["firstrecommendationtime"], format="%Y-%m-%d %H:%M:%S.%f")
    userAnswers = get_userAnswers()
    userAnswers["date"] = pd.to_datetime(userAnswers["date"], format="%Y-%m-%d %H:%M:%S.%f")
    userAnswersMin = userAnswers.groupby("userid")["date"].min()
    userAnswersMin = userAnswersMin.reset_index()
    userAnswersMin = userAnswersMin.rename(columns={"date": "firstanswer"})
    userAnswersMax = userAnswers.groupby("userid")["date"].max()
    userAnswersMax = userAnswersMax.reset_index()
    userAnswersMax = userAnswersMax.rename(columns={"date": "lastanswer"})
    userAnswers = pd.merge(userAnswersMin, userAnswersMax, on="userid")
    users = pd.merge(users, userAnswers, left_on="id", right_on="userid")
    users["minutes_to_first_answer"] = users["firstanswer"] - users["firstrecommendationtime"]
    users["minutes_to_first_answer"] = [delta.total_seconds() / 60 for delta in users["minutes_to_first_answer"]]
    users["minutes_to_last_answer"] = users["lastanswer"] - users["firstrecommendationtime"]
    users["minutes_to_last_answer"] = [delta.total_seconds() / 60 for delta in users["minutes_to_last_answer"]]
    if (two_hours_max):
        users = users[users["minutes_to_last_answer"] <=120]
    minutes_to_first_answer = users.agg({'minutes_to_first_answer':['mean','std']})
    minutes_to_first_answer.to_csv(os.path.join(folder_with_graphs,f"minutes_to_first_answer{str(two_hours_max)}.csv"))
    minutes_to_last_answer = users.agg({'minutes_to_last_answer':['mean','std']})
    minutes_to_last_answer.to_csv(os.path.join(folder_with_graphs,f"minutes_to_last_answer{str(two_hours_max)}.csv"))

def variant_performance_on_ratings_by_user_graph(df, column):
    """
    Graph of mean positive ratings by user when variants of {column} (metric, tweak mechanism) used in recommmender query

    Parameters
    ----------
    df : pd.DataFrame
        Enriched recommender query dataset of users seen, clicks, ratings and positive ratings 
    column : str
        name of column with different variants
    """
    dict = {
        "positive_ratings_per_seen": "users mean positive ratings per seen",
        "clicks_per_seen":"users mean number of clicks per seen"
    }
    data = df.rename(columns=dict)
    if(column == "rank"):
        data[column] = round(df[column]/5)*5
    count=0
    y_max = [0.3, 0.05]
    for value in dict.values():
        g = sns.barplot(data, y=value, x=column)
        g.set(title=("\n".join(wrap(f"{value} when variants of {column} used in recommmender query", 60))))
        g.set_ylim(-0.01,y_max[count])
        wrap_labels(g, 20)
        plt.savefig(os.path.join(folder_with_graphs,f"ratings_and_interactions_per_{column.replace(' ', '_')}_{count}.png"), bbox_inches='tight')
        plt.close('all')
        plt.clf()
        plt.cla()
        count+=1

def process_interactions_and_ratings():
    """
    Computes dataset based on users interactions and ratings to the response of recommender system

    Returns
    -------
    pd.DataFrame
        DataFrame with variants of tweak mechanism, metrics,... used in recommender query that had most positive ratings 
    """
    recommender_queries_rating_interaction = get_recommender_queries_rating_interaction()   
    recommender_queries_rating_interaction[~recommender_queries_rating_interaction["userid"].isin(author_users)]
    recommender_queries_rating_interaction = recommender_queries_rating_interaction\
        [recommender_queries_rating_interaction["seens"]>0] 
    recommender_queries_rating_interaction["tweak mechanism"] = ["PlusMinusButtons" if x == "Buttons" else "DragAndDrop" \
                                                        if x=="Drag and drop"
                                                        else x 
                                                        for x in list(recommender_queries_rating_interaction["tweak mechanism"])]
    grouped_dfs = []
    acts = get_table("Acts").rename(columns={"id": "actid", "code":"actcode"})
    agg = {
        "positive_ratings_per_seen": ["mean", "std"],
        "clicks_per_seen": ["mean", "std"]
        }
    checked_columns = ["relevance type", "diversity type", "novelty type", "popularity type", "tweak mechanism", "rank"]
    for column in checked_columns:
        stats = recommender_queries_rating_interaction.groupby(column).agg(agg)        
        stats.to_csv(os.path.join(folder_with_graphs,f"ratings_and_interactions_per_{column.replace(' ', '_')}.csv"))
        variant_performance_on_ratings_by_user_graph(recommender_queries_rating_interaction, column)
    user_best_performance_df = []
    for userid in recommender_queries_rating_interaction["userid"].unique():
        u_stats = recommender_queries_rating_interaction[recommender_queries_rating_interaction["userid"] == userid]
        for column in checked_columns[:-1]:
            stats = u_stats.groupby([column, "userid"])["positive_ratings_per_seen"].mean()
            stats = stats.reset_index().sort_values("positive_ratings_per_seen").drop_duplicates(["userid"], keep="last")
            stats = stats.rename(columns={column: "actcode"})
            x=1
            user_best_performance_df.append(stats)
    user_best_performance_df = pd.concat(user_best_performance_df)
    user_best_performance_df = pd.merge(user_best_performance_df, acts[["actcode","typeofact"]], on="actcode")
    return user_best_performance_df




def process():
    """
    Compute all results and save them as graphs
    """
    set_discarded_users()
    #process_interactions_and_ratings()
    number_of_recommended_queries(True)
    number_of_recommended_queries(False)
    time_in_user_study(True)
    time_in_user_study(False)
    number_of_finished_groups_of_acts_by_priority(False)
    number_of_finished_groups_of_acts_by_priority(True)
    number_of_users_made_act(False)
    number_of_users_made_act(True)
    process_questions()
    process_metrics()
    print("DONE!")


if __name__ == "__main__":
    process()
 


