import pypyodbc as odbc
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
from textwrap import wrap
import textwrap



FirstActTypeToQuestionSection = {
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
LikertScale = ["Strongly agree", "Agree", "Neutral / Don't Know", "Disagree", "Strongly disagree"]

def wrap_labels(ax, width, break_long_words=False):
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
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0)


def get_connection_string():
    """
        Connects to the database
    """
    DriverName = "SQL Server"
    #DriverName = "ODBC Driver 18 for SQL Server"
    ServerName =  "localhost,1401"#"np:\\\\.\\pipe\LOCALDB#ED18BEF1\\tsql\\query"
    #ServerName = "sql-server-db"
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
    userAnswers["valueanswer"] = [LikertScale[int(x)] if not np.isnan(x) else x for x in list(userAnswers["value"])]
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


def get_recommender_queries():
    """
    Returns
    -------
    pd.DataFrame
        All queries to the recommender called by user
    """
    colnames=["relevance type","diversity type","novelty type","popularity type","calibration type",
              "relevance","diversity","novelty","popularity","calibration","tweak mechanism", "user ID", "date"] 
    recommenderqueries = pd.read_csv("Logs/RecommenderQueries.txt", sep=';', names=colnames)
    bymetrics = []
    for metric in ["relevance","diversity","novelty","popularity","calibration"]:
        bymetric = pd.DataFrame()
        bymetric["Metric"] = [metric] * len(recommenderqueries)
        bymetric["Metric importance"] = recommenderqueries[metric]
        bymetric["Metric variant"] = recommenderqueries[metric+" type"]
        bymetric["Metric variant"] = [s if pd.isna(s) else s.replace("_"," ")
                                      for s in list(bymetric["Metric variant"])]
        bymetric["Tweak mechanism"] = recommenderqueries["tweak mechanism"]
        bymetric["User ID"] = recommenderqueries["user ID"]
        bymetric["Date"] = recommenderqueries["date"]
        bymetrics.append(bymetric)
    bymetrics = pd.concat(bymetrics)
    return bymetrics

def process():
    """
    Compute all results and save them as graphs
    """
    process_questions()
    process_metrics()
    print("DONE!")


def process_metrics():
    """
    Compute all stats objectives weights in recommender queries and save them as graphs
    """
    recommenderqueries = get_recommender_queries()
    cm = sns.color_palette("plasma",len(recommenderqueries["Metric"].unique()))
    g = sns.violinplot(data = recommenderqueries, x= "Metric", y= "Metric importance",
               palette=cm)
    g.set(title = f"Metrics weights specified by user")
    wrap_labels(g, 12)
    plt.savefig(os.path.join("Results",f"Metrics_importances.png"), bbox_inches='tight')
    plt.close('all')
    plt.clf()
    plt.cla()
    g = sns.violinplot(data = recommenderqueries, x= "Metric variant", y= "Metric importance",
               palette=cm)
    g.set(title = f"Metrics variants weights specified by user")
    wrap_labels(g, 6)
    plt.savefig(os.path.join("Results",f"Metrics_variants_importances.png"), bbox_inches='tight')
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
            plt.savefig(os.path.join("Results",f"variants_of_{metric}_importances.png"), bbox_inches='tight')
            plt.close('all')
            plt.clf()
            plt.cla()
            g = sns.violinplot(data = metric_recommenderqueries, x= "Tweak mechanism", y= "Metric importance",
               split=True, palette=cm)
            g.set(title = f"{metric} weight specified by user per tweak mechanism")
            wrap_labels(g, 12)
            plt.savefig(os.path.join("Results",f"by_tweak_mechanism_{metric}_importances.png"), bbox_inches='tight')
            plt.close('all')
            plt.clf()
            plt.cla() 

def process_questions():
    """
    Compute all results from user answers and save them as graphs
    """
    userAnswers = get_userAnswers()
    firstUserActs = get_first_user_acts()
    questions = get_table("Questions")

    for questionid in questions["id"]:
        process_one_question(questionid,userAnswers, firstUserActs)

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
        answersToQuestion = [answer for answer in LikertScale]
        answersToQuestion.reverse()
    elif (questionType == 1):
        answersToQuestion = list(allAnswers[allAnswers["questionid"] == questionid]["text"].unique())
    return answersToQuestion

def process_one_question(questionid, userAnswers, firstUserActs):
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
    q_userAnswers = userAnswers[userAnswers["questionid"] == questionid]   
    first = q_userAnswers.iloc[0]
    questiontext = first["questiontext"]
    sectionname = first["sectionname"]
    questionType = first["answertype"]
    answersToQuestion = get_possible_answers_to_question(questionid, questionType)
    process_one_question_all_answers(questionid, q_userAnswers, questiontext, answersToQuestion)
    process_one_question_type_of_act(questionid, sectionname, firstUserActs, q_userAnswers, questiontext, answersToQuestion)
    process_one_question_other_question(questionid, sectionname,userAnswers, q_userAnswers, questiontext, answersToQuestion)

def process_one_question_all_answers(questionid, q_userAnswers, questiontext, answersToQuestion):
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
    q_userAnswers = pd.DataFrame({
        "answer" : answersToQuestion,
        "count" : counts
    })
    cm = sns.color_palette("plasma",len(q_userAnswers["answer"].unique()))
    g = sns.barplot(data=q_userAnswers, x="answer",y="count")
    for bin_,i in zip(g.patches, cm):
        bin_.set_facecolor(i)
    g.set(title=("\n".join(wrap(questiontext, 60))))
    wrap_labels(g, 12)
    plt.savefig(os.path.join("Results",f"{questionid}_answers.png"), bbox_inches='tight')
    plt.close('all')
    plt.clf()
    plt.cla()


def process_one_question_type_of_act(questionid, sectionname, firstUserActs, q_userAnswers, questiontext, answersToQuestion):
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
    for typeOfAct in FirstActTypeToQuestionSection[sectionname]:
        type_userActs = firstUserActs[firstUserActs["typeofact"] == typeOfAct]
        type_userAnswers = pd.merge(q_userAnswers, type_userActs, how="left", on=["userid"])
        actName = f"First variant of {typeOfAct}"
        type_userAnswers.rename(columns={"actcode" : actName}, inplace=True)
        data = []
        for possibleAnswer in answersToQuestion:
            type_userAnswers_by_answer = type_userAnswers[type_userAnswers["answer"] == possibleAnswer]
            for typeOfAct in type_userAnswers[actName].unique():
                type_userAnswers_by_act = type_userAnswers_by_answer[type_userAnswers_by_answer[actName] == typeOfAct]
                data.append(pd.DataFrame({
                    "answer" : [possibleAnswer],
                    "count" : [len(type_userAnswers_by_act)],
                    actName: ["\n".join(wrap(typeOfAct.replace("_"," "), 25))]
                }))
        type_userAnswers = pd.concat(data)
        cm = sns.color_palette("plasma",len(type_userAnswers[actName].unique()))
        g = sns.barplot(data=type_userAnswers, x="answer", y="count", hue = actName, palette=cm)
        g.set(title=("\n".join(wrap(questiontext, 40))))
        wrap_labels(g, 12)
        sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
        plt.savefig(os.path.join("Results",f"by_type_of_act_{questionid}_answers.png"), bbox_inches='tight')
        plt.close('all')
        plt.clf()
        plt.cla()
        #plt.show()

def process_one_question_other_question(questionid, sectionname,userAnswers, q_userAnswers, questiontext,
                                         answersToQuestion):
    """_summary_

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
        wrap_labels(g, 12)
        sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
        plt.savefig(os.path.join("Results",f"by_{qid}_answers_{questionid}.png"), bbox_inches='tight')
        plt.close('all')
        plt.clf()
        plt.cla()
        #plt.show()

if __name__ == "__main__":
    process()
 