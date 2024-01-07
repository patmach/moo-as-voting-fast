import pypyodbc as odbc
import pandas as pd 
import sys 
import time

DriverName = "SQL Server" #code running locally on Windows
#DriverName = "ODBC Driver 18 for SQL Server" #code running in container
#ServerName =  "np:\\\\.\\pipe\LOCALDB#29683B1E\\tsql\\query" #local sql server need to update the code everytime sql server was restarted
ServerName = "localhost,1401" #sql server in local container
#ServerName = "sql-server-db" #sql server on same contianer network as this code
DatabaseName = "aspnet-53bc9b9d-9d6a-45d4-8429-2a2761773502"
Username = 'RS'
file = open('pswd.txt',mode='r')    
Password = file.read()
file.close()

def get_connection_string():
    global DriverName, ServerName, DatabaseName, Username, Password
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



def get_ratings(connectionstring, only_MovieLens = False):
    """
    Retrieves all ratings from database

    Parameters
    ----------
    connectionstring : str
        Connection string to the db

    Returns
    -------
    pd.DataFrame
        Dataframe with columns userid, itemid, ratingscore
    """    
    conn = odbc.connect(connectionstring)
    query = "SELECT  UserID, ItemID, RatingScore FROM ratings"
    if (only_MovieLens):
        query += " r join users u on u.id = r.userid where u.username like '%movielens%' "
    df = pd.read_sql_query(query, conn)
    df.columns = df.columns.str.lower()
    conn.close()
    return df


def get_ratings_of_user(connectionstring, userID, only_positive=True, order=False):
    """
    Retrieves ratings from one user from database

    Parameters
    ----------
    connectionstring : str
        Connection string to the db
    userID : int
        ID of the user whose ratings will be returned
    only_positive : bool, optional
        if true - returns only positive ratings (> 5), by default True

    Returns
    -------
    pd.DataFrame
        Dataframe with columns userid, itemid, ratingscore
    """        
    conn = odbc.connect(connectionstring)
    query = f"""SELECT  UserID, ItemID, RatingScore 
                           FROM ratings
                           where userid = {userID} """
    if (only_positive):
        query += " and ratingscore > 5 "    
    if (order):
        query += " order by date " 
    df = pd.read_sql_query(query, conn)
    df.columns = df.columns.str.lower()
    conn.close()
    return df




def get_avg_ratings_of_items(connectionstring):
    """
    Retrieve from database dataset with all items (their IDs) and their average score

    Parameters
    ----------
    connectionstring : str
        Connection string to the db

    Returns
    -------
    pd.Dataframe
        Dataframe with columns itemid, averagescore
    """    
    conn = odbc.connect(connectionstring)
    df = pd.read_sql_query(f"""SELECT  ItemID, avg(Cast(RatingScore as Float)) as averagescore 
                           FROM ratings 
                           group by ItemID""", conn)
    df.columns = df.columns.str.lower()
    conn.close()
    return df

    
def try_to_connect_to_db(connectionstring):
    """
    Tries connect to db multiple times - checking if the database correctly started

    Parameters
    ----------
    connectionstring : str
        Connection string to the db
    """
    retrycount = 0
    while retrycount < 100:
        try:
           conn = odbc.connect(connectionstring)
           conn.close()
           break
        except Exception as e:
            print(e,file=sys.stderr)
            seconds = 90
            print (f"SQL server wasn't started yet. Or database wasn't restorted yet if its first run of the docker compose app.",\
                   file=sys.stderr)
            print (f"Retry after {seconds} sec",file=sys.stderr)
            retrycount += 1
            time.sleep(seconds)


