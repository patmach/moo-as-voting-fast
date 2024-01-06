def get_connection_string():
    DriverName = "SQL Server" #code running locally
#    DriverName = "ODBC Driver 18 for SQL Server" #code running in container
    ServerName =  "np:\\\\.\\pipe\LOCALDB#2EB6953D\\tsql\\query" #local sql server need to update the code everytime sql server was restarted
#    ServerName = "localhost,1401" #sql server in local container
#    ServerName = "sql-server-db" #sql server on same contianer network as this code
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