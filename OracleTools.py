from langchain_core.tools import Tool
#*args,**kawrgs
dblist= {
        "crysis1":"CLOSE",
        "crysis2":"CLOSE",
        "crysis3":"CLOSE",
        "crysis4":"CLOSE",
        "crysis5":"CLOSE",
    }

def getDBStatus(databaseName,**kawrgs):
    print(databaseName)
    
    if not databaseName in dblist:
        return "Database "+databaseName+" not found."
    return "database "+databaseName+" is "+dblist[databaseName.lower()]

def restartDatabase(databaseName,**kawrgs):
    if not databaseName in dblist:
        return "Database "+databaseName+" not found."
    return f"Database {databaseName} restarted successfully"

def startDatabase(databaseName,**kawrgs):
    if not databaseName in dblist:
        return "Database "+databaseName+" not found."
    dblist[databaseName]="OPEN"
    return f"Database {databaseName} started successfully"

def getTools():
    return [
        Tool(
        name="get Database status",
        func=getDBStatus,
        description="""Usefull when you want check if database is OPEN or CLOSE
        Input arguments
        1. JobID:int : Valid JobID can be found in get All Jobs API"""
    ),
    Tool(
        name="restart database",
        func=restartDatabase,
        description="""Usefull when you want to restart database
        Input arguments
        1. databasename:str : Name of the database to restart"""
    ),
    Tool(
        name="start database",
        func=restartDatabase,
        description="""Usefull when you want to restart database
        Input arguments
        1. databasename:str : Name of the database to start"""
    )
]