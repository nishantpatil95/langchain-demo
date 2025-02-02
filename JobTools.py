from langchain_core.tools import Tool
from typing import Optional, Type

import aiohttp
import requests

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools import BaseTool
from pydantic import BaseModel

JobsData=[
            {
                "jobID":1,
                "databaseName":"crysis1",
                "state":"failed",
                "reason":"database not open"
            },
            {
                "jobID":2,
                "databaseName":"crysis2",
                "state":"success",
            }
            ]

def getAllJobs(*args,**kawrgs):

    return JobsData

class GetAllJobsTool(BaseTool):
    """get_all_jobs."""

    name: str = "get_all_jobs"
    description: str = """Usefull when you want get all jobs info
        Args : None"""



    def _run(
        self,
        *tool_args, **tool_kwargs
    ) -> dict:
        """Run the tool"""
 
        return JobsData


class GetDatabaseNameFromJobID(BaseTool):
    """
    
    GetDatabaseNameFromJobID."""

    name: str = "GetDatabaseNameFromJobID"
    description: str = """Usefull when you want get all jobs info
    Input arguments
        1. JobID:int : Valid JobID can be found in get All Jobs API
    """
    #args_schema: Type[GetHuggingFaceModelsToolSchema] = GetHuggingFaceModelsToolSchema
    #base_url: str = "<https://huggingface.co/api/models>"
    #api_key: str = Field(..., env="HUGGINGFACE_API_KEY")



    def _run(
        self,
        JobID: str
    ) -> dict:
        """Run the tool"""
    
        if not JobID in JobsData:
            return "JobID {JobID} Not found"
        return JobsData[JobID]["databaseName"]

    # async def _arun(
    #     self,
    #     path: str = "",
    #     query_params: Optional[dict] = None,
    #     run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    # ) -> dict:
    #     """Run the tool asynchronously."""

    #     async with aiohttp.ClientSession() as session:
    #         async with session.get(
    #             self.base_url + path, params=query_params, headers=self._headers
    #         ) as response:
    #             return await response.json()


def getTools():
    return [
        GetAllJobsTool(),
    #    GetDatabaseNameFromJobID()
    #     Tool(
    #     name="get All Jobs status",
    #     func=getAllJobs,
    #     description="Usefull when you want get all jobs info"
    # )
]

