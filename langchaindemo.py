from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain_ollama.llms import OllamaLLM

import datetime

template = """Question: {question}

Answer: Let's think step by step."""

prompt = ChatPromptTemplate.from_template(template)

model = OllamaLLM(model="llama3.2",host="127.0.0.1:11434")

chain = prompt | model

#print(chain.invoke({"question": "What is LangChain?"}))


from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage("Keep answer short as possible, dont give big answers"),
    HumanMessage("give me 3 best christopher nolan movies"),
]

#print(model.invoke(messages))

from langchain_ollama import ChatOllama

model = ChatOllama(model="llama3.2",host="127.0.0.1:11434")

def get_current_time(*args,**kawrgs):
    """
    Args No Arguments 
    returns current time """
    now=datetime.datetime.now()
    return now.strftime("%I:%M %p")

def write_time_into_file(time,**kawrgs):
    """
    Args No Arguments 
    returns current time """
    print("Time is written in the file")
    now=datetime.datetime.now()
    return None



print(get_current_time())
import os
#os.exit(1)
from langchain import hub
prompt = hub.pull("hwchase17/react")

#promt_template=ChatPromptTemplate.from_template(prompt)

tools=[
    Tool(
        name="getTime",
        func=get_current_time,
        description="Usefull when you want to get time"
    ),
    Tool(
        name="setTime",
        func=write_time_into_file,
        description="write time into file"
    )
]

#print(model.invoke(messages).content)

from langchain.agents import (
    AgentExecutor,
    create_react_agent,
)

agent=create_react_agent(
    llm=model,
    tools=tools,
    prompt=prompt
)

ae = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

res=ae.invoke({"input":"what time is it?"})

print('response',res)

