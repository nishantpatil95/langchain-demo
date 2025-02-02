
import OracleTools
import JobTools

from langchain_ollama import ChatOllama
from  langchain_core.messages import SystemMessage,HumanMessage,AIMessage

from langchain.memory import ConversationBufferMemory
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
import apiexprt

import langgraph
from langgraph.checkpoint.memory import MemorySaver

#model=ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
model=ChatOpenAI(model="gpt-4o-mini")
#model = ChatOllama(model="llama3.2",host="127.0.0.1:11434")
#model = ChatOllama(model="deepseek-r1",host="127.0.0.1:11434")
#... (this Thought/Action/Action Input/Observation can repeat max 10 times)
from langchain import hub
prompt = hub.pull("hwchase17/react")

systemMsg = '''Answer the following questions as best you can. 
Dont use any tools unless user ask you to use it
Keep answer short and quick
Do NOT Repeat action
Assist user.

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Action: give final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
{agent_scratchpad}'''

human = '''

{input}

{agent_scratchpad}
'''



prompt = ChatPromptTemplate.from_messages(
    [
        ("system", systemMsg),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", human),
    ]
)

prompt = PromptTemplate.from_template(systemMsg)

tools=[]

OraTools=OracleTools.getTools()
JobTools=JobTools.getTools()

#for t in OraTools:
 #   tools.append(t)
for t in JobTools:
    tools.append(t)

from langchain.agents import (
    AgentExecutor,
    create_react_agent,
)
members=["ErrorTroubleshooter","ErrorResolver"]

systemessage2=[
    
    "You are an support engineer",
    f" you are working with {members}",
    " Backup jobs are running for different database.",
    " You can view jobs and respond to user queries regarding jobs using getJobTool"
    " You need to ask permission to user for two things",
    " 1. Investigate Job",
    " 2. Resolve Issue",
    " if user responds y for Investigate Job then respond with ErrorTroubleshooter <JobID>\n"
    ]


structrechatsystemmsg="\n".join(systemessage2) +"""
"Respond to the human as helpfully and accurately as possible. You have access to the following tools:

{tools}

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:

```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}

Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation"),
{agent_scratchpad}(reminder to respond in a JSON blob no matter what)
"""

#prompt = hub.pull("hwchase17/structured-chat-agent")
prompt = ChatPromptTemplate.from_messages([
  ("system", structrechatsystemmsg)
    ,  ("placeholder", "{chat_history}"),
  ("human", "{input}")

])
print("prompt")
print(prompt.messages)
print("prompt")
# agent=create_react_agent(
#    llm=model,
#    tools=tools,
#    prompt=prompt,
# )

agent = create_structured_chat_agent(model, tools, prompt)

memory = ConversationBufferMemory(memory_key="chat_history_2", return_messages=True)

ae = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    memory=memory,
    max_iterations=10,
)

chathistory=[]

members=["ErrorTroubleshooter","ErrorResolver"]

systemessage2=SystemMessage([
    
    "You are an support engineer",
    f" you are working with {members}",
    " Backup jobs are running for different database.",
    " You can view jobs and respond to user queries regarding jobs using getJobTool"
    " You need to ask permission to user for two things",
    " 1. Investigate Job",
    " 2. Resolve Issue",
    " if user responds y for Investigate Job then respond with ErrorTroubleshooter <JobID>"
    ])







from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langgraph.graph import START,END
from langgraph.graph.state import StateGraph
from IPython.display import Image,display


#chat_history = memory.buffer_as_messages
#chathistory.append(systemessage)
while True:
    query=input("You:")
    if query=="exit":
        break

    chat_history = memory.buffer_as_messages
    print(chat_history)
    #chathistory.append(HumanMessage(content=[{"type": "text", "text": query}]))
    result=ae.invoke(
         {
             "input":query,
             "chat_history_2":chat_history
         }
    )
    #resp=result.content
    #print(result)
    # result=ae.invoke(
    #     {"messages": [{"role": "user", "content": query}]}
    # )
    #chathistory.append(AIMessage(content=[{"type": "text", "text": resp}]))

    print("AI:",result)


