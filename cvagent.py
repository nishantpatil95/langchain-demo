
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
import string
import random
import TicketTool
import langgraph
from langgraph.checkpoint.memory import MemorySaver

#model=ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
model=ChatOpenAI(model="gpt-4o")
#model = ChatOllama(model="llama3.2",host="127.0.0.1:11434")
#model = ChatOllama(model="deepseek-r1",host="127.0.0.1:11434")
#... (this Thought/Action/Action Input/Observation can repeat max 10 times)
from langchain import hub
prompt = hub.pull("hwchase17/react")

tools=[]

JobTools=JobTools.getTools()
TicketTool=TicketTool.getTools()

#for t in JobTools:
 #   tools.append(t)
for t in TicketTool:
    tools.append(t)
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
)
members=["ErrorTroubleshooter","ErrorResolver"]

systemessage2=[
    
    "You are an support engineer",
    "You will help customer with his queries",
    "If customer asks you to investiget any issue then you can create ticket for it",
    "You can ask customer for JobID and database name which is required to create ticket"
   # "Do not use get_all_jobs unless you want to see all jobs detail",
   # "If asked by customer about failed job, you can use get_all_jobs tool.",
  #  "You will get all jobs in output, then you can parse output and identify failed job."
  #  "For each failed job, only one ticket can be created.",
  #  "Use createTicket tool to create support ticket.",
 #   "Ask permission of user to before creating ticket."
    ]


structrechatsystemmsg="\n".join(systemessage2) +"""
"Respond to the human as helpfully and accurately as possible. You have access to the following tools:
DO NOT REUSE SAME TOOL AGAIN AND AGAIN.
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
Thought: I know what to respond, Do not use get_all_jobs again and again, think about what you learn from APIs output
Action:
```
{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}

Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation"),
{agent_scratchpad}(reminder to respond in a JSON blob no matter what. THIS IS VERY IMPORTTANT)
"""

prompt = hub.pull("hwchase17/structured-chat-agent")

print("prompt")
print(prompt.messages)
print("prompt")


agent = create_structured_chat_agent(model, tools, prompt)


def generate_random_key(length):
  """Generates a random key of specified length."""

  characters = string.ascii_letters + string.digits
  return ''.join(random.choice(characters) for i in range(length))

chathistorykey=generate_random_key(16)

memory = ConversationBufferMemory(memory_key=chathistorykey, return_messages=True)

ae = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    memory=memory,
    max_iterations=100,
)


while True:
    query=input("You:")
    if query=="exit":
        break

    chat_history = memory.buffer_as_messages
    print(chat_history)
    result=ae.invoke(
         {
             "input":query,
             chathistorykey:chat_history
         }
    )

    print("AI:",result)


