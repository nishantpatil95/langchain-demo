
import OracleTools
import JobTools

from langchain_ollama import ChatOllama
from  langchain_core.messages import SystemMessage,HumanMessage,AIMessage

from langchain.memory import ConversationBufferMemory
from langchain_core.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import apiexprt

import langgraph
from langgraph.checkpoint.memory import MemorySaver

#model=ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")

model = ChatOllama(model="llama3.2",host="127.0.0.1:11434")
#model = ChatOllama(model="deepseek-r1",host="127.0.0.1:11434")
#... (this Thought/Action/Action Input/Observation can repeat max 10 times)
from langchain import hub
prompt = hub.pull("hwchase17/react")

systemMsg = '''Answer the following questions as best you can. 
Dont use any tools unless user ask you to use it
Keep answer short and quick
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

for t in OraTools:
    tools.append(t)
for t in JobTools:
    tools.append(t)

from langchain.agents import (
    AgentExecutor,
    create_react_agent,
)

prompt = hub.pull("hwchase17/structured-chat-agent")

# agent=create_react_agent(
#    llm=model,
#    tools=tools,
#    prompt=prompt,
# )
model.bind_tools(tools)
agent = create_structured_chat_agent(model, tools, prompt)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

ae = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    memory=memory,
    max_iterations=10,
)

chathistory=[]

systemessage=SystemMessage([
    
    "You are an support engineer",
    " Backup jobs are running for different database.",
    " If any job fails then you will help user to resolve the error",
    "Multiple Jobs are running for each database (databaseName)",
    "JobList also has Job Info which conatins DatabaseName and its current State",
    "Failure reason can be found in Job List"
    ])

chathistory.append(systemessage)





from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from typing import Literal

tool_node = ToolNode(tools)

def should_continue(state: MessagesState) -> Literal["tools", END]:
    print("INISDE SHOULD CONTINUE")
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    print("SHOULD CONTINUE LASTMSG",last_message)
    if last_message.tool_calls:
        print("TOOLS USED")
        return "tools"
    # Otherwise, we stop (reply to the user)
    print("NO TOOLS")
    return END

def call_model(state: MessagesState):
    print("INISDE call_model")
    messages = state['messages']
    last_message = messages[-1]
    print(messages)
    #response = model.invoke(messages)


    response=ae.invoke(
        {
            "input":last_message.content,
           # "chat_history":chat_history
        },
        config={"configurable": {"thread_id": 44}}
    )
    print('RESPONSE',response["chat_history"][-1])

    # We return a list, because this will get added to the existing list
    return {"messages": [{"role": "ai", "content": response["chat_history"][-1].content}]}


# Define a new graph
workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("tools", 'agent')

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable.
# Note that we're (optionally) passing the memory when compiling the graph
app = workflow.compile(checkpointer=checkpointer)
#ae.bind_tools(tools)

final_state = app.invoke(
    {"messages": [{"role": "user", "content": "hwo many jobs are there?"}]},
    config={"configurable": {"thread_id": 44}}
)
print(final_state["messages"][-1].content)



"""

while True:
    query=input("You:")
    if query=="exit":
        break

    chat_history = memory.buffer_as_messages
    print(chat_history)
    chathistory.append(HumanMessage(content=query))
    result=ae.invoke(
        {
            "input":query,
            "chat_history":chat_history
        }
    )

    chathistory.append(AIMessage(content=query))

    print("AI:",result)

"""


