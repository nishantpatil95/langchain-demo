from typing import Literal
from typing_extensions import TypedDict

from langchain_anthropic import ChatAnthropic
from langgraph.graph import MessagesState, END
from langgraph.types import Command
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq

import apiexprt
import OracleTools
import JobTools


members = ["oracle_expert", "script_executer","self"]
# Our team supervisor is an LLM node. It just picks the next agent to process
# and decides when the work is completed
options = members + ["FINISH"]
#options = members + ["self"]

system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    f" following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
    " oracle_expert is expert in trouble shooting errors in jobs and oracle"
    " once oracle_expert identifies the issue he will give steps to resolve the issue."
    " you will need to provide these steps to script_executer and he will perform these steps."
    " Assist user queries, only transfer to oracle_expert if user asked you to investigate any job"
    " If you can resolve query by yourself then respond with self"
)


class Router(TypedDict):
    """Worker to route to next. If no workers needed, route to FINISH."""

    next: Literal[*options]

tools=[]

OraTools=OracleTools.getTools()
JobTools=JobTools.getTools()

for t in OraTools:
    tools.append(t)
for t in JobTools:
    tools.append(t)

#llm = ChatAnthropic(model="claude-3-5-sonnet-latest")
#llm=ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
llm = ChatOllama(model="llama3.2",host="127.0.0.1:11434")
#llm = ChatOllama(model="deepseek-r1",host="127.0.0.1:11434")

class State(MessagesState):
    next: str


def supervisor_node(state: State) -> Command[Literal[*members, "__end__"]]:
    messages = [
        {"role": "system", "content": system_prompt},
    ] + state["messages"]
    print("supervisor_node messages",messages)

    response = llm.with_structured_output(Router).invoke(messages)
    print("supervisor_node response",response)
    print("================================")
    goto = response["next"]
    if goto == "FINISH":
        goto = END

    return Command(goto=goto, update={"next": goto})



from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent


oracle_expert_agent = create_react_agent(
    llm, tools=tools, prompt="You are a Oracle Expert. You will investigae why oracle database backup jobs are failing. You will give deatiled steps to resolve error"
)


def oracle_expert_node(state: State) -> Command[Literal["supervisor"]]:
    result = oracle_expert_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="oracle_expert")
            ]
        },
        goto="supervisor",
    )


# NOTE: THIS PERFORMS ARBITRARY CODE EXECUTION, WHICH CAN BE UNSAFE WHEN NOT SANDBOXED
script_executer_agent = create_react_agent(llm, tools=tools)


def script_executer_node(state: State) -> Command[Literal["supervisor"]]:
    result = script_executer_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="script_executer")
            ]
        },
        goto="supervisor",
    )


builder = StateGraph(State)
builder.add_edge(START, "supervisor")
builder.add_node("supervisor", supervisor_node)
builder.add_node("self", supervisor_node)
builder.add_node("oracle_expert", oracle_expert_node)
builder.add_node("script_executer", script_executer_node)
graph = builder.compile()


while True:
    query=input("You:")
    if query=="exit":
        break

    #chat_history = memory.buffer_as_messages
    #print(chat_history)
    #chathistory.append(HumanMessage(content=query))
    result=graph.invoke(
        {"messages": [("user", query)]}, subgraphs=True
    )

    #chathistory.append(AIMessage(content=query))

    print("AI:",result)

