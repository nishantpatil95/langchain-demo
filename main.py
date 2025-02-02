



from phi.agent import Agent

from phi.model.groq import Groq

import apiexprt
import myinfo


agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[myinfo.ME()],
    show_tool_calls=True,
    markdown=True,
    debug_mode=True
    #instructions=[""]
)

agent.print_response("what else you know about me?")

