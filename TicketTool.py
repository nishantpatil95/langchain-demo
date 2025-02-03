from langchain.tools import StructuredTool
from langchain.tools.base import BaseTool

# Ticket storage
ticket_list = {}

class CreateTicketTool(BaseTool):
    """Creates a new ticket with a job ID and associated database name."""
    
    name: str = "create_ticket"
    description: str = """Useful when you want to create a ticket.
        Args:
        jobID: int - Unique Job ID for the ticket. (Must be mentioned by user)
        databaseName: str - Name of the related database. (Must be mentioned by user)"""

    def _run(self, jobID: int, databaseName: str, *args, **kwargs) -> str:
        if jobID in ticket_list:
            return f"Ticket with JobID {jobID} already exists."
        
        ticket_list[jobID] = {
            "isResolved": False,
            "databaseName": databaseName,
            "Solution": None
        }
        return f"Ticket {jobID} created successfully."

class GetTicketTool(BaseTool):
    """Retrieves details of a ticket by job ID."""
    
    name: str = "get_ticket"
    description: str = """Useful when you want to retrieve ticket details.
        Args:
        jobID: int - Unique Job ID of the ticket."""

    def _run(self, jobID: int, *args, **kwargs) -> dict:
        if jobID not in ticket_list:
            return {"error": f"Ticket with JobID {jobID} not found."}
        return ticket_list[jobID]

class UpdateTicketTool(BaseTool):
    """Updates an existing ticket's resolution status or solution details."""
    
    name: str = "update_ticket"
    description: str = """Useful when you want to update a ticket.
        Args:
        jobID: int - Unique Job ID of the ticket.
        isResolved: bool - True if the issue is resolved.
        Solution: str - Solution applied to resolve the issue."""

    def _run(self, jobID: int, isResolved: bool = None, Solution: str = None, *args, **kwargs) -> str:
        if jobID not in ticket_list:
            return f"Ticket with JobID {jobID} not found."
        
        if isResolved is not None:
            ticket_list[jobID]["isResolved"] = isResolved
        if Solution is not None:
            ticket_list[jobID]["Solution"] = Solution
        
        return f"Ticket {jobID} updated successfully."

def getTools():
    return [
        CreateTicketTool(),
        GetTicketTool(),
        UpdateTicketTool()
    ]