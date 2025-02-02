
from typing import List

from phi.tools import Toolkit
from phi.utils.log import logger





class ME(Toolkit):
    def __init__(self):
        """ Use this tool to get more information about me"""
        super().__init__(name="my info")
        self.name_me = "Nishant"
        self.address = "New York, Hell's Kitchen, Apt 5, Floor 3"
        self.age = "18"
        self.wife = "No wife"
        self.register(self.get_name)
        self.register(self.get_address)
        self.register(self.get_age)
        self.register(self.get_wife)

    def get_name(self):
        """
        Retrieves the name of the person talking to you.

        Args:
            None

        Returns:
            str: The name of the person talking to you.
        """
        return self.name_me

    def get_address(self):
        """
        Retrieves the address of the person talking to you.

        Args:
            None

        Returns:
            str: The address of the person talking to you.
        """
        return self.address

    def get_age(self):
        """
        Retrieves the age of the person talking to you.

        Args:
            None

        Returns:
            int: The age of the person talking to you.
        """
        return self.age

    def get_wife(self):
        """
        Retrieves the marital status or spouse information of the person talking to you.

        Args:
            None

        Returns:
            str: Information about the persons wife who is talking to you.
        """
        return self.wife