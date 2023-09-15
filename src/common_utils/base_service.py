"""
Module: common_utils.base_service

Basic operation of a service repository

"""
from abc import ABC, abstractmethod


class BaseService(ABC):
    """
    Abstract service for micro services
    """
    def __init__(self) -> None:
        """
        Initialize service resources

        @param session: Database session that service's business will be executed on

        """
        super().__init__()


    def __del__(self):
        """
        Clean up service object & resource
        """
        self.tear_down()


    @abstractmethod
    def set_up(self):
        """
        Service's specific initialization
        """
        pass

    @abstractmethod
    def tear_down(self):
        """
        Service's specific clean-up
        """
        pass
