from abc import ABC, abstractmethod


class Preconditioner(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def apply(self, v):
        pass

    @abstractmethod
    def apply_t(self, v):
        pass
