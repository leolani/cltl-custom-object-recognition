import abc
from cltl.combot.event.emissor import LeolaniContext


class VisualResponder(abc.ABC):
    def respond(self, statement: str, context: LeolaniContext) -> str:
        raise NotImplementedError("")
