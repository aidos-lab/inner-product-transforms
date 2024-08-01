"""
A wrapper to give our model the same signature as the 
one in pointflow and make it accept the same type of data.

"""


class TopologicalModel:
    def __init__(self, config) -> None:
        self.config = config
