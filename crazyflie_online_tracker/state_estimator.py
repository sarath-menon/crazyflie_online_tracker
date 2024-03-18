#!/usr/bin/env python3
from abc import ABC, abstractmethod
from controller import MotionIndex


class StateEstimator(ABC):

    def __init__(self):
        self.state_pub = None
        self.state = None

    @abstractmethod
    def publish_state(self, state):
        pass
