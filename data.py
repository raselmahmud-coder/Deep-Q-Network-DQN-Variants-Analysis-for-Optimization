# data.py

from dataclasses import dataclass

@dataclass
class Data:
    state: any
    action: any
    reward: any
    next_state: any
    done: any
