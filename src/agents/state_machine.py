"""
state_machine.py

Defines the state names used by the informal transport agent.

The informal transport agent is not modeled as a simple vehicle.
It moves through buildingness-relevant states such as anchoring,
boarding threshold, temporary interior, alighting threshold, and repositioning.
"""

from enum import Enum


class InformalTransportState(str, Enum):
    """
    States for the informal transport finite state machine.
    """

    IDLE_AT_ANCHOR = "IDLE_AT_ANCHOR"
    SEARCHING_FOR_DEMAND = "SEARCHING_FOR_DEMAND"
    APPROACHING_PASSENGER = "APPROACHING_PASSENGER"
    BOARDING_THRESHOLD = "BOARDING_THRESHOLD"
    MOVING_WITH_PASSENGER = "MOVING_WITH_PASSENGER"
    TEMPORARY_PAUSE = "TEMPORARY_PAUSE"
    ALIGHTING_THRESHOLD = "ALIGHTING_THRESHOLD"
    REPOSITIONING = "REPOSITIONING"
    REPAIR_OR_DELAY = "REPAIR_OR_DELAY"


THRESHOLD_STATES = {
    InformalTransportState.BOARDING_THRESHOLD.value,
    InformalTransportState.ALIGHTING_THRESHOLD.value,
}


OCCUPIED_STATES = {
    InformalTransportState.MOVING_WITH_PASSENGER.value,
    InformalTransportState.TEMPORARY_PAUSE.value,
    InformalTransportState.ALIGHTING_THRESHOLD.value,
}


ANCHOR_STATES = {
    InformalTransportState.IDLE_AT_ANCHOR.value,
}


def is_threshold_state(state: str) -> bool:
    """
    Return True if the state creates a boarding/alighting threshold.
    """

    return state in THRESHOLD_STATES


def is_occupied_state(state: str) -> bool:
    """
    Return True if the agent is carrying a passenger.
    """

    return state in OCCUPIED_STATES


def is_anchor_state(state: str) -> bool:
    """
    Return True if the agent is waiting at an anchor.
    """

    return state in ANCHOR_STATES
