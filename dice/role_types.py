from enum import IntEnum


class Role(IntEnum):
    SCOUT = 0
    EXECUTOR = 1
    RELAY = 2
    GUARDIAN = 3
    LOGISTICS = 4


def role_names(n: int = 3) -> list[str]:
    names = [Role.SCOUT.name, Role.EXECUTOR.name, Role.RELAY.name, Role.GUARDIAN.name, Role.LOGISTICS.name]
    return names[:n]


NUM_ROLES_DEFAULT = 3
