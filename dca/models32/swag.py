from ..swag import SWAG
from .standard import preresnet20


__all__ = ('preresnet20_swag',)


def preresnet20_swag(outclass: int, max_rank: int = 20) -> SWAG:
    return SWAG(preresnet20(outclass), max_rank)
