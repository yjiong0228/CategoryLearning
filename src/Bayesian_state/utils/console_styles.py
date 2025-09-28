import secrets
import json
import hashlib
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def gen_rand_str(n: int = 32):
    """
    Generate random string.

    Default length 32. Only in [A-Za-z0-9].
    """
    return secrets.token_urlsafe(n)


def compose_print(l):
    match l:
        case int(a):
            return "".join(f"\033[{x}m" for x in [a])
        case list():
            return "".join(f"\033[{x}m" for x in l)


SINGLE_STYLES = ([1, 3, 4, 5, 7, 21, 53] + list(range(30, 37)) +
                 list(range(40, 48)))
COMPOSITE_STYLES = [[1, 3, 4, x]
                    for x in range(40, 47)] + [[1, 4, x]
                                               for x in range(40, 47)]

PRINT_STYLES = COMPOSITE_STYLES + SINGLE_STYLES

PRINT_RESUME = compose_print(0)

dummy_print = print


def print(*args, **kwargs):
    style = kwargs.get("s", None)
    if style is None:
        dummy_print(*args, **kwargs)
    else:
        end = kwargs.pop("end", "\n")
        kwargs.pop("s")
        dummy_print(compose_print(PRINT_STYLES[style]), end="")
        dummy_print(*args, **kwargs, end="")
        dummy_print(PRINT_RESUME, end=end)
