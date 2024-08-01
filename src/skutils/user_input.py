# Standard library
from collections.abc import Collection
import logging
import select
import sys
import time
from typing import Any, Callable

# Globals
log = logging.getLogger(__name__)


################################################################################
def request_permission(prompt_prefix: str) -> bool:
    result = validated_input(
        prompt_prefix=prompt_prefix,
        valid_inputs=("y", "n"),
        transform=lambda x: x.lower() if x is not None else None,
        timeout=30,
        default="n",
    )
    return result == "y"


def validated_input(
    prompt_prefix: str,
    valid_inputs: Collection[object],
    transform: Callable[[str | None], Any] = lambda x: x,
    timeout: float | None = None,
    default: str | None = None,
) -> str | None:
    """Get input from user that validated to be an expected value.

    Parameters
    ==========
    prompt:
        Prompt prefix that will be appended with valid input values
    valid_inputs:
        Valid input values for user to provide
    transform:
        Transformation to apply to user input before checking if valid
    timeout & default:
        see input_with_timeout()
    """
    prompt = f'{prompt_prefix} [{",".join(map(str, valid_inputs))}] '
    result = None
    start = time.perf_counter()  # for timeout
    while result not in valid_inputs:
        if timeout is not None:
            timeout = max(0, timeout - (time.perf_counter() - start))
        result = input_with_timeout(prompt, timeout, default)
        if transform(result) not in valid_inputs:
            log.error("Unacceptable input, %r", result)

    return result


def input_with_timeout(
    prompt: str,
    timeout: float | None = None,
    default: str | None = None,
) -> str | None:
    """Get user input with option to timeout.

    Parameters
    ==========
    prompt:
        Prompt printed to standard output without a trailing newline before
        reading input.
    timeout:
        Time (sec) allowed for user to provide input after which a default value
        is returned. Currently, only supported on UNIX. None implies no timeout.
    default:
        The default value returned if the user does not provide input within the
        timeout limit.
    """
    if timeout is None:
        return input(prompt)

    # Code below was largly copied from:
    # https://stackoverflow.com/questions/15528939/time-limited-input
    # Write prompt
    sys.stdout.write(prompt)
    sys.stdout.flush()

    # Await user input
    # select.select() will block until either stdin is ready with user input or
    # the timeout has occurred. If a timeout occurs, all returned values will be
    # set to False.
    ready, _, _ = select.select([sys.stdin], [], [], timeout)

    result: str | None
    if ready:
        result = sys.stdin.readline().rstrip("\n")
    else:
        sys.stdout.write("\n")
        log.warning("User input not provided. Returning default: %s", default)
        result = default
    return result
