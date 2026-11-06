"""Argument validation helpers shared by the modules (internal)."""

from __future__ import annotations

import math
import sys

__all__ = ["check_int", "check_number"]


def check_int(
    name: str,
    value: object,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Checks that ``value`` is an ``int`` (not a ``bool``) within bounds.

    Args:
        name: The argument name for messages.
        value: The value to check.
        minimum: The smallest accepted value, if any.
        maximum: The largest accepted value, if any.

    Returns:
        The value.

    Raises:
        TypeError: If ``value`` is not an ``int``.
        ValueError: If it is out of bounds.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        low = "-inf" if minimum is None else str(minimum)
        high = "inf" if maximum is None else str(maximum)
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def check_number(
    name: str,
    value: object,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    """Checks that ``value`` is a finite real number within bounds.

    Args:
        name: The argument name for messages.
        value: The value to check.
        minimum: The lower bound, if any.
        maximum: The upper bound (inclusive), if any.
        exclusive_minimum: Whether ``minimum`` itself is refused.

    Returns:
        The value as a ``float``.

    Raises:
        TypeError: If ``value`` is not an ``int`` or ``float``.
        ValueError: If it is not finite or out of bounds, including an
            ``int`` too large for a ``float`` (never ``OverflowError``).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    try:
        number = float(value)
    except OverflowError:  # an int beyond the float range is beyond any float bound
        number = math.inf if value > 0 else -math.inf
    else:
        if not math.isfinite(number):
            raise ValueError(f"{name} must be finite")
    if minimum is not None and (number < minimum or (exclusive_minimum and number == minimum)):
        raise ValueError(
            f"{name} must be {'above' if exclusive_minimum else 'at least'} {minimum:g}"
        )
    if maximum is not None and number > maximum:
        raise ValueError(f"{name} must be at most {maximum:g}")
    if not math.isfinite(number):  # an overflowing int on a side without a bound
        bound = sys.float_info.max if number > 0 else -sys.float_info.max
        raise ValueError(f"{name} must be {'at most' if number > 0 else 'at least'} {bound:g}")
    return number
