# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Dataclass tools for data namespaces (bins)
"""
from __future__ import annotations

from typing import Any, Iterable


class DataBinMeta(type):
    """This metaclass no longer has a purpose and will be removed."""


class DataBin(metaclass=DataBinMeta):
    """Namespace for storing data.

    .. code-block:: python

        data = DataBin(
            alpha=BitArray.from_bitstrings(["0010"]),
            beta=np.array([1.2])
        )

        print("alpha data:", data.alpha)
        print("beta data:", data.beta)

    """

    _RESTRICTED_NAMES = ("_RESTRICTED_NAMES", "_SHAPE", "_FIELDS", "_FIELD_TYPES")

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __setattr__(self, key, value):
        raise NotImplementedError

    def __len__(self):
        return len(self.__dict__)

    def __repr__(self):
        vals = (f"{name}={getattr(self, name)}" for name in self._FIELDS if hasattr(self, name))
        return f"{type(self).__name__}({', '.join(vals)})"

    # The following properties exist to provide support to legacy private class attributes which
    # gained widespread usage in several internal projects due to the lack of alternatives prior to
    # qiskit 1.1. These properties will be removed once the internal projects have made the
    # appropriate changes.

    @property
    def _FIELDS(self) -> tuple[str, ...]:  # pylint: disable=invalid-name
        return tuple(self.__dict__)

    @property
    def _FIELD_TYPES(self) -> tuple[Any, ...]:  # pylint: disable=invalid-name
        return tuple(type(val) for val in self.__dict__.values())

    @property
    def _SHAPE(self) -> tuple[int, ...] | None:  # pylint: disable=invalid-name
        return None


# pylint: disable=unused-argument
def make_data_bin(
    fields: Iterable[tuple[str, type]], shape: tuple[int, ...] | None = None
) -> DataBinMeta:
    """Return the :class:`~DataBin` type.

    .. note::
        This class used to return a subclass of :class:`~DataBin`. However, that caused confusion
        and didn't have a useful purpose. Several internal projects made use of this internal
        function prior to qiskit 1.1. This function will be removed once these internal projects
        have made the appropriate changes.

    Args:
        fields: Tuples ``(name, type)`` specifying the attributes of the returned class.
        shape: The intended shape of every attribute of this class.

    Returns:
        The :class:`DataBin` type.
    """
    return DataBin
