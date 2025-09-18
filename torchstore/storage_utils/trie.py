# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A wrapper around pygtrie.Trie to provide a trie data structure based dictionary."""

from collections.abc import ItemsView, KeysView, Mapping, MutableMapping, ValuesView
from typing import Any, Iterator, List, Optional

import pygtrie


class TrieKeysView(KeysView[str]):
    """A custom KeysView implementation for Trie that supports prefix filtering."""

    def __init__(self, trie: pygtrie.Trie):
        self._trie = trie

    def __iter__(self) -> Iterator[str]:
        return iter(self._trie)

    def __len__(self) -> int:
        return len(self._trie)

    def __contains__(self, key: object) -> bool:
        return key in self._trie

    def filter_by_prefix(self, prefix: str) -> List[str]:
        """Return a list of keys that start with the given prefix."""
        try:
            return [str(key) for key in self._trie.iterkeys(prefix=prefix)]
        except KeyError:
            return []


class Trie(MutableMapping[str, Any]):
    """
    A wrapper around pygtrie.StringTrie that implements the MutableMapping interface
    and provides prefix-based key filtering with consistent return types.

    This can be used as a drop-in replacement for a standard dictionary, but with
    an additional ``filter_keys`` method that allows for prefix-based filtering of
    keys.

    Args:
        mapping: Optional Mapping to initialize the trie with

    Example:
        trie = StringTrie({"abc.xyz": 1, "abc": 2, "xyz": 3})
        for key in trie.keys().filter_by_prefix("abc"):
            print(key)  # Prints "abc.xyz" and "abc"
    """

    def __init__(
        self, mapping: Optional[Mapping[str, Any]] = None, *, separator: str = "."
    ) -> None:
        self._trie = pygtrie.StringTrie(separator=separator)

        if mapping:
            for key, value in mapping.items():
                self._trie[key] = value

    # Mapping interface methods
    def __getitem__(self, key: str) -> Any:
        """Get a value by key from the trie."""
        return self._trie[key]

    def __iter__(self) -> Iterator[str]:
        """Iterate over keys in the trie."""
        for key in self._trie:
            yield str(key)

    def __len__(self) -> int:
        """Return the number of items in the trie."""
        return len(self._trie)

    def __contains__(self, key: object) -> bool:
        """Check if a key exists in the trie."""
        return key in self._trie

    # MutableMapping interface methods
    def __setitem__(self, key: str, value: Any) -> None:
        """Set a key-value pair in the trie."""
        self._trie[key] = value

    def __delitem__(self, key: str) -> None:
        """Delete a key from the trie."""
        del self._trie[key]

    def keys(self) -> TrieKeysView:  # type: ignore[override]
        """
        Return a KeysView-like object that supports prefix filtering.

        Returns:
            StringTrieKeysView that supports both standard iteration and prefix filtering
        """
        return TrieKeysView(self._trie)

    def values(self) -> ValuesView[Any]:
        """Return a view of all values in the trie."""
        return self._trie.values()  # type: ignore[return-value]

    def items(self) -> ItemsView[str, Any]:
        """Return a view of all key-value pairs in the trie."""
        return self._trie.items()  # type: ignore[return-value]

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value by key, returning default if not found.

        Args:
            key: The key to look up
            default: Value to return if key is not found

        Returns:
            The value associated with the key, or default if not found
        """
        return self._trie.get(key, default)

    def clear(self) -> None:
        """Remove all items from the trie."""
        self._trie.clear()

    def pop(self, key: str, default: Any = ...) -> Any:  # type: ignore[misc]
        """
        Remove and return the value for a key, or default if not found.

        Args:
            key: The key to remove
            default: Value to return if key is not found

        Returns:
            The value that was removed, or default if key not found
        """
        if default is ...:
            return self._trie.pop(key)
        return self._trie.pop(key, default)

    def update(self, other: Mapping[str, Any], **kwargs: Any) -> None:  # type: ignore[override]
        """
        Update the trie with key-value pairs from another mapping.

        Args:
            other: Mapping to update from
            **kwargs: Additional key-value pairs
        """
        if isinstance(other, Trie):
            self._trie.update(other._trie)
        else:
            self._trie.update(other)
        if kwargs:
            self._trie.update(kwargs)

    def __bool__(self) -> bool:
        """Return True if the trie is not empty."""
        return bool(self._trie)

    def __repr__(self) -> str:
        """Return a string representation of the trie."""
        return f"Trie({dict(self._trie)})"

    def __eq__(self, other: object) -> bool:
        """Check equality with another Trie or mapping."""
        if isinstance(other, Trie):
            return dict(self._trie) == dict(other._trie)
        elif isinstance(other, Mapping):
            return dict(self._trie) == dict(other)
        return False
