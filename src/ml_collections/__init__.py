"""Minimal ConfigDict compatibility for PI0.5-only PPU environments.

The full ``ml-collections`` dependency is only required when constructing a
PI0-FAST model. OpenPI imports the FAST config while building its global config
registry, so PI0.5 training only needs basic dictionary attribute access.
"""


class ConfigDict(dict):
    def __init__(self, initial_dictionary=None, **kwargs):
        super().__init__(initial_dictionary or {}, **kwargs)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value
