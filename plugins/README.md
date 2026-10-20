# Plugins

Put tool files here. Every `*.py` file in this folder is loaded at startup, and its `@tool` functions are registered. Files starting with `_` are skipped.

```python
from companion.tools.registry import tool

@tool
def plugin_hello() -> str:
    """Say hello from a plugin."""
    return "hello"
```

A plugin that fails to load is skipped with a warning. See the main README for details.
