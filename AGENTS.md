# Overview

This repository assumes zero assumptions by any agent. All agents must:

Activate the Python virtual environment before running code.

Trace upstream to understand the full execution path.

Never guess behavior from isolated snippets. Always read more code.

Agents that operate on this repo must behave as if they are pair programming with a rigorous senior engineer: skeptical, exacting, and unwilling to trust abstractions without inspection.

# 🧠 Behavioral Protocol
❗ Do Not Assume. Ever.

Any code question must be answered only after:

Tracing the function or symbol upstream to its origin.

Reviewing the implementation—not just the interface.

Reading all relevant control flow (imports, decorators, monkey patches, etc).

Identifying dynamic behavior: factory patterns, metaclass usage, or runtime injection.

If a file references other modules, those must be loaded and read before any comment or suggestion is made.

# 🧩 Never Isolate Analysis

Understanding must be contextual:

If a function is called from multiple locations, identify them.

If behavior changes depending on environment flags, configs, or request context, highlight them.

If there is a decorator, middleware, signal, listener, or any indirection: inspect it.