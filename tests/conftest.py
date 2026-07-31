"""Keep the suite offline-capable and fast.

Importing foodsafe.llm probes Gemini to decide which provider can actually
answer. That is right at runtime and wrong in tests: it adds a network round
trip to every run and makes results depend on an account's credit balance.
"""

import os

os.environ.setdefault("FOODSAFE_SKIP_PROBE", "1")
