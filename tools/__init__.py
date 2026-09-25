"""Deterministic data science tools.

Agents never compute statistics themselves; they call these functions and reason
about the returned JSON. Keeping the computation here means every number in the
final report is reproducible and traceable to code rather than to an LLM.
"""
