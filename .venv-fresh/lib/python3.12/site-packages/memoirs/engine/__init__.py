"""Layers 2–5: extraction, embeddings, memory engine, graph.

Modules:
  - gemma.py          — extract_pending + Gemma cascade + heuristic fallback
  - extract_spacy.py  — spaCy POS/dep extractor (fallback when Gemma absent)
  - embeddings.py     — sentence-transformers + sqlite-vec ANN
  - memory_engine.py  — consolidation / scoring / lifecycle / reasoning / events
  - graph.py          — entities, relationships, project context
"""
from . import embeddings, gemma, graph, memory_engine

__all__ = ["embeddings", "gemma", "graph", "memory_engine"]
