# get dependencies

```bash
uv sync
```

# launch evaluation

```bash
uv run python
```

```python
from src_rag import evaluate

evaluate.run_evaluate_retrieval(config={"model": {"chunk_size": 128,"overlap": 12}})
```