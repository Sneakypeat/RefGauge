# Contributing

Please open an issue before changing a statistical default. Pull requests should:

1. include a test that fails before the change;
2. preserve the input and output contracts or document a versioned migration;
3. run `ruff check .` and `pytest`;
4. update the statistical-methods document when inference changes;
5. avoid adding raw human data or controlled-access metadata.

Bug reports should include the RefGauge version, Python version, command, minimal
input schema, and traceback. Never attach participant-level data to a public issue.

Before opening a pull request, run the same local gates as CI:

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m pytest --cov=refgauge --cov-branch --cov-report=term-missing
python -m build
python -m twine check --strict dist/*
check-wheel-contents dist/*.whl
```
