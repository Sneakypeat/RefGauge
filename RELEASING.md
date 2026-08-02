# Release checklist

1. Confirm that `CHANGELOG.md`, `CITATION.cff`, and `refgauge.__version__` name
   the same release and that the working tree contains only intended changes.
2. Refresh and verify the lock file with `uv lock` and `uv lock --check`.
3. Run the complete local quality suite:

   ```bash
   python -m ruff check .
   python -m ruff format --check .
   python -m pytest --cov=refgauge --cov-branch --cov-report=term-missing
   ```

4. Build from a clean checkout and validate both distributions:

   ```bash
   python -m build
   python -m twine check --strict dist/*
   check-wheel-contents dist/*.whl
   ```

   The `dist/` directory must be absent or empty before this step. Never upload a
   wildcard from a directory that also contains artifacts from an older version.

5. Install the wheel in a new environment, run `python -m pip check`, print the
   version, and execute the demo. Inspect the wheel and sdist file lists to confirm
   that no raw human data, credentials, caches, or local paths are present.
6. Build twice with the same `SOURCE_DATE_EPOCH` and compare SHA-256 digests. Record
   final artifact digests alongside the release.
7. Only after CI succeeds, create a signed version tag matching `v<version>` and
   attach the checked wheel, sdist, and digest file to the release.

Publishing, tagging, and credentialed release actions are deliberately manual and
are not performed by the test workflow.
