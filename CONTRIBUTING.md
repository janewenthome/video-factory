# Contributing

Thanks for helping improve Video Factory. Keep changes focused and preserve the project's privacy and media-processing boundaries.

## Before editing

- Read `README.md` and the relevant skill or module documentation.
- Keep original photos and videos immutable. Put generated files under a project's `work/` or `outputs/` directories.
- Never commit personal media, transcripts, caches, Colab credentials, API keys, or per-user project state.
- Do not add a cloud provider or paid model as an automatic fallback. Cloud transfers need an explicit per-run consent gate.
- Keep Colab work limited to approved perception inference. Proxy generation, general filesystem work, and final rendering remain local.

## Changes and verification

- Prefer deterministic code for routing, validation, hashing, serialization, retries, and cleanup.
- Update user-facing documentation when behavior or command interfaces change.
- Add behavior-focused unit tests for new safety or data-contract behavior.
- Before submitting, run the relevant tests. To run the full Python suite:

  ```bash
  python3 -m unittest discover -s tests -v
  ```

- State what you ran and report any live service or hardware checks that were not performed.

## Licensing

Contributed code is offered under the repository's MIT license. Third-party dependencies and downloaded media retain their own terms. In particular, the Remotion runtime is governed by Remotion's separate license; review it before commercial use or redistribution.
