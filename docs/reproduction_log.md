# Reproduction log

One entry per reproduction attempt (fresh clone on the second laptop, Docker container, public-repo clone).

## Template

```
### YYYY-MM-DD — <native | docker> on <hostname>
- Commit / tag:
- Command: uv run python scripts/run_all.py --level 1
- Environment: uv.lock hash / Docker image digest; CPU; OS
- Wall-clock:
- Comparison: uv run python scripts/compare_results.py --reference <dir> --candidate <dir>
- Result: all files within tolerance | mismatches listed below
- Mismatches and explanation:
```

## Entries

(none yet)
