# Changelog

All notable changes to DealPilot are documented here.

## [1.1.0] - 2026-03-02

### Added
- Unit tests for all deterministic pipeline steps (Steps 1-5)
- GitHub Actions CI/CD workflow for automated testing and linting
- Dashboard screenshots in README
- CHANGELOG.md for version tracking

### Changed
- Migrated LLM backend from Claude/Anthropic to Groq API (Llama 4 Scout)
- Model updated to `meta-llama/llama-4-scout-17b-16e-instruct`
- Temperature set to 0.7 with 3-retry logic
- Replaced `use_container_width` with `width='stretch'` for Streamlit compatibility
- ASCII-safe comparison table for Windows subprocess support

### Removed
- All Anthropic/Claude SDK dependencies
- `anthropic` from requirements.txt

## [1.0.0] - 2026-03-01

### Added
- 8-step deterministic CRM pipeline
- Lead ranking with weighted composite scoring
- Churn prediction with NPS imputation
- Stalled deal detection with inactivity thresholds
- LLM-powered action generation (Step 6)
- Cross-signal confidence adjustments (Step 7)
- Pydantic-validated JSON output (Step 8)
- Streamlit dashboard with 5 pages
- Benchmark evaluation suite with scoring formula
- Synthetic CRM dataset generator (200 records)
- Claude baseline benchmark comparison
