# Candilyzer

Candilyzer is an evidence-first technical candidate evaluation application. It collects approved candidate sources, compresses them into grounded facts, evaluates each candidate against explicit role requirements, and produces an auditable ranking for human review.

The application does not make hiring decisions. Protected characteristics must not be inferred or used, and every assessment should be checked against the cited source evidence.

## Product capabilities

- GitHub, LinkedIn search, PDF, and DOCX evidence inputs
- Deterministic candidate/source pairing by row and upload order
- Isolated LLM sessions per candidate to prevent cross-candidate context leakage
- Structured requirement-level output with evidence, match status, and importance
- Transparent scoring based on requirement coverage, confidence, and red flags
- Evidence-focused interview question generation
- Responsive Streamlit interface with a labeled, API-free sample analysis
- Bounded input sizes, provider retries, and actionable partial-failure reporting

## Architecture

```text
Approved sources
  GitHub API | Exa search | Resume parser
                  |
                  v
        Bounded evidence collection
                  |
                  v
        Injection-resistant compression
                  |
                  v
     Isolated structured evaluation sessions
                  |
                  v
  Deterministic scoring + responsive review UI
```

The pipeline keeps preprocessing, model execution, validation, scoring, and presentation separate:

```text
app.py
src/
  candidates.py          candidate/source validation and pairing
  fetcher.py             external evidence collection
  resume_parser.py       bounded PDF and DOCX extraction
  compressor.py          grounded evidence compression
  evaluator.py           structured output parsing and validation
  evaluator_factory.py   evaluation-agent construction
  pipeline.py            isolated concurrent evaluation
  ranking.py             deterministic scoring and ranking
  question_generator.py  follow-up interview questions
  schemas.py             shared typed contracts
  llm.py                 retry policy for provider calls
  ui.py                  responsive review interface
tests/                    deterministic unit tests
```

## Local setup

Requirements: Python 3.11 or newer.

```bash
git clone https://github.com/ssamadjon3106/PROJECTS.git
cd PROJECTS/Candilyzer
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create `.env` in `Candilyzer/`:

```dotenv
GEMINI_API_KEY=required_for_live_analysis
GITHUB_API_KEY=optional_higher_github_rate_limit
EXA_API_KEY=optional_linkedin_search
```

Never commit `.env` or provider credentials.

Run the application:

```bash
streamlit run app.py
```

The **View sample analysis** action exercises the full review interface without sending data to external providers.

## Input contract and limits

- Add one GitHub username and/or LinkedIn URL per candidate row.
- Resumes are paired with candidates by upload order.
- At least one source is required for each candidate.
- Up to 10 candidates may be evaluated in one run.
- Resume formats: PDF and DOCX only.
- Maximum resume size: 10 MB.
- Maximum PDF length: 40 pages.
- Maximum extracted resume text: 100,000 characters.
- Source text is bounded before it enters an LLM context.

LinkedIn collection is search-based and marked unavailable when `EXA_API_KEY` is not configured. A GitHub token is optional for public profiles but avoids the smallest unauthenticated rate limit.

## Scoring

Candilyzer calculates a deterministic evidence score after the model has returned validated requirement-level findings:

```text
requirement weight: Must Have = 2, Nice to Have = 1
match value:        Met = 1, Partial = 0.5, Not Met = 0
coverage:           sum(match value * weight) / sum(weight) * 100
confidence:         High = 1.0, Medium = 0.9, Low = 0.8
red-flag penalty:   5 points each, capped at 15
final score:        clamp(coverage * confidence - penalty, 0, 100)
```

Ties are resolved by evidence coverage and then candidate label. For `C` candidates and `R` evaluated requirements per candidate, scoring is `O(C * R)` time. Ranking is `O(C log C)` time. Evaluation result storage is `O(C * R)`.

## Reliability and security controls

- Evidence is explicitly delimited as untrusted data in model prompts.
- Source content cannot override system or evaluation instructions.
- LLM output is accepted only after strict JSON, enum, and evidence validation.
- Provider calls use bounded exponential retry for transient failures.
- Candidate evaluations use separate agent instances.
- UI-rendered provider and model content is HTML-escaped.
- File size, page count, candidate count, and extracted text are bounded.
- Partial failures remain visible instead of silently changing the ranking.

Production operators should also configure provider-side spend limits, centralized secrets management, audit logging, deletion/retention policies, and access controls appropriate to their jurisdiction.

## Verification

```bash
python -m unittest discover -s tests -v
python -m compileall -q app.py src tests
```

The unit suite covers candidate pairing and validation, structured-output rejection, evidence requirements, retry bounds, and deterministic ranking.

## Deployment

The Streamlit app entry point is `Candilyzer/app.py`. Configure secrets in the deployment environment rather than committing them to the repository.

Live application: [candilyzer-sss.streamlit.app](https://candilyzer-sss.streamlit.app)

## License

MIT
