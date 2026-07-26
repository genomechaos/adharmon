# adharmon — multi-cohort AD metadata harmonization on Azure

[![CI](https://github.com/genomechaos/adharmon/actions/workflows/ci.yml/badge.svg)](https://github.com/genomechaos/adharmon/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Ingests sample metadata from several Alzheimer's disease transcriptomics studies
whose annotation conventions disagree with each other, maps them onto one
canonical schema, validates quality at the ingestion boundary, and lands the
result in an ADLS Gen2 medallion layout with a per-run lineage manifest.

The interesting problem here is not the transport, it is that three cohorts
describe the same biology in mutually incompatible ways:

| Field | GSE5281 | GSE33000 | GSE125583 |
|---|---|---|---|
| sex | `M` / `F` / `male` | `1` / `2` | `male` / `female` |
| age | `79` | `90+`, `77 years` | `94` (needs censoring) |
| diagnosis | `Alzheimer's Disease`, `normal` | `AD`, `non-demented` | `Alzheimer disease`, `probable AD` |
| Braak | *not reported* | `VI`, `III-IV` | `Braak V`, `stage 4` |
| CERAD | *not reported* | *not reported* | `frequent`, `sparse` |
| APOE | `34` | *not reported* | `43` |

Naively concatenating these produces a dataset where `34` and `43` are two
different APOE genotypes, a wall of `90.0` ages looks like real data, and three
spellings of Alzheimer's split one cohort into three.

## Architecture

```
GEO series matrices
        │
        ▼
  ┌───────────┐   Azure Data Factory Copy activity (scheduled trigger)
  │  bronze   │   raw, immutable, exactly as delivered
  └───────────┘
        │        adharmon: crosswalk → normalize → validate
        ▼
  ┌───────────┐        ┌──────────────┐
  │  silver   │        │  quarantine  │  rejected rows + machine-readable reason
  │ canonical │        └──────────────┘
  │ + flags   │
  └───────────┘
        │
        ▼
  run manifest (JSON): source paths, mapping audit, rule counts, row counts
```

### Reject vs. flag

The split is the design decision worth defending in review:

- **Reject** — violates the schema contract (missing required field, duplicate
  `sample_id`, Braak stage of 9). Diverted to quarantine *with a reason*, never
  silently dropped, so an upstream partner can be sent a concrete defect list.
- **Flag** — schema-valid but biologically suspicious: an AD diagnosis with
  Braak 0, a nominal control at Braak VI, RIN below 5. These stay in silver with
  a `quality_flags` column, because excluding them is a scientific judgement
  that belongs to the analyst, not the pipeline.

Rejecting on suspicion discards real signal — clinical/neuropathological
discordance is a finding, not an error. Passing everything silently ships
artifacts into model training. Hence two severities.

### Design notes

- **`dtype=str` on every read.** Type inference on raw source metadata is
  actively harmful: pandas reads APOE `33` as `int64` and a `sample_id` of
  `0012` as `12`. Parsing belongs to the normalizers, which expect strings.
- **Crosswalks are data, not code** (`mappings.py`). Adding a fourth cohort is a
  config entry plus a test, not a new branch in the transform.
- **Mapping audit before transform.** A partner renaming a column is the most
  common way a working pipeline breaks, and it is silent — the transform still
  runs, the column is just null forever after. `audit_mapping` makes it visible
  and alertable.
- **HIPAA Safe Harbor enforced in-pipeline.** Ages over 90 are capped *and*
  flagged via `age_censored`, whether the study already censored them or
  reported a true 94. Downstream survival analysis can then exclude censored
  ages instead of treating a spike at exactly 90.0 as real.
- **Nullable extension dtypes throughout**, so "missing" survives a Parquet
  round trip instead of becoming `NaN`-as-float or the string `"nan"`.

## Quick start

```bash
pip install -e ".[dev]"
python scripts/make_sample_data.py --root ./data
adharmon run --root ./data \
  --study GSE5281=GSE5281_metadata.csv \
  --study GSE33000=GSE33000_metadata.csv \
  --study GSE125583=GSE125583_metadata.csv
pytest
```

Against Azure, swap the backend — nothing else changes:

```bash
az login
adharmon run --account-url https://<account>.dfs.core.windows.net \
  --study GSE5281=GSE5281_metadata.csv
```

The CLI exits non-zero when the validation pass rate falls below
`--min-pass-rate` (default 0.95). That is what lets an ADF activity fail loudly
and trigger an Azure Monitor alert rather than quietly publishing a bad batch.

## Azure setup

Roughly an afternoon in the portal. Costs a few dollars — new accounts get $200
of credit for 30 days, and ADF orchestration is about $1 per 1,000 activity runs.
Set a budget alert anyway.

1. **Storage** — create an ADLS Gen2 storage account with **hierarchical
   namespace enabled at creation** (it cannot be cleanly toggled later; this is
   what makes it a data lake rather than plain object storage). Add containers
   `bronze`, `silver`, `quarantine`.
2. **Data Factory** — a pipeline with a Copy activity pulling source files into
   `bronze`, plus a schedule trigger. Then a Custom activity or Azure Function
   invoking `adharmon run`.
3. **Identity** — give the Data Factory a **system-assigned managed identity**
   and grant it **Storage Blob Data Contributor** on the storage account. No
   account keys, no connection strings, no SAS tokens in config.

   > Being **Owner** of a storage account grants *no* data-plane access. Owner
   > is control plane (manage the account); reading blobs needs a
   > `Storage Blob Data *` role. This is the most common cause of a 403 here.

4. **Key Vault** — store any remaining secrets and reference them from ADF
   linked services rather than inlining them.
5. **Monitoring** — an Azure Monitor alert rule on pipeline-run failure, wired
   to an action group that emails you.
6. **Least privilege for consumers** — grant researchers
   **Storage Blob Data Reader** on `silver` only. Quarantine and bronze stay
   restricted: they hold unvalidated data and, in a real deployment, would be
   the layers under a data use agreement.

## Layout

```
src/adharmon/
  schema.py       canonical columns, dtypes, controlled vocabularies, bounds
  normalizers.py  field-level parsers (age, sex, diagnosis, Braak, CERAD, APOE)
  mappings.py     per-study crosswalks — declarative, testable
  harmonize.py    apply a crosswalk; audit it against the real frame first
  validate.py     rule registry, reject/flag severities, quarantine report
  storage.py      LakeStore protocol; local + ADLS Gen2 backends
  pipeline.py     orchestration and the run manifest
  cli.py          entrypoint for local runs and ADF activities
tests/            unit + end-to-end, no Azure credentials required
scripts/          synthetic bronze-layer data generator
```

## Scope and caveats

- Sample **metadata** only. Expression matrices are out of scope; the silver
  table is the join key for them.
- Column names in `mappings.py` are **illustrative**. GEO encodes metadata in
  free-form `characteristics_ch1` fields whose labels differ per submission —
  confirm against the real series matrix before trusting a mapping.
  `audit_mapping` will tell you what it could not find.
- Sample data from `scripts/make_sample_data.py` is synthetic. Structurally
  faithful, biologically meaningless.
- The `Diagnosis` vocabulary is deliberately coarse. Source cohorts use
  incompatible diagnostic criteria (clinical vs. neuropathological, CERAD vs.
  NIA-Reagan), so five buckets is the widest grouping that stays defensible.
  Finer source labels remain in bronze.

## License

MIT.
