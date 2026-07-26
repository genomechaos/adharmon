"""End-to-end tests over the local lake backend.

No Azure credentials, no network. :class:`~adharmon.storage.LocalLakeStore` and
:class:`~adharmon.storage.AzureLakeStore` satisfy the same protocol, so these
tests exercise the real orchestration path.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from adharmon.cli import main
from adharmon.pipeline import run_all, run_study
from adharmon.schema import CANONICAL_COLUMNS
from adharmon.storage import Layer, LocalLakeStore


class TestRunStudy:
    def test_reads_and_harmonizes(self, local_lake: LocalLakeStore) -> None:
        frame, summary = run_study(local_lake, "GSE5281", "GSE5281_metadata.csv")
        assert len(frame) == 3
        assert tuple(frame.columns) == CANONICAL_COLUMNS
        assert "GSE5281" in summary

    def test_leading_zeros_preserved(self, tmp_path) -> None:
        """dtype=str on read is what stops sample_id '0012' becoming 12."""
        store = LocalLakeStore(tmp_path)
        bronze = tmp_path / "bronze"
        bronze.mkdir(parents=True)
        pd.DataFrame(
            {
                "geo_accession": ["0012"],
                "subject": ["S1"],
                "age": ["70"],
                "sex": ["M"],
                "disease_state": ["AD"],
                "region": ["cortex"],
                "apoe": ["33"],
            }
        ).to_csv(bronze / "z.csv", index=False)
        frame, _ = run_study(store, "GSE5281", "z.csv")
        assert frame.loc[0, "sample_id"] == "0012"


class TestRunAll:
    def test_writes_silver_and_manifest(self, local_lake: LocalLakeStore) -> None:
        sources = {
            "GSE5281": "GSE5281_metadata.csv",
            "GSE33000": "GSE33000_metadata.csv",
            "GSE125583": "GSE125583_metadata.csv",
        }
        manifest = run_all(local_lake, sources, run_id="testrun")

        assert manifest.rows_in == 9
        assert manifest.rows_clean == 9
        assert manifest.rows_quarantined == 0
        assert sorted(manifest.studies) == sorted(sources)
        assert manifest.silver_path is not None
        assert manifest.finished_at is not None

        written = pd.read_parquet(manifest.silver_path)
        assert len(written) == 9
        assert set(written["source_study"]) == set(sources)

    def test_manifest_records_lineage(self, local_lake: LocalLakeStore) -> None:
        manifest = run_all(local_lake, {"GSE33000": "GSE33000_metadata.csv"}, run_id="lineage")
        payload = json.loads(manifest.to_json())
        assert payload["source_paths"]["GSE33000"] == "GSE33000_metadata.csv"
        assert "Safe Harbor" in payload["mapping_notes"]["GSE33000"]
        assert payload["run_id"] == "lineage"

    def test_quarantine_written_only_when_needed(self, local_lake: LocalLakeStore, tmp_path) -> None:
        bronze = tmp_path / "bronze"
        pd.DataFrame(
            {
                "sample": ["X1", "X1"],  # duplicate -> both quarantined
                "donor_id": ["D1", "D1"],
                "age_at_death": ["70", "70"],
                "gender": ["1", "1"],
                "dx": ["AD", "AD"],
                "braak": ["III", "III"],
                "tissue": ["pfc", "pfc"],
                "pmi": ["5", "5"],
                "scan_batch": ["B", "B"],
            }
        ).to_csv(bronze / "dupes.csv", index=False)

        manifest = run_all(local_lake, {"GSE33000": "dupes.csv"}, run_id="dupes")
        assert manifest.rows_quarantined == 2
        assert manifest.quarantine_path is not None
        rejected = pd.read_parquet(manifest.quarantine_path)
        assert "duplicate_sample_id" in rejected.loc[0, "quarantine_reason"]

    def test_partitioned_output_path(self, local_lake: LocalLakeStore) -> None:
        """Hive-style run_id partitioning keeps runs immutable and queryable."""
        manifest = run_all(local_lake, {"GSE5281": "GSE5281_metadata.csv"}, run_id="abc123")
        assert "run_id=abc123" in manifest.silver_path


class TestStorage:
    def test_list_files(self, local_lake: LocalLakeStore) -> None:
        files = local_lake.list_files(Layer.BRONZE)
        assert "GSE5281_metadata.csv" in files

    def test_list_missing_layer_is_empty(self, tmp_path) -> None:
        assert LocalLakeStore(tmp_path).list_files("nonexistent") == []

    def test_azure_store_requires_extra(self, monkeypatch) -> None:
        """Without azure-* installed, the error names the fix."""
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name.startswith("azure"):
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        from adharmon.storage import AzureLakeStore

        with pytest.raises(ImportError, match=r"\[azure\]"):
            AzureLakeStore("https://example.dfs.core.windows.net")


class TestCli:
    def test_run_exits_zero_on_clean_batch(self, local_lake: LocalLakeStore, capsys) -> None:
        code = main(
            [
                "run",
                "--root",
                str(local_lake.root),
                "--study",
                "GSE5281=GSE5281_metadata.csv",
                "--run-id",
                "cli1",
            ]
        )
        assert code == 0
        assert "cli1" in capsys.readouterr().out

    def test_run_exits_nonzero_below_threshold(self, local_lake: LocalLakeStore, tmp_path) -> None:
        """A bad batch must fail the ADF activity so the alert fires."""
        pd.DataFrame(
            {
                "sample": ["Y1", "Y1"],
                "donor_id": ["D", "D"],
                "age_at_death": ["70", "70"],
                "gender": ["1", "1"],
                "dx": ["AD", "AD"],
                "braak": ["II", "II"],
                "tissue": ["pfc", "pfc"],
                "pmi": ["5", "5"],
                "scan_batch": ["B", "B"],
            }
        ).to_csv(tmp_path / "bronze" / "bad.csv", index=False)

        code = main(["run", "--root", str(local_lake.root), "--study", "GSE33000=bad.csv", "--run-id", "cli2"])
        assert code == 1

    def test_bad_study_argument_rejected(self, local_lake: LocalLakeStore) -> None:
        with pytest.raises(SystemExit):
            main(["run", "--root", str(local_lake.root), "--study", "no-equals-sign"])

    def test_root_and_account_url_mutually_exclusive(self) -> None:
        with pytest.raises(SystemExit):
            main(["run", "--root", "/tmp/x", "--account-url", "https://a.dfs.core.windows.net", "--study", "A=b.csv"])
