"""Data lake I/O for the bronze / silver / quarantine layers.

Two interchangeable backends behind one :class:`LakeStore` protocol:

:class:`LocalLakeStore`
    Plain directories. Used by the test suite and for local development, so
    nothing in CI needs an Azure subscription or network access.

:class:`AzureLakeStore`
    ADLS Gen2 via ``DefaultAzureCredential``. **No account keys, no connection
    strings, no SAS tokens in config.** In Azure the credential resolves to the
    Data Factory or Function app's managed identity; locally it falls back to
    the developer's ``az login`` session. The identity needs the
    *Storage Blob Data Contributor* role -- note that being *Owner* of the
    storage account grants no data-plane access on its own, which is the single
    most common cause of a 403 here.

The Azure SDK is imported lazily inside :class:`AzureLakeStore` so that the
package, and the whole test suite, import cleanly without ``azure-*`` installed.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd

__all__ = ["LakeStore", "LocalLakeStore", "AzureLakeStore", "Layer"]

logger = logging.getLogger(__name__)


class Layer:
    """Medallion layer names, matching the container names in Azure."""

    BRONZE = "bronze"
    SILVER = "silver"
    QUARANTINE = "quarantine"


@runtime_checkable
class LakeStore(Protocol):
    """Minimal read/write surface the pipeline depends on."""

    def read_csv(self, layer: str, path: str) -> pd.DataFrame:
        """Read a delimited file from ``layer``/``path`` as all-string columns."""
        ...

    def write_parquet(self, frame: pd.DataFrame, layer: str, path: str) -> str:
        """Write ``frame`` to ``layer``/``path``, returning the resolved location."""
        ...

    def list_files(self, layer: str, prefix: str = "") -> list[str]:
        """List file paths within ``layer``, optionally filtered by ``prefix``."""
        ...


class LocalLakeStore:
    """Filesystem-backed store mirroring the container/path layout of ADLS.

    Args:
        root: Directory containing one subdirectory per layer.
    """

    def __init__(self, root: str | Path) -> None:
        """Point the store at ``root``, which holds one subdirectory per layer."""
        self.root = Path(root)

    def _resolve(self, layer: str, path: str) -> Path:
        return self.root / layer / path

    def read_csv(self, layer: str, path: str) -> pd.DataFrame:
        """Read a CSV, keeping every column as string.

        ``dtype=str`` is deliberate: type inference on raw source metadata is
        actively harmful. Pandas will read a Braak column of ``1,2,3,III`` as
        object, an APOE column of ``33,34`` as int64, and a sample_id column of
        ``0012`` as int -- silently dropping the leading zero. Parsing is the
        normalizers' job, and they expect strings.
        """
        target = self._resolve(layer, path)
        return pd.read_csv(target, dtype=str, keep_default_na=False, na_values=[""])

    def write_parquet(self, frame: pd.DataFrame, layer: str, path: str) -> str:
        """Write ``frame`` as Parquet, creating parent directories as needed."""
        target = self._resolve(layer, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(target, index=False)
        logger.info("Wrote %d rows to %s", len(frame), target)
        return str(target)

    def list_files(self, layer: str, prefix: str = "") -> list[str]:
        """List layer-relative file paths, or ``[]`` if the layer does not exist."""
        base = self.root / layer
        if not base.exists():
            return []
        relative = (str(p.relative_to(base)) for p in base.rglob("*") if p.is_file())
        return sorted(path for path in relative if path.startswith(prefix))


class AzureLakeStore:
    """ADLS Gen2 store authenticated with ``DefaultAzureCredential``.

    Args:
        account_url: e.g. ``https://<account>.dfs.core.windows.net``.
        credential: Optional override, mainly for tests. Defaults to
            ``DefaultAzureCredential()``.

    Raises:
        ImportError: If ``azure-storage-file-datalake`` and ``azure-identity``
            are not installed. Install with the ``azure`` extra.
    """

    def __init__(self, account_url: str, credential: object | None = None) -> None:
        """Connect to an ADLS Gen2 account. See the class docstring for RBAC notes."""
        try:
            from azure.identity import DefaultAzureCredential
            from azure.storage.filedatalake import DataLakeServiceClient
        except ImportError as exc:  # pragma: no cover - exercised only without extras
            raise ImportError("AzureLakeStore requires the 'azure' extra: pip install -e '.[azure]'") from exc

        self.account_url = account_url
        self._credential = credential or DefaultAzureCredential()
        self._service = DataLakeServiceClient(account_url=account_url, credential=self._credential)

    def _file_client(self, layer: str, path: str):  # noqa: ANN202 - SDK type is lazily imported
        return self._service.get_file_system_client(layer).get_file_client(path)

    def read_csv(self, layer: str, path: str) -> pd.DataFrame:
        """Download a CSV from ``layer``/``path``. See :meth:`LocalLakeStore.read_csv`."""
        stream = self._file_client(layer, path).download_file()
        buffer = io.BytesIO(stream.readall())
        return pd.read_csv(buffer, dtype=str, keep_default_na=False, na_values=[""])

    def write_parquet(self, frame: pd.DataFrame, layer: str, path: str) -> str:
        """Upload ``frame`` as Parquet, overwriting any existing file."""
        buffer = io.BytesIO()
        frame.to_parquet(buffer, index=False)
        payload = buffer.getvalue()
        self._file_client(layer, path).upload_data(payload, overwrite=True)
        logger.info("Wrote %d rows (%d bytes) to %s/%s", len(frame), len(payload), layer, path)
        return f"{self.account_url}/{layer}/{path}"

    def list_files(self, layer: str, prefix: str = "") -> list[str]:
        """List file paths in a container, recursively, excluding directories."""
        system = self._service.get_file_system_client(layer)
        paths = system.get_paths(path=prefix or None, recursive=True)
        return sorted(item.name for item in paths if not item.is_directory)
