from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from research_agents.config import settings


@dataclass
class SandboxResult:
    stdout: str = ""
    stderr: str = ""
    error: bool = False


class SandboxSession:
    """Context manager wrapping E2B sandbox with convenience methods."""

    def __enter__(self) -> "SandboxSession":
        if not settings.e2b_api_key:
            raise RuntimeError("E2B_API_KEY is required for sandbox execution.")
        try:
            from e2b_code_interpreter import Sandbox
        except ImportError as exc:
            raise RuntimeError(
                "e2b-code-interpreter is not installed. Install requirements.txt first."
            ) from exc

        self.sandbox = Sandbox(api_key=settings.e2b_api_key)
        self.sandbox.notebook.exec_cell(
            "pip install -q scikit-learn pandas numpy matplotlib seaborn torch datasets"
        )
        return self

    def run_code(self, code: str) -> Any:
        return self.sandbox.notebook.exec_cell(code)

    def upload_data(self, data_output) -> None:
        for dataset in data_output.datasets:
            local_path = Path(dataset.local_path)
            if local_path.exists():
                self.sandbox.files.write(
                    path=f"/home/user/data/processed/{local_path.name}",
                    data=local_path.read_bytes(),
                )

    def list_artifacts(self) -> list[str]:
        try:
            files = self.sandbox.files.list("/home/user/outputs/")
        except Exception:
            return []
        return [getattr(file, "name", str(file)) for file in files]

    def __exit__(self, *args) -> None:
        self.sandbox.close()

