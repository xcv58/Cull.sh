from __future__ import annotations

from typing import Protocol

from rich.console import Console
from rich.progress import BarColumn
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TaskProgressColumn
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn


class PipelineReporter(Protocol):
    def message(self, text: str) -> None:
        ...

    def start_phase(self, key: str, description: str, total: int) -> None:
        ...

    def advance_phase(self, key: str, advance: int = 1, description: str | None = None) -> None:
        ...

    def complete_phase(self, key: str, description: str | None = None, total: int | None = None) -> None:
        ...


class NullReporter:
    def message(self, text: str) -> None:
        return None

    def start_phase(self, key: str, description: str, total: int) -> None:
        return None

    def advance_phase(self, key: str, advance: int = 1, description: str | None = None) -> None:
        return None

    def complete_phase(self, key: str, description: str | None = None, total: int | None = None) -> None:
        return None


class RichPipelineReporter:
    def __init__(self, console: Console) -> None:
        self.console = console
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        )
        self.task_ids: dict[str, int] = {}

    def __enter__(self) -> "RichPipelineReporter":
        self.progress.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.progress.stop()

    def message(self, text: str) -> None:
        self.progress.console.print(text)

    def start_phase(self, key: str, description: str, total: int) -> None:
        if key in self.task_ids:
            self.progress.update(self.task_ids[key], description=description, total=total, completed=0)
            return
        self.task_ids[key] = self.progress.add_task(description, total=total)

    def advance_phase(self, key: str, advance: int = 1, description: str | None = None) -> None:
        task_id = self.task_ids[key]
        kwargs: dict[str, object] = {"advance": advance}
        if description is not None:
            kwargs["description"] = description
        self.progress.update(task_id, **kwargs)

    def complete_phase(self, key: str, description: str | None = None, total: int | None = None) -> None:
        task_id = self.task_ids[key]
        task = next(task for task in self.progress.tasks if task.id == task_id)
        completed = total if total is not None else int(task.completed)
        kwargs: dict[str, object] = {"completed": completed}
        if total is not None:
            kwargs["total"] = total
        elif task.total is not None:
            kwargs["completed"] = task.total
        if description is not None:
            kwargs["description"] = description
        self.progress.update(task_id, **kwargs)
