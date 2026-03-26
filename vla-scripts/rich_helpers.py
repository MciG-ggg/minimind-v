"""
Rich 终端输出辅助函数，供所有脚本复用
"""

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

console = Console(force_terminal=True)


def ok(msg: str) -> None:
    console.print(f"[green]✓[/green] {msg}")


def info(msg: str) -> None:
    console.print(f"[dim]{msg}[/dim]")


def warn(msg: str) -> None:
    console.print(f"[yellow]⚠[/yellow] {msg}")


def err(msg: str) -> None:
    console.print(f"[bold red]✗[/bold] {msg}")


def fatal(msg: str) -> None:
    err(msg)
    raise SystemExit(1)


def section(title: str, style: str = "cyan") -> None:
    console.print(Rule(f"[bold {style}]{title}[/bold {style}]", style=style))


def kv_table(title: str, rows: list[tuple[str, str]]) -> None:
    """键值对表格"""
    tbl = Table(title=title, box=None, show_header=False, padding=(0, 2))
    tbl.add_column(style="yellow bold")
    tbl.add_column()
    for k, v in rows:
        tbl.add_row(k, v)
    console.print(tbl)


def cols_table(columns: list[str], title: str = "列名") -> None:
    """单列表格（字段 / 列名列表）"""
    tbl = Table(title=title, box=None)
    tbl.add_column("字段", style="yellow")
    for c in columns:
        tbl.add_row(c)
    console.print(tbl)


def sample_table(title: str, rows: list[tuple[str, str]], style: str = "cyan") -> None:
    """键值对样本详情表格"""
    tbl = Table(title=title, show_header=False, box=None, padding=(0, 2))
    tbl.add_column(style="bold")
    tbl.add_column()
    for k, v in rows:
        tbl.add_row(f"[{style}]{k}[/{style}]", v)
    console.print(tbl)


def progress_panel(title: str, lines: list[tuple[str, str]], border: str = "cyan") -> None:
    """进度摘要 Panel"""
    body = "\n".join(f"[yellow]{k}:[/yellow] {v}" for k, v in lines)
    console.print(Panel.fit(body, title=title, border_style=border))
