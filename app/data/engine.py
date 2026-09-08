"""Callable collection engine and command-line entry point."""

import argparse
from .config import DataConfig
from .providers import CollectionResult, ProviderStatus
from .yahoo import _default_downloader, collect_yahoo, persist_yahoo


def collect_market_data(config: DataConfig | None = None, *, start: str | None = None,
                        end: str | None = None, downloader=None) -> CollectionResult:
    config = config or DataConfig.from_env()
    config.ensure_paths()
    try:
        frame = collect_yahoo(config.tickers, start, end, downloader=downloader or _default_downloader)
        path = persist_yahoo(frame, config.processed_path / "yahoo_observations.csv")
        return CollectionResult(ProviderStatus.SUCCESS, "yahoo", len(frame), str(path))
    except (RuntimeError, OSError, ValueError) as exc:
        return CollectionResult(ProviderStatus.FAILED, "yahoo", error=str(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect normalized Kronos market observations")
    parser.add_argument("--start")
    parser.add_argument("--end")
    args = parser.parse_args()
    result = collect_market_data(start=args.start, end=args.end)
    print({"provider": result.provider, "status": result.status.value, "rows": result.rows, "path": result.path, "error": result.error})


if __name__ == "__main__":
    main()
