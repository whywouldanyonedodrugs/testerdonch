from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from debug_cutoff import (
    main as debug_cutoff_main,
    parquet_max_timestamp,
    signals_max_timestamp,
    trades_max_timestamp,
)


def test_parquet_max_timestamp(tmp_path):
    pq_dir = tmp_path / "parquet"
    pq_dir.mkdir()
    df = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=3, tz="UTC")})
    df.to_parquet(pq_dir / "BTC.parquet", index=False)

    result = parquet_max_timestamp(str(pq_dir))
    assert not result.empty
    assert list(result["symbol"]) == ["BTC"]
    assert result["ts_min"].iloc[0] == pd.Timestamp("2025-01-01", tz="UTC")
    assert result["ts_max"].iloc[0] == pd.Timestamp("2025-01-03", tz="UTC")
    assert result["n"].iloc[0] == 3


def test_signals_max_timestamp(tmp_path):
    sig_dir = tmp_path / "signals"
    part_dir = sig_dir / "symbol=ETH"
    part_dir.mkdir(parents=True)
    df = pd.DataFrame({"timestamp": pd.to_datetime([
        "2025-02-01T00:00:00Z", "2025-02-03T00:00:00Z"
    ])})
    df.to_parquet(part_dir / "part-0000.parquet", index=False)

    mn, mx, count = signals_max_timestamp(str(sig_dir))
    assert mn == pd.Timestamp("2025-02-01", tz="UTC")
    assert mx == pd.Timestamp("2025-02-03", tz="UTC")
    assert count == 1


def test_trades_max_timestamp(tmp_path):
    trades_path = tmp_path / "trades.csv"
    df = pd.DataFrame({
        "entry_ts": pd.to_datetime([
            "2025-01-01T00:00:00Z",
            "2025-01-02T00:00:00Z"
        ]),
        "exit_ts": pd.to_datetime([
            "2025-01-01T05:00:00Z",
            "2025-01-03T07:00:00Z"
        ]),
        "pnl_R": [1.2, -0.5]
    })
    df.to_csv(trades_path, index=False)

    emin, emax, xmax, n = trades_max_timestamp(str(trades_path))
    assert emin == pd.Timestamp("2025-01-01T00:00:00+0000")
    assert emax == pd.Timestamp("2025-01-02T00:00:00+0000")
    assert xmax == pd.Timestamp("2025-01-03T07:00:00+0000")
    assert n == 2


def test_missing_artifacts(tmp_path):
    empty_dir = tmp_path / "missing"
    pq_result = parquet_max_timestamp(str(empty_dir))
    assert pq_result.empty

    sig_result = signals_max_timestamp(str(empty_dir))
    assert sig_result is None

    trades_result = trades_max_timestamp(str(empty_dir / "trades.csv"))
    assert trades_result is None


def test_main_handles_missing_data(tmp_path, monkeypatch, capsys):
    """Ensure running the CLI helper without assets still reports gracefully."""

    monkeypatch.chdir(tmp_path)

    debug_cutoff_main()

    captured = capsys.readouterr()
    assert "no parquet files found" in captured.out.lower()
    assert "no signals found" in captured.out.lower()
    assert "trades.csv not found" in captured.out.lower()
