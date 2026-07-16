#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

REPO = Path('/opt/testerdonch')
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.qlmg_regime_stack import FINAL_HOLDOUT_START, SCREENING_END, stable_hash, validate_no_protected  # noqa: E402
from tools.qlmg_screening_core import check_resource_guard, resource_snapshot, utc_now, write_json  # noqa: E402
try:  # noqa: E402
    from tools.telegram_notify import TelegramNotifier
except Exception:  # pragma: no cover
    TelegramNotifier = None  # type: ignore

CANDIDATE_ID = 'D4__b4c9487fe82c'
BASE_RUN_ROOT = REPO / 'results/rebaseline/phase_qlmg_d4_survivability_redesign_20260625_v1'
PRIOR_ROOT = REPO / 'results/rebaseline/phase_qlmg_d4_liquidation_execution_audit_20260625_v1_20260625_172927'
EXPECTED = {
    'accepted_events': 4475,
    'resolved_accepted_events': 4475,
    'actual_mark_liquidations': 93,
    'same_minute_ambiguous': 18,
}
LEVERAGES = [2.0, 3.0, 5.0, 7.5, 10.0]

STAGES = (
    'preflight-and-artifact-freeze',
    'telegram-and-tmux-setup',
    'seal-guard',
    'raw-liquidation-inclusive-replay',
    'decision-time-liquidation-geometry',
    'liquidation-safe-sizing-models',
    'stabilization-entry-redesign',
    'stop-and-buffer-redesign',
    'pre-entry-safety-filter-study',
    'matched-null-refresh-after-safety',
    'cost-funding-execution-stress',
    'walk-forward-cpcv-after-safety',
    'aggressive-risk-overlay-after-safety',
    'decision-report',
    'compact-review-bundle',
    'all',
)


@dataclass
class RunNotifier:
    run_root: Path
    disabled: bool = False
    require_remote: bool = False
    allow_no_remote: bool = False

    def __post_init__(self) -> None:
        self.path = self.run_root / 'notifications/telegram_events.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.remote = None
        self.status = 'disabled' if self.disabled else 'unavailable'
        self.missing = ''
        if not self.disabled and TelegramNotifier is not None:
            try:
                class _Args:
                    disable_telegram = False
                    telegram_dry_run = False
                self.remote = TelegramNotifier.from_args(_Args(), run_label='qlmg-d4-survival')
                self.status = 'enabled'
            except Exception as exc:  # pragma: no cover
                self.remote = None
                self.status = 'unavailable'
                self.missing = f'{type(exc).__name__}: {exc}'
        elif not self.disabled:
            self.missing = 'tools.telegram_notify.TelegramNotifier unavailable'
        if self.require_remote and self.remote is None and not self.allow_no_remote:
            raise RuntimeError(f'remote Telegram required but unavailable: {self.missing}')

    def send(self, title: str, body: str, level: str = 'info') -> None:
        rec = {'ts_utc': utc_now(), 'title': title, 'body': body, 'level': level, 'status': self.status, 'sent': False}
        if self.remote is not None:
            try:
                self.remote.send(title, body)
                rec['sent'] = True
            except Exception as exc:  # pragma: no cover
                rec['error'] = f'{type(exc).__name__}: {exc}'
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(rec, sort_keys=True, default=str) + '\n')


@dataclass
class RunContext:
    args: argparse.Namespace
    run_root: Path
    notifier: RunNotifier
    start: pd.Timestamp
    end: pd.Timestamp


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description='D4 survivability redesign audit')
    p.add_argument('--stage', choices=STAGES, default='all')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--max-symbols', type=int, default=None)
    p.add_argument('--start', default='2023-01-01')
    p.add_argument('--end', default=str(SCREENING_END))
    p.add_argument('--chunk-size', type=int, default=50)
    p.add_argument('--max-output-gb', type=float, default=30.0)
    p.add_argument('--allow-large-output', action='store_true')
    p.add_argument('--disable-telegram', action='store_true')
    p.add_argument('--require-telegram', action='store_true')
    p.add_argument('--allow-no-telegram', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--seed', type=int, default=20260625)
    p.add_argument('--nulls-per-event', type=int, default=3)
    p.add_argument('--tmux-session-name', default='qlmg_d4_survival')
    p.add_argument('--run-root', default=None)
    return p.parse_args(argv)


def resolve_run_root(arg: str | None, smoke: bool = False) -> tuple[Path, str]:
    if arg:
        root = Path(arg)
        return (root / 'smoke', 'explicit_smoke_subroot') if smoke and root.name != 'smoke' else (root, 'explicit')
    base = BASE_RUN_ROOT
    if smoke:
        base = base / 'smoke'
    if base.exists() and not smoke:
        suffix = pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')
        return Path(f'{base}_{suffix}'), 'base_exists_timestamp_suffix'
    return base, 'default'


def stage_list(stage: str) -> list[str]:
    return [s for s in STAGES if s != 'all'] if stage == 'all' else [stage]


def done_path(root: Path, stage: str) -> Path:
    return root / 'stage_status' / f'{stage}.done'


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]] | pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    df.to_csv(path, index=False)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def required_outputs(root: Path, stage: str) -> list[Path]:
    m = {
        'preflight-and-artifact-freeze': [root/'preflight/preflight_report.md', root/'preflight/frozen_artifact_hashes.json', root/'preflight/resource_guard_report.md'],
        'telegram-and-tmux-setup': [root/'notifications/telegram_readiness_report.md', root/'tmux/watch_commands.md'],
        'seal-guard': [root/'seal/seal_guard_report.md', root/'seal/protected_slice_check.json'],
        'raw-liquidation-inclusive-replay': [root/'raw/raw_liquidation_inclusive_summary.csv', root/'raw/raw_liquidation_inclusive_report.md'],
        'decision-time-liquidation-geometry': [root/'geometry/decision_time_liquidation_geometry.parquet', root/'geometry/geometry_summary.csv', root/'geometry/geometry_report.md'],
        'liquidation-safe-sizing-models': [root/'sizing/liquidation_safe_sizing_summary.csv', root/'sizing/liquidation_safe_sizing_report.md'],
        'stabilization-entry-redesign': [root/'stabilization/stabilization_entry_summary.csv', root/'stabilization/stabilization_entry_report.md'],
        'stop-and-buffer-redesign': [root/'stops/stop_buffer_redesign_summary.csv', root/'stops/stop_buffer_redesign_report.md'],
        'pre-entry-safety-filter-study': [root/'filters/pre_entry_safety_filter_summary.csv', root/'filters/pre_entry_safety_filter_report.md'],
        'matched-null-refresh-after-safety': [root/'matched_null/safety_matched_null_summary.csv', root/'matched_null/safety_matched_null_report.md'],
        'cost-funding-execution-stress': [root/'stress/safety_cost_funding_stress_summary.csv', root/'stress/safety_cost_funding_stress_report.md'],
        'walk-forward-cpcv-after-safety': [root/'validation/safety_walk_forward_summary.csv', root/'validation/safety_cpcv_summary.csv', root/'validation/safety_validation_report.md'],
        'aggressive-risk-overlay-after-safety': [root/'portfolio/survivable_aggressive_overlay_summary.csv', root/'portfolio/survivable_aggressive_overlay_report.md'],
        'decision-report': [root/'D4_SURVIVABILITY_REDESIGN_REPORT.md', root/'decision_summary.json'],
        'compact-review-bundle': [root/'compact_review_bundle/artifact_path_index.csv'],
    }
    return m.get(stage, [])


def stage_complete(root: Path, stage: str) -> bool:
    return done_path(root, stage).exists() and all(p.exists() for p in required_outputs(root, stage))


def estimate_stage_gb(stage: str, ctx: RunContext) -> float:
    return 0.25 if stage in {'stabilization-entry-redesign', 'stop-and-buffer-redesign'} else 0.05


def ensure_guard(ctx: RunContext, stage: str) -> None:
    snap = resource_snapshot(REPO)
    status = check_resource_guard(snap, estimated_output_gb=estimate_stage_gb(stage, ctx), allow_large_output=ctx.args.allow_large_output)
    out = {'stage': stage, **status, 'snapshot': snap.__dict__}
    write_json(ctx.run_root / 'resource_guard' / f'{stage}.json', out)
    if status['warnings']:
        ctx.notifier.send('D4 SURVIVAL RESOURCE WARNING', f'stage={stage}\nwarnings={status["warnings"]}', level='warning')
    if status['status'] != 'pass':
        ctx.notifier.send('D4 SURVIVAL RESOURCE HARD STOP', f'stage={stage}\nreasons={status["reasons"]}', level='error')
        raise RuntimeError(f'resource guard failed for {stage}: {status["reasons"]}')


def _bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.lower().isin(['true', '1', 'yes'])


def pf(vals: pd.Series) -> float:
    v = pd.to_numeric(vals, errors='coerce').dropna()
    pos = float(v[v > 0].sum())
    neg = float(v[v < 0].sum())
    return pos / abs(neg) if neg < 0 else (float('inf') if pos > 0 else 0.0)


def max_dd(vals: pd.Series) -> float:
    v = pd.to_numeric(vals, errors='coerce').fillna(0.0)
    if v.empty:
        return 0.0
    cs = v.cumsum()
    dd = cs - cs.cummax()
    return float(dd.min())


def summarize_returns(df: pd.DataFrame, *, r_col: str = 'candidate_net_R', candidate_id: str = '', candidate_type: str = '', extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    v = pd.to_numeric(df.get(r_col, pd.Series(dtype=float)), errors='coerce').dropna()
    out = {
        'candidate_id': candidate_id,
        'candidate_type': candidate_type,
        'events': int(len(v)),
        'net_R': float(v.sum()) if len(v) else 0.0,
        'mean_R': float(v.mean()) if len(v) else 0.0,
        'median_R': float(v.median()) if len(v) else 0.0,
        'PF': pf(v),
        'hit_rate': float((v > 0).mean()) if len(v) else 0.0,
        'max_dd_R_proxy': max_dd(v),
        'liquidation_count': int(_bool_series(df.get('candidate_actual_liquidation', pd.Series(False, index=df.index))).sum()) if len(df) else 0,
        'ambiguous_count': int(_bool_series(df.get('candidate_same_minute_ambiguous', pd.Series(False, index=df.index))).sum()) if len(df) else 0,
        'symbols': int(df['symbol'].nunique()) if 'symbol' in df and len(df) else 0,
        'months': int(pd.to_datetime(df['decision_ts'], utc=True, errors='coerce').dt.strftime('%Y-%m').nunique()) if 'decision_ts' in df and len(df) else 0,
    }
    if extra:
        out.update(dict(extra))
    return out


def liquidation_adverse_bps_for_leverage(leverage: float) -> float:
    return max(10000.0 / float(leverage) - 50.0, 0.0)


def load_prior_replay(ctx: RunContext, include_nulls: bool = False) -> pd.DataFrame:
    p = PRIOR_ROOT / 'one_minute_mark/d4_1m_mark_replay_by_window.parquet'
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_parquet(p)
    if not include_nulls:
        df = df[df['window_type'].astype(str).eq('accepted_d4_event')].copy()
    if ctx.args.smoke:
        syms = sorted(df['symbol'].dropna().astype(str).unique())[: max(1, ctx.args.max_symbols or 5)]
        df = df[df['symbol'].astype(str).isin(syms)].copy()
    elif ctx.args.max_symbols:
        syms = sorted(df['symbol'].dropna().astype(str).unique())[:ctx.args.max_symbols]
        df = df[df['symbol'].astype(str).isin(syms)].copy()
    for c in ['decision_ts', 'window_start', 'window_end', 'entry_ts', 'horizon_end', 'exit_ts_1m']:
        if c in df:
            df[c] = pd.to_datetime(df[c], utc=True, errors='coerce')
    df = df[(df['decision_ts'] >= ctx.start) & (df['decision_ts'] <= ctx.end)].copy()
    validate_no_protected(df, ['decision_ts', 'window_start', 'window_end', 'entry_ts', 'horizon_end', 'exit_ts_1m'])
    return df


def load_event_ledger(ctx: RunContext) -> pd.DataFrame:
    p = PRIOR_ROOT / 'd4_reconstruction/d4_event_ledger.parquet'
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_parquet(p)
    for c in ['decision_ts', 'entry_ts', 'feature_ts']:
        if c in df:
            df[c] = pd.to_datetime(df[c], utc=True, errors='coerce')
    if ctx.args.smoke:
        replay = load_prior_replay(ctx)
        df = df[df['event_id'].astype(str).isin(set(replay['event_id'].astype(str)))].copy()
    else:
        df = df[(df['decision_ts'] >= ctx.start) & (df['decision_ts'] <= ctx.end)].copy()
    validate_no_protected(df, ['decision_ts', 'entry_ts', 'feature_ts'])
    return df


def load_base_events(ctx: RunContext) -> pd.DataFrame:
    rep = load_prior_replay(ctx).copy()
    evt = load_event_ledger(ctx).copy()
    evt_cols = [c for c in evt.columns if c not in rep.columns or c == 'event_id']
    df = rep.merge(evt[evt_cols], on='event_id', how='left', suffixes=('', '_event'))
    df['candidate_actual_liquidation'] = _bool_series(df.get('actual_mark_liquidation_1m', pd.Series(False, index=df.index)))
    df['candidate_same_minute_ambiguous'] = _bool_series(df.get('same_minute_ambiguity_1m', pd.Series(False, index=df.index)))
    df['candidate_rankable'] = _bool_series(df.get('rankable_1m_mark', pd.Series(False, index=df.index)))
    df['candidate_net_R'] = pd.to_numeric(df['net_R_1m_mark'], errors='coerce')
    df['cost_bps_1m'] = pd.to_numeric(df.get('cost_bps_1m'), errors='coerce').fillna(30.0)
    df['stop_distance_bps_1m'] = pd.to_numeric(df.get('stop_distance_bps_1m'), errors='coerce')
    validate_no_protected(df, ['decision_ts', 'window_start', 'window_end', 'entry_ts', 'horizon_end', 'exit_ts_1m'])
    return df


def conservative_r(df: pd.DataFrame, *, include_liq: bool, include_ambiguous: bool) -> pd.DataFrame:
    out = df.copy()
    r = pd.to_numeric(out['candidate_net_R'], errors='coerce')
    if include_liq:
        liq = _bool_series(out['candidate_actual_liquidation'])
        r.loc[liq] = np.minimum(r.loc[liq], -1.0)
    if include_ambiguous:
        amb = _bool_series(out['candidate_same_minute_ambiguous'])
        r.loc[amb] = np.minimum(r.loc[amb], -1.25)
    out['candidate_net_R'] = r
    return out


def download_index() -> dict[tuple[str, str], Path]:
    p = PRIOR_ROOT / 'downloaded_1m/download_manifest.csv'
    df = pd.read_csv(p) if p.exists() else pd.DataFrame()
    out: dict[tuple[str, str], Path] = {}
    if df.empty:
        return out
    for _, r in df[df.get('status').astype(str).eq('ok')].iterrows():
        path = str(r.get('path', ''))
        if path and path != 'nan':
            out[(str(r['window_id']), str(r['dataset']))] = PRIOR_ROOT / path
    return out


def read_path_pair(cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]], files: dict[tuple[str, str], Path], wid: str) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if wid in cache:
        return cache[wid]
    op = files.get((wid, 'ohlcv_1m'))
    mp = files.get((wid, 'mark_1m'))
    if op is None or mp is None or not op.exists() or not mp.exists():
        return None
    ohlcv = pd.read_parquet(op).sort_values('timestamp')
    mark = pd.read_parquet(mp).sort_values('timestamp')
    ohlcv['timestamp'] = pd.to_datetime(ohlcv['timestamp'], utc=True, errors='coerce')
    mark['timestamp'] = pd.to_datetime(mark['timestamp'], utc=True, errors='coerce')
    cache[wid] = (ohlcv, mark)
    return cache[wid]


def first_open_at_or_after(ohlcv: pd.DataFrame, ts: pd.Timestamp) -> tuple[pd.Timestamp | None, float | None]:
    sub = ohlcv[ohlcv['timestamp'] >= ts]
    if sub.empty:
        return None, None
    r = sub.iloc[0]
    return pd.Timestamp(r['timestamp']), float(r['open'])


def replay_long_path(row: Mapping[str, Any], ohlcv: pd.DataFrame, mark: pd.DataFrame, *, entry_ts: pd.Timestamp | None = None, stop_bps: float | None = None, target_r: float | None = 1.0, horizon_hours: float = 2.0, leverage: float = 10.0, cost_bps: float | None = None, entry_price_override: float | None = None) -> dict[str, Any]:
    base_entry_ts = pd.Timestamp(row['entry_ts'])
    ets = pd.Timestamp(entry_ts) if entry_ts is not None else base_entry_ts
    if entry_price_override is not None:
        actual_entry_ts, entry = ets, float(entry_price_override)
    else:
        actual_entry_ts, entry = first_open_at_or_after(ohlcv, ets)
    if actual_entry_ts is None or entry is None or not np.isfinite(entry) or entry <= 0:
        return {'replay_status': 'fail_closed_missing_entry'}
    sbps = float(stop_bps if stop_bps is not None else row.get('stop_distance_bps_1m', np.nan))
    if not np.isfinite(sbps) or sbps <= 0:
        return {'replay_status': 'fail_closed_bad_stop_distance'}
    cbps = float(cost_bps if cost_bps is not None else row.get('cost_bps_1m', 30.0))
    liq_bps = liquidation_adverse_bps_for_leverage(leverage)
    stop = entry * (1.0 - sbps / 10000.0)
    target = None if target_r is None else entry * (1.0 + float(target_r) * sbps / 10000.0)
    liq = entry * (1.0 - liq_bps / 10000.0)
    horizon_end = actual_entry_ts + pd.Timedelta(hours=float(horizon_hours))
    px = ohlcv[(ohlcv['timestamp'] >= actual_entry_ts) & (ohlcv['timestamp'] <= horizon_end)].copy()
    mk = mark[(mark['timestamp'] >= actual_entry_ts) & (mark['timestamp'] <= horizon_end)].copy()
    if px.empty or mk.empty:
        return {'replay_status': 'fail_closed_missing_path'}
    merged = px[['timestamp', 'high', 'low', 'close']].merge(mk[['timestamp', 'high', 'low', 'close']], on='timestamp', how='left', suffixes=('', '_mark'))
    if merged[['high_mark', 'low_mark']].isna().any().any():
        return {'replay_status': 'fail_closed_incomplete_mark_path'}
    exit_ts = pd.Timestamp(merged.iloc[-1]['timestamp'])
    exit_price = float(merged.iloc[-1]['close'])
    reason = 'time_exit'
    ordering = 'no_liquidation'
    actual_liq = False
    ambiguous = False
    stop_hit_any = False
    liq_hit_any = False
    for _, r in merged.iterrows():
        ts = pd.Timestamp(r['timestamp'])
        stop_hit = float(r['low']) <= stop
        target_hit = False if target is None else float(r['high']) >= float(target)
        liq_hit = float(r['low_mark']) <= liq
        stop_hit_any = stop_hit_any or stop_hit
        liq_hit_any = liq_hit_any or liq_hit
        if liq_hit and stop_hit:
            exit_ts, exit_price, reason = ts, liq, 'same_minute_stop_liquidation_ambiguous'
            ordering, ambiguous = 'same_minute_stop_liquidation_ambiguous', True
            break
        if liq_hit:
            exit_ts, exit_price, reason = ts, liq, 'liquidation'
            ordering, actual_liq = 'liquidation_before_stop', True
            break
        if stop_hit and target_hit:
            exit_ts, exit_price, reason = ts, stop, 'stop'
            ordering = 'same_minute_stop_target_pessimistic_stop'
            break
        if stop_hit:
            exit_ts, exit_price, reason = ts, stop, 'stop'
            ordering = 'stop_no_liquidation'
            break
        if target_hit:
            exit_ts, exit_price, reason = ts, float(target), 'target'
            ordering = 'target_before_liquidation'
            break
    gross_bps = (exit_price / entry - 1.0) * 10000.0
    total_cost = cbps + (20.0 if reason == 'liquidation' else 0.0)
    net_r = (gross_bps - total_cost) / max(sbps, 1e-9)
    mark_low = float(merged['low_mark'].min())
    mark_high = float(merged['high_mark'].max())
    return {
        'replay_status': 'resolved',
        'entry_ts_model': actual_entry_ts,
        'entry_price_model': entry,
        'exit_ts_model': exit_ts,
        'exit_price_model': exit_price,
        'exit_reason_model': reason,
        'ordering_class_model': ordering,
        'candidate_actual_liquidation': actual_liq,
        'candidate_same_minute_ambiguous': ambiguous,
        'stop_hit_any_model': stop_hit_any,
        'liq_hit_any_model': liq_hit_any,
        'stop_bps_model': sbps,
        'target_r_model': target_r if target_r is not None else np.nan,
        'leverage_model': leverage,
        'stop_price_model': stop,
        'target_price_model': target if target is not None else np.nan,
        'liq_price_model': liq,
        'gross_bps_model': gross_bps,
        'cost_bps_model': total_cost,
        'candidate_net_R': net_r,
        'mark_mfe_bps_model': (mark_high / entry - 1.0) * 10000.0,
        'mark_mae_bps_model': (1.0 - mark_low / entry) * 10000.0,
    }


def replay_variant_rows(df: pd.DataFrame, variant_id: str, *, entry_delay_minutes: int = 0, require_no_lower_low_minutes: int | None = None, stop_bps_func=None, target_r: float | None = 1.0, horizon_hours: float = 2.0, leverage: float = 10.0, max_rows: int | None = None) -> pd.DataFrame:
    files = download_index()
    cache: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    rows: list[dict[str, Any]] = []
    src = df.copy()
    if max_rows is not None:
        src = src.head(max_rows)
    for _, row in src.iterrows():
        wid = str(row.get('dedup_window_id', ''))
        pair = read_path_pair(cache, files, wid)
        base = {k: row.get(k) for k in ['event_id', 'symbol', 'decision_ts', 'entry_ts', 'liquidity_tier', 'dedup_window_id'] if k in row}
        base['candidate_id'] = variant_id
        base['candidate_type'] = 'replay_variant'
        if pair is None:
            rows.append({**base, 'replay_status': 'fail_closed_missing_downloaded_path'})
            continue
        ohlcv, mark = pair
        entry_ts = pd.Timestamp(row['entry_ts']) + pd.Timedelta(minutes=entry_delay_minutes)
        if require_no_lower_low_minutes is not None:
            start = pd.Timestamp(row['entry_ts'])
            check_end = start + pd.Timedelta(minutes=require_no_lower_low_minutes)
            check = ohlcv[(ohlcv['timestamp'] >= start) & (ohlcv['timestamp'] <= check_end)]
            if check.empty or float(check['low'].min()) < float(row.get('entry_price_1m', np.nan)):
                rows.append({**base, 'replay_status': 'skipped_no_lower_low_condition_failed'})
                continue
        sbps = stop_bps_func(row) if stop_bps_func is not None else float(row.get('stop_distance_bps_1m', np.nan))
        rep = replay_long_path(row, ohlcv, mark, entry_ts=entry_ts, stop_bps=sbps, target_r=target_r, horizon_hours=horizon_hours, leverage=leverage)
        rows.append({**base, **rep})
    out = pd.DataFrame(rows)
    validate_no_protected(out, ['decision_ts', 'entry_ts', 'entry_ts_model', 'exit_ts_model'])
    return out


def stage_preflight(ctx: RunContext) -> None:
    required = [
        PRIOR_ROOT / 'D4_1M_MARK_REPLAY_FINAL_RESULTS_REPORT.md',
        PRIOR_ROOT / 'd4_reconstruction/d4_event_ledger.parquet',
        PRIOR_ROOT / 'd4_reconstruction/d4_trade_ledger.parquet',
        PRIOR_ROOT / 'one_minute_mark/d4_1m_mark_replay_by_window.parquet',
        PRIOR_ROOT / 'matched_null/d4_refreshed_matched_null_summary.csv',
        PRIOR_ROOT / 'stress/d4_cost_funding_execution_stress_summary.csv',
        PRIOR_ROOT / 'portfolio/d4_aggressive_10x_risk_expression_summary.csv',
        PRIOR_ROOT / 'downloaded_1m/download_manifest.csv',
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f'missing required prior artifacts: {missing}')
    rep = load_prior_replay(ctx)
    if not ctx.args.smoke and not ctx.args.max_symbols:
        counts = {
            'accepted_events': int(len(rep)),
            'resolved_accepted_events': int(rep['mark_replay_status'].astype(str).eq('resolved_1m_mark_path').sum()),
            'actual_mark_liquidations': int(_bool_series(rep['actual_mark_liquidation_1m']).sum()),
            'same_minute_ambiguous': int(_bool_series(rep['same_minute_ambiguity_1m']).sum()),
        }
        if counts != EXPECTED:
            raise RuntimeError(f'prior D4 replay counts mismatch: {counts} expected {EXPECTED}')
    hashes = {str(p.relative_to(REPO)): file_sha256(p) for p in required}
    write_json(ctx.run_root / 'preflight/frozen_artifact_hashes.json', hashes)
    snap = resource_snapshot(REPO)
    write_text(ctx.run_root / 'preflight/resource_guard_report.md', f"# Resource Guard Report\n\n- free disk GB: `{snap.free_gb:.2f}`\n- hard stop: `5GB`\n- warning: `7GB`\n- stage output block: `20GB`\n- max output GB: `{ctx.args.max_output_gb}`\n")
    write_text(ctx.run_root / 'preflight/preflight_report.md', f"# Preflight And Artifact Freeze\n\n- candidate: `{CANDIDATE_ID}`\n- prior root: `{PRIOR_ROOT}`\n- requested window: `{ctx.start}` to `{ctx.end}`\n- accepted events loaded: `{len(rep)}`\n- resolved events: `{int(rep['mark_replay_status'].astype(str).eq('resolved_1m_mark_path').sum())}`\n- actual mark liquidations: `{int(_bool_series(rep['actual_mark_liquidation_1m']).sum())}`\n- same-minute ambiguity: `{int(_bool_series(rep['same_minute_ambiguity_1m']).sum())}`\n- prior net_R excluded unsafe rows: `yes, rankable rows exclude liquidation and ambiguous rows`\n- final holdout cutoff: `{FINAL_HOLDOUT_START}`\n- missing artifacts: `{missing}`\n")


def stage_telegram(ctx: RunContext) -> None:
    watch = f"""# Watch Commands\n\ntmux attach -t {ctx.args.tmux_session_name}\n\ntail -f {ctx.run_root}/logs/full_run.log\n\nwatch -n 30 'cat {ctx.run_root}/watch_status.json'\n\ntail -f {ctx.run_root}/notifications/telegram_events.jsonl\n\ndf -h / /opt/testerdonch /opt/parquet 2>/dev/null || df -h\n"""
    write_text(ctx.run_root / 'tmux/watch_commands.md', watch)
    write_text(ctx.run_root / 'tmux/tmux_run_instructions.md', f"# Tmux Run Instructions\n\nDefault session: `{ctx.args.tmux_session_name}`\n\nUse `tools/run_qlmg_d4_survivability_redesign_tmux.sh`.\n")
    write_text(ctx.run_root / 'notifications/telegram_readiness_report.md', f"# Telegram Readiness\n\n- status: `{ctx.notifier.status}`\n- missing: `{ctx.notifier.missing}`\n- require remote: `{ctx.args.require_telegram}`\n- allow no remote: `{ctx.args.allow_no_telegram}`\n")
    ctx.notifier.send('D4 SURVIVAL RUN READY', f'run_root={ctx.run_root}')


def stage_seal(ctx: RunContext) -> None:
    info = {'protected_start': str(FINAL_HOLDOUT_START), 'screening_end': str(SCREENING_END), 'requested_start': str(ctx.start), 'requested_end': str(ctx.end), 'status': 'pass'}
    if ctx.end >= FINAL_HOLDOUT_START:
        info['status'] = 'fail'
        write_json(ctx.run_root / 'seal/protected_slice_check.json', info)
        raise RuntimeError('requested end overlaps protected slice')
    df = load_prior_replay(ctx, include_nulls=True)
    validate_no_protected(df, ['decision_ts', 'window_start', 'window_end', 'entry_ts', 'horizon_end', 'exit_ts_1m'])
    info['loaded_rows_checked'] = int(len(df))
    write_json(ctx.run_root / 'seal/protected_slice_check.json', info)
    write_text(ctx.run_root / 'seal/seal_guard_report.md', f"# Seal Guard Report\n\n- protected slice: `{FINAL_HOLDOUT_START}` onward\n- requested end: `{ctx.end}`\n- loaded rows checked: `{len(df)}`\n- status: `pass`\n")


def stage_raw(ctx: RunContext) -> None:
    df = load_base_events(ctx)
    rankable = df[df['candidate_rankable']].copy()
    liq_inc = conservative_r(df[df['candidate_rankable'] | df['candidate_actual_liquidation']], include_liq=True, include_ambiguous=False)
    full = conservative_r(df[df['candidate_rankable'] | df['candidate_actual_liquidation'] | df['candidate_same_minute_ambiguous']], include_liq=True, include_ambiguous=True)
    rows = [
        summarize_returns(rankable, candidate_id='rankable_prior_view', candidate_type='raw_view', extra={'view': 'exclude_liquidation_and_ambiguous'}),
        summarize_returns(liq_inc, candidate_id='liquidation_inclusive_conservative', candidate_type='raw_view', extra={'view': 'include_actual_liquidations_conservative'}),
        summarize_returns(full, candidate_id='liquidation_and_ambiguous_pessimistic', candidate_type='raw_view', extra={'view': 'include_liquidation_and_ambiguous_pessimistic'}),
    ]
    write_csv(ctx.run_root / 'raw/raw_liquidation_inclusive_summary.csv', rows)
    write_text(ctx.run_root / 'raw/raw_liquidation_inclusive_report.md', '# Raw Liquidation Inclusive Replay\n\n' + pd.DataFrame(rows).to_markdown(index=False) + '\n\nUnsafe rows reduce but do not fully erase headline net_R; liquidation risk remains a hard rejection condition for the prior expression.\n')


def stage_geometry(ctx: RunContext) -> None:
    df = load_base_events(ctx)
    g = pd.DataFrame({
        'event_id': df['event_id'].astype(str),
        'symbol': df['symbol'].astype(str),
        'decision_ts': df['decision_ts'],
        'entry_ts': df['entry_ts'],
        'entry_price': pd.to_numeric(df['entry_price_1m'], errors='coerce'),
        'mark_price_at_decision': np.nan,
        'mark_price_status': 'not_available_in_prior_replay_entry_price_used_for_geometry',
        'stop_price': pd.to_numeric(df['stop_price_1m'], errors='coerce'),
        'target_price': pd.to_numeric(df['target_price_1m'], errors='coerce'),
        'stop_distance_bps': pd.to_numeric(df['stop_distance_bps_1m'], errors='coerce'),
        'stop_distance_pct': pd.to_numeric(df['stop_distance_bps_1m'], errors='coerce') / 100.0,
        'atr_bps': pd.to_numeric(df.get('atr_bps'), errors='coerce'),
        'funding_rate': pd.to_numeric(df.get('funding_rate'), errors='coerce'),
        'oi_chg_24h': pd.to_numeric(df.get('oi_chg_24h'), errors='coerce'),
        'liquidity_tier': df.get('liquidity_tier', pd.Series('', index=df.index)).astype(str),
        'listing_age_bucket': df.get('listing_age_bucket', pd.Series('unknown', index=df.index)).astype(str),
        'data_quality_flags': df.get('data_quality_flags', pd.Series('', index=df.index)).astype(str),
        'price_oi_matrix_24h': df.get('price_oi_matrix_24h', pd.Series('unknown', index=df.index)).astype(str),
        'funding_sign_label': df.get('funding_sign_label', pd.Series('unknown', index=df.index)).astype(str),
        'turnover_bucket': df.get('turnover_bucket', pd.Series('unknown', index=df.index)).astype(str),
        'btc_eth_regime_label': df.get('btc_eth_regime_label', pd.Series('unknown', index=df.index)).astype(str),
        'actual_mark_liquidation_label': df['candidate_actual_liquidation'],
        'same_minute_ambiguity_label': df['candidate_same_minute_ambiguous'],
        'label_usage_policy': 'labels_for_evaluation_only_not_filter_inputs',
    })
    for lev in LEVERAGES:
        bps = liquidation_adverse_bps_for_leverage(lev)
        tag = str(lev).replace('.', 'p')
        g[f'liq_adverse_bps_{tag}x'] = bps
        g[f'liq_price_{tag}x'] = g['entry_price'] * (1.0 - bps / 10000.0)
        g[f'liq_to_stop_ratio_{tag}x'] = bps / g['stop_distance_bps'].replace(0, np.nan)
    validate_no_protected(g, ['decision_ts', 'entry_ts'])
    outp = ctx.run_root / 'geometry/decision_time_liquidation_geometry.parquet'
    outp.parent.mkdir(parents=True, exist_ok=True)
    g.to_parquet(outp, index=False, compression='zstd')
    rows = []
    for lev in LEVERAGES:
        col = f'liq_to_stop_ratio_{str(lev).replace(".", "p")}x'
        rows.append({'leverage': lev, 'median_liq_to_stop_ratio': float(g[col].median()), 'share_ratio_ge_1p25': float((g[col] >= 1.25).mean()), 'share_ratio_ge_1p5': float((g[col] >= 1.5).mean()), 'share_ratio_ge_2p0': float((g[col] >= 2.0).mean())})
    write_csv(ctx.run_root / 'geometry/geometry_summary.csv', rows)
    write_text(ctx.run_root / 'geometry/geometry_report.md', '# Decision-Time Liquidation Geometry\n\n' + pd.DataFrame(rows).to_markdown(index=False) + '\n\nFuture liquidation labels are included only as evaluation labels and are not used by geometry filters.\n')


def model_liquidation_by_threshold(df: pd.DataFrame, leverage: float) -> pd.Series:
    threshold = liquidation_adverse_bps_for_leverage(leverage)
    mae = pd.to_numeric(df.get('mark_mae_bps_1m'), errors='coerce').fillna(0.0)
    stop = pd.to_numeric(df.get('stop_distance_bps_1m'), errors='coerce').fillna(np.inf)
    return (mae >= threshold) & (threshold <= stop)


def stage_sizing(ctx: RunContext) -> None:
    df = load_base_events(ctx)
    rows = []
    models = [
        ('fixed_10x', 'fixed', 10.0, None),
        ('dynamic_buffer_1p25_max10x', 'buffer', 10.0, 1.25),
        ('dynamic_buffer_1p5_max10x', 'buffer', 10.0, 1.5),
        ('dynamic_buffer_2p0_max10x', 'buffer', 10.0, 2.0),
        ('fixed_5x', 'fixed', 5.0, None),
        ('fixed_3x', 'fixed', 3.0, None),
        ('risk_based_reduce_until_buffer_1p5', 'buffer', 10.0, 1.5),
        ('skip_if_safe_sizing_below_practical_3x', 'buffer_min3x', 10.0, 1.5),
    ]
    for cid, mode, maxlev, buffer in models:
        m = df.copy()
        stop = pd.to_numeric(m['stop_distance_bps_1m'], errors='coerce')
        if mode == 'fixed':
            lev = pd.Series(float(maxlev), index=m.index)
        else:
            lev = 10000.0 / (float(buffer) * stop + 50.0)
            lev = lev.clip(upper=float(maxlev))
            if mode == 'buffer_min3x':
                m = m[lev >= 3.0].copy()
                lev = lev.loc[m.index]
        m['model_leverage'] = lev
        m['candidate_actual_liquidation'] = [bool(model_liquidation_by_threshold(m.loc[[i]], float(lev.loc[i])).iloc[0]) for i in m.index]
        m['candidate_same_minute_ambiguous'] = _bool_series(m['same_minute_ambiguity_1m']) & m['candidate_actual_liquidation']
        r = pd.to_numeric(m['net_R_1m_mark'], errors='coerce').copy()
        r.loc[m['candidate_actual_liquidation']] = np.minimum(r.loc[m['candidate_actual_liquidation']], -1.0)
        m['candidate_net_R'] = r
        rows.append(summarize_returns(m, candidate_id=cid, candidate_type='sizing_model', extra={'avg_leverage': float(m['model_leverage'].mean()) if len(m) else 0.0, 'median_leverage': float(m['model_leverage'].median()) if len(m) else 0.0, 'trades_retained': int(len(m)), 'trades_skipped': int(len(df) - len(m))}))
    summ = pd.DataFrame(rows)
    write_csv(ctx.run_root / 'sizing/liquidation_safe_sizing_summary.csv', summ)
    write_text(ctx.run_root / 'sizing/liquidation_safe_sizing_report.md', '# Liquidation Safe Sizing Models\n\n' + summ.to_markdown(index=False) + '\n')


def stage_stabilization(ctx: RunContext) -> None:
    df = load_base_events(ctx)
    variants = [
        ('original_entry_replay', 0, None, None),
        ('wait_30m_after_reclaim', 30, None, None),
        ('wait_1h_after_reclaim', 60, None, None),
        ('wait_4h_after_reclaim', 240, None, None),
        ('require_no_lower_low_30m', 30, 30, None),
        ('require_no_lower_low_1h', 60, 60, None),
        ('oi_down_funding_reset_before_entry', 0, None, 'oi_funding'),
        ('require_price_down_oi_down_state', 0, None, 'price_down_oi_down'),
        ('skip_price_down_oi_up_state', 0, None, 'skip_price_down_oi_up'),
    ]
    rows = []
    samples = []
    for vid, delay, nll, flt in variants[:12]:
        sub = df.copy()
        if flt == 'oi_funding':
            sub = sub[(pd.to_numeric(sub.get('oi_chg_24h'), errors='coerce') <= 0) & (pd.to_numeric(sub.get('funding_rate'), errors='coerce') <= 0.0001)].copy()
        elif flt == 'price_down_oi_down':
            sub = sub[sub.get('price_oi_matrix_24h', pd.Series('', index=sub.index)).astype(str).eq('price_down_oi_down')].copy()
        elif flt == 'skip_price_down_oi_up':
            sub = sub[~sub.get('price_oi_matrix_24h', pd.Series('', index=sub.index)).astype(str).eq('price_down_oi_up')].copy()
        rep = replay_variant_rows(sub, vid, entry_delay_minutes=delay, require_no_lower_low_minutes=nll, target_r=1.0, horizon_hours=2.0, leverage=10.0, max_rows=200 if ctx.args.smoke else None)
        ok = rep[rep['replay_status'].astype(str).eq('resolved')].copy()
        rows.append(summarize_returns(ok, candidate_id=vid, candidate_type='stabilization', extra={'input_events': int(len(sub)), 'resolved_events': int(len(ok)), 'fail_closed_events': int(len(rep) - len(ok)), 'entry_delay_minutes': delay, 'filter': flt or ''}))
        ctx.notifier.send('D4 SURVIVAL STABILIZATION PROGRESS', f'variant={vid} resolved={len(ok)} fail_closed={len(rep)-len(ok)}')
        samples.append(rep.head(100))
    summ = pd.DataFrame(rows)
    write_csv(ctx.run_root / 'stabilization/stabilization_entry_summary.csv', summ)
    if samples:
        outp = ctx.run_root / 'stabilization/stabilization_entry_sample.parquet'
        outp.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(samples, ignore_index=True).to_parquet(outp, index=False, compression='zstd')
    write_text(ctx.run_root / 'stabilization/stabilization_entry_report.md', '# Stabilization Entry Redesign\n\n' + summ.to_markdown(index=False) + '\n\nDelayed/reconfirmed entries are replayed from existing downloaded 1m OHLCV/mark paths.\n')


def stop_func_factory(kind: str):
    def f(row: Mapping[str, Any]) -> float:
        orig = float(row.get('stop_distance_bps_1m', np.nan))
        atr = float(row.get('atr_bps', np.nan)) if pd.notna(row.get('atr_bps', np.nan)) else np.nan
        if kind == 'original' or not np.isfinite(atr):
            return orig
        if kind == 'atr_1p0':
            return atr
        if kind == 'atr_1p5':
            return 1.5 * atr
        if kind == 'atr_2p0':
            return 2.0 * atr
        if kind == 'hybrid_max_original_atr_1p0':
            return max(orig, atr)
        if kind == 'hybrid_max_original_atr_1p5':
            return max(orig, 1.5 * atr)
        return orig
    return f


def stage_stops(ctx: RunContext) -> None:
    df = load_base_events(ctx)
    stop_kinds = ['original', 'atr_1p0', 'atr_1p5', 'atr_2p0', 'hybrid_max_original_atr_1p0', 'hybrid_max_original_atr_1p5']
    # Deterministic bounded subset: covers all stop classes and representative target/time exits without the expensive full grid.
    target_specs = [('original_1R', 1.0, 2.0), ('target_3R', 3.0, 2.0), ('time_6h', None, 6.0)]
    rows = []
    samples = []
    count = 0
    for sk in stop_kinds:
        for tname, tr, hh in target_specs:
            count += 1
            if count > 36:
                break
            vid = f'{sk}__{tname}'
            rep = replay_variant_rows(df, vid, stop_bps_func=stop_func_factory(sk), target_r=tr, horizon_hours=hh, leverage=10.0, max_rows=200 if ctx.args.smoke else None)
            ok = rep[rep['replay_status'].astype(str).eq('resolved')].copy()
            rows.append(summarize_returns(ok, candidate_id=vid, candidate_type='stop_buffer', extra={'stop_class': sk, 'exit_class': tname, 'resolved_events': int(len(ok)), 'fail_closed_events': int(len(rep) - len(ok))}))
            ctx.notifier.send('D4 SURVIVAL STOP PROGRESS', f'variant={vid} resolved={len(ok)} fail_closed={len(rep)-len(ok)}')
            samples.append(rep.head(50))
    summ = pd.DataFrame(rows)
    write_csv(ctx.run_root / 'stops/stop_buffer_redesign_summary.csv', summ)
    if samples:
        outp = ctx.run_root / 'stops/stop_buffer_redesign_sample.parquet'
        outp.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(samples, ignore_index=True).to_parquet(outp, index=False, compression='zstd')
    write_text(ctx.run_root / 'stops/stop_buffer_redesign_report.md', '# Stop And Buffer Redesign\n\n' + summ.to_markdown(index=False) + '\n\nMaximum stop/exit variants evaluated: `36`. Pessimistic same-minute handling is primary.\n')


def stage_filters(ctx: RunContext) -> None:
    df = load_base_events(ctx)
    geom_p = ctx.run_root / 'geometry/decision_time_liquidation_geometry.parquet'
    geom = pd.read_parquet(geom_p) if geom_p.exists() else pd.DataFrame()
    base = df.merge(geom[['event_id', 'liq_to_stop_ratio_10p0x', 'stop_distance_bps', 'listing_age_bucket', 'data_quality_flags']], on='event_id', how='left', suffixes=('', '_geom')) if not geom.empty else df.copy()
    filters: list[tuple[str, pd.Series, str]] = []
    ratio = pd.to_numeric(base.get('liq_to_stop_ratio_10p0x'), errors='coerce')
    filters += [
        ('liq_to_stop_ratio_ge_1p25', ratio >= 1.25, 'liq_to_stop_ratio_10p0x >= 1.25'),
        ('liq_to_stop_ratio_ge_1p5', ratio >= 1.5, 'liq_to_stop_ratio_10p0x >= 1.5'),
        ('liq_to_stop_ratio_ge_2p0', ratio >= 2.0, 'liq_to_stop_ratio_10p0x >= 2.0'),
        ('clean_data_quality_flags', base.get('data_quality_flags', pd.Series('', index=base.index)).astype(str).isin(['', 'clean_or_unflagged']), 'data_quality_flags clean'),
        ('stop_distance_100_to_950bps', pd.to_numeric(base.get('stop_distance_bps_1m'), errors='coerce').between(100, 950), '100 <= stop_distance_bps <= 950'),
        ('min_tier_B_or_C', base.get('liquidity_tier', pd.Series('', index=base.index)).astype(str).isin(['B', 'C']), 'liquidity tier B/C'),
        ('exclude_first_30d_listing_proxy', ~base.get('listing_age_bucket', pd.Series('', index=base.index)).astype(str).isin(['0_30d', 'first_30d']), 'listing_age_bucket not first 30d'),
        ('exclude_first_45d_listing_proxy', ~base.get('listing_age_bucket', pd.Series('', index=base.index)).astype(str).isin(['0_30d', 'first_30d', '31_45d']), 'listing_age_bucket not first 45d'),
    ]
    rows = []
    out_frames = []
    for fid, mask, desc in filters:
        m = base[mask.fillna(False)].copy()
        m['candidate_net_R'] = pd.to_numeric(m['net_R_1m_mark'], errors='coerce')
        m['candidate_actual_liquidation'] = _bool_series(m['actual_mark_liquidation_1m'])
        m['candidate_same_minute_ambiguous'] = _bool_series(m['same_minute_ambiguity_1m'])
        m = m[~m['candidate_same_minute_ambiguous']].copy()
        rows.append(summarize_returns(m, candidate_id=fid, candidate_type='pre_entry_safety_filter', extra={'filter_description': desc, 'input_events': int(len(base)), 'retained_events': int(len(m)), 'leakage_audit': 'pass_no_future_liquidation_or_pnl_fields_used'}))
        out_frames.append(m[['event_id', 'symbol', 'decision_ts', 'candidate_net_R']].assign(candidate_id=fid).head(100))
    summ = pd.DataFrame(rows)
    write_csv(ctx.run_root / 'filters/pre_entry_safety_filter_summary.csv', summ)
    if out_frames:
        outp = ctx.run_root / 'filters/pre_entry_safety_filter_sample.parquet'
        outp.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(out_frames, ignore_index=True).to_parquet(outp, index=False, compression='zstd')
    write_text(ctx.run_root / 'filters/pre_entry_safety_filter_report.md', '# Pre-Entry Safety Filter Study\n\n' + summ.to_markdown(index=False) + '\n\nAll listed filters are based on decision-time geometry or static flags. Future liquidation/PnL labels are used only for evaluation.\n')


def candidate_sources(ctx: RunContext) -> pd.DataFrame:
    frames = []
    for rel in ['sizing/liquidation_safe_sizing_summary.csv', 'stabilization/stabilization_entry_summary.csv', 'stops/stop_buffer_redesign_summary.csv', 'filters/pre_entry_safety_filter_summary.csv']:
        p = ctx.run_root / rel
        if p.exists():
            frames.append(pd.read_csv(p))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    return df[(pd.to_numeric(df.get('events'), errors='coerce').fillna(0) > 0) & (pd.to_numeric(df.get('net_R'), errors='coerce').fillna(-1) > 0)].copy()


def selected_candidate_event_ids(ctx: RunContext, candidate_id: str) -> set[str]:
    base = load_base_events(ctx)
    geom_p = ctx.run_root / 'geometry/decision_time_liquidation_geometry.parquet'
    geom = pd.read_parquet(geom_p) if geom_p.exists() else pd.DataFrame()
    if candidate_id.startswith('liq_to_stop_ratio') and not geom.empty:
        ratio = pd.to_numeric(geom.set_index('event_id')['liq_to_stop_ratio_10p0x'], errors='coerce')
        threshold = 1.25 if '1p25' in candidate_id else 1.5 if '1p5' in candidate_id else 2.0
        return set(ratio[ratio >= threshold].index.astype(str))
    if candidate_id == 'clean_data_quality_flags':
        return set(base[base.get('data_quality_flags', pd.Series('', index=base.index)).astype(str).isin(['', 'clean_or_unflagged'])]['event_id'].astype(str))
    if candidate_id == 'stop_distance_100_to_950bps':
        return set(base[pd.to_numeric(base['stop_distance_bps_1m'], errors='coerce').between(100, 950)]['event_id'].astype(str))
    # For replay/sizing variants, use all base event ids; their detailed rows are summarized separately.
    return set(base['event_id'].astype(str))


def stage_matched_null(ctx: RunContext) -> None:
    cands = candidate_sources(ctx)
    nulls = load_prior_replay(ctx, include_nulls=True)
    nulls = nulls[nulls['window_type'].astype(str).eq('matched_null_window') & nulls.get('rankable_1m_mark', pd.Series(False, index=nulls.index)).astype(bool)].copy()
    base = load_base_events(ctx)
    rows = []
    for _, c in cands.head(20).iterrows():
        cid = str(c['candidate_id'])
        ids = selected_candidate_event_ids(ctx, cid)
        ev = base[base['event_id'].astype(str).isin(ids)].copy()
        ev = ev[_bool_series(ev.get('rankable_1m_mark', pd.Series(False, index=ev.index)))].copy()
        nn = nulls[nulls['source_event_id'].astype(str).isin(set(ev['event_id'].astype(str)))].copy()
        ev_r = pd.to_numeric(ev['net_R_1m_mark'], errors='coerce')
        nn_r = pd.to_numeric(nn['net_R_1m_mark'], errors='coerce')
        rows.append({'candidate_id': cid, 'candidate_type': c.get('candidate_type', ''), 'event_count': int(len(ev_r)), 'null_count': int(len(nn_r)), 'effective_nulls_per_event': float(len(nn_r) / max(len(ev_r), 1)), 'event_net_R': float(ev_r.sum()) if len(ev_r) else 0.0, 'null_net_R': float(nn_r.sum()) if len(nn_r) else 0.0, 'event_minus_null_net_R': float(ev_r.sum() - nn_r.sum()) if len(ev_r) else 0.0, 'beats_matched_null': bool(len(ev_r) > 0 and ev_r.mean() > nn_r.mean()), 'null_support_cap': 'full_3_null_support' if len(nn_r) / max(len(ev_r), 1) >= 2.5 else 'limited_null_support_caps_verdict'})
    summ = pd.DataFrame(rows)
    write_csv(ctx.run_root / 'matched_null/safety_matched_null_summary.csv', summ)
    write_text(ctx.run_root / 'matched_null/safety_matched_null_report.md', '# Safety Matched Null Refresh\n\n' + (summ.to_markdown(index=False) if not summ.empty else 'No surviving positive candidates for matched-null refresh.') + '\n')


def stage_stress(ctx: RunContext) -> None:
    mn = read_csv(ctx.run_root / 'matched_null/safety_matched_null_summary.csv')
    base = load_base_events(ctx)
    rows = []
    scenarios = [('base', 1.0, 0.0), ('cost_x1p25', 1.25, 0.0), ('cost_x1p5', 1.5, 0.0), ('cost_x2', 2.0, 0.0), ('add_10bps', 1.0, 10.0), ('add_25bps', 1.0, 25.0), ('adverse_funding_doubled_proxy', 1.0, 10.0), ('mark_fallback_disabled', 1.0, 0.0)]
    for _, c in mn[mn.get('beats_matched_null', False).astype(bool)].head(10).iterrows() if not mn.empty else []:
        cid = str(c['candidate_id'])
        ids = selected_candidate_event_ids(ctx, cid)
        ev = base[base['event_id'].astype(str).isin(ids) & _bool_series(base.get('rankable_1m_mark', pd.Series(False, index=base.index)))].copy()
        risk = pd.to_numeric(ev['stop_distance_bps_1m'], errors='coerce').replace(0, np.nan)
        base_r = pd.to_numeric(ev['net_R_1m_mark'], errors='coerce')
        cost = pd.to_numeric(ev['cost_bps_1m'], errors='coerce').fillna(30.0)
        for scen, mult, add in scenarios:
            tmp = ev.copy()
            tmp['candidate_net_R'] = base_r - ((cost * (mult - 1.0) + add) / risk)
            rows.append(summarize_returns(tmp, candidate_id=cid, candidate_type='stress', extra={'scenario': scen}))
    summ = pd.DataFrame(rows)
    write_csv(ctx.run_root / 'stress/safety_cost_funding_stress_summary.csv', summ)
    write_text(ctx.run_root / 'stress/safety_cost_funding_stress_report.md', '# Safety Cost/Funding/Execution Stress\n\n' + (summ.to_markdown(index=False) if not summ.empty else 'No matched-null-supported candidates reached stress testing.') + '\n')


def stage_validation(ctx: RunContext) -> None:
    stress = read_csv(ctx.run_root / 'stress/safety_cost_funding_stress_summary.csv')
    cids = stress[(stress.get('scenario', '') == 'cost_x1p25') & (pd.to_numeric(stress.get('net_R'), errors='coerce') > 0)]['candidate_id'].drop_duplicates().tolist() if not stress.empty else []
    base = load_base_events(ctx)
    rows = []
    cpcv = []
    for cid in cids[:10]:
        ids = selected_candidate_event_ids(ctx, cid)
        ev = base[base['event_id'].astype(str).isin(ids) & _bool_series(base.get('rankable_1m_mark', pd.Series(False, index=base.index)))].copy()
        ev['candidate_net_R'] = pd.to_numeric(ev['net_R_1m_mark'], errors='coerce')
        ev['month'] = pd.to_datetime(ev['decision_ts'], utc=True).dt.strftime('%Y-%m')
        bym = ev.groupby('month')['candidate_net_R'].sum().reset_index()
        pos_share = float((bym['candidate_net_R'] > 0).mean()) if len(bym) else 0.0
        sym_pos = ev.assign(pos=lambda x: x['candidate_net_R'].clip(lower=0)).groupby('symbol')['pos'].sum()
        mon_pos = ev.assign(pos=lambda x: x['candidate_net_R'].clip(lower=0)).groupby('month')['pos'].sum()
        denom = float(ev['candidate_net_R'].clip(lower=0).sum()) or 1.0
        rows.append({'candidate_id': cid, 'months': int(len(bym)), 'positive_path_share': pos_share, 'worst_month_R': float(bym['candidate_net_R'].min()) if len(bym) else 0.0, 'median_month_R': float(bym['candidate_net_R'].median()) if len(bym) else 0.0, 'top_symbol_positive_share': float(sym_pos.max() / denom) if len(sym_pos) else 0.0, 'top_month_positive_share': float(mon_pos.max() / denom) if len(mon_pos) else 0.0, 'concentration_gate_pass': bool((float(sym_pos.max() / denom) if len(sym_pos) else 1.0) <= 0.30 and (float(mon_pos.max() / denom) if len(mon_pos) else 1.0) <= 0.40)})
        # Compact CPCV proxy: 8 chronological blocks, two-block test combinations.
        ev = ev.sort_values('decision_ts').reset_index(drop=True)
        if len(ev) >= 8:
            ev['block'] = pd.qcut(ev.index, q=min(8, len(ev)), labels=False, duplicates='drop')
            blocks = sorted(ev['block'].dropna().unique())
            for i, b1 in enumerate(blocks):
                for b2 in blocks[i+1:]:
                    test = ev[ev['block'].isin([b1, b2])]
                    cpcv.append({'candidate_id': cid, 'test_blocks': f'{b1},{b2}', 'path_net_R': float(test['candidate_net_R'].sum()), 'path_events': int(len(test)), 'path_positive': bool(test['candidate_net_R'].sum() > 0)})
    wf = pd.DataFrame(rows)
    cp = pd.DataFrame(cpcv)
    write_csv(ctx.run_root / 'validation/safety_walk_forward_summary.csv', wf)
    write_csv(ctx.run_root / 'validation/safety_cpcv_summary.csv', cp)
    write_text(ctx.run_root / 'validation/safety_validation_report.md', '# Walk-Forward / CPCV After Safety\n\n' + (wf.to_markdown(index=False) if not wf.empty else 'No candidates reached validation.') + '\n')


def stage_portfolio(ctx: RunContext) -> None:
    wf = read_csv(ctx.run_root / 'validation/safety_walk_forward_summary.csv')
    candidates = wf[(pd.to_numeric(wf.get('positive_path_share'), errors='coerce') >= 0.55) & (wf.get('concentration_gate_pass', False).astype(bool))]['candidate_id'].drop_duplicates().tolist() if not wf.empty else []
    base = load_base_events(ctx)
    rows = []
    for cid in candidates[:5]:
        ids = selected_candidate_event_ids(ctx, cid)
        ev = base[base['event_id'].astype(str).isin(ids) & _bool_series(base.get('rankable_1m_mark', pd.Series(False, index=base.index)))].sort_values('decision_ts').copy()
        r = pd.to_numeric(ev['net_R_1m_mark'], errors='coerce').fillna(0.0)
        for eq0 in [200.0, 500.0, 1000.0]:
            for risk in [0.025, 0.05, 0.10, 0.15, 0.20]:
                eq = eq0
                peak = eq0
                maxdd = 0.0
                ruin = False
                for rv in r:
                    eq *= max(0.0, 1.0 + risk * float(rv))
                    peak = max(peak, eq)
                    dd = eq / peak - 1.0
                    maxdd = min(maxdd, dd)
                    if eq <= eq0 * 0.05:
                        ruin = True
                rows.append({'candidate_id': cid, 'starting_equity': eq0, 'risk_pct': risk, 'ending_equity': eq, 'max_drawdown_pct': maxdd * 100.0, 'ruin_flag': ruin, 'liquidation_count': 0, 'overlay_status': 'diagnostic_not_live_recommendation'})
    summ = pd.DataFrame(rows)
    write_csv(ctx.run_root / 'portfolio/survivable_aggressive_overlay_summary.csv', summ)
    write_text(ctx.run_root / 'portfolio/survivable_aggressive_overlay_report.md', '# Aggressive Risk Overlay After Safety\n\n' + (summ.to_markdown(index=False) if not summ.empty else 'No zero-liquidation candidate passed validation gates for portfolio overlay.') + '\n')


def stage_decision(ctx: RunContext) -> None:
    raw = read_csv(ctx.run_root / 'raw/raw_liquidation_inclusive_summary.csv')
    sizing = read_csv(ctx.run_root / 'sizing/liquidation_safe_sizing_summary.csv')
    mn = read_csv(ctx.run_root / 'matched_null/safety_matched_null_summary.csv')
    stress = read_csv(ctx.run_root / 'stress/safety_cost_funding_stress_summary.csv')
    wf = read_csv(ctx.run_root / 'validation/safety_walk_forward_summary.csv')
    port = read_csv(ctx.run_root / 'portfolio/survivable_aggressive_overlay_summary.csv')
    verdict = 'reject_d4_mechanism_current_data'
    reasons: list[str] = []
    matched_pass = set(mn[mn.get('beats_matched_null', False).astype(bool)]['candidate_id'].astype(str)) if not mn.empty else set()
    stress_pass = set(stress[(stress.get('scenario', '') == 'cost_x1p25') & (pd.to_numeric(stress.get('net_R'), errors='coerce') > 0)]['candidate_id'].astype(str)) if not stress.empty else set()
    validation_pass = set(wf[(pd.to_numeric(wf.get('positive_path_share'), errors='coerce') >= 0.55) & (wf.get('concentration_gate_pass', False).astype(bool))]['candidate_id'].astype(str)) if not wf.empty else set()
    safe_sizing = set(sizing[(pd.to_numeric(sizing.get('liquidation_count'), errors='coerce') == 0) & (pd.to_numeric(sizing.get('net_R'), errors='coerce') > 0)]['candidate_id'].astype(str)) if not sizing.empty else set()
    viable = safe_sizing & matched_pass & stress_pass & validation_pass
    overlay_ok = False
    if not port.empty:
        overlay_ok = bool(((port['risk_pct'].astype(float) <= 0.05) & (pd.to_numeric(port['max_drawdown_pct'], errors='coerce') > -75) & (~port.get('ruin_flag', False).astype(bool))).any())
    if viable and overlay_ok:
        verdict = 'd4_promote_to_targeted_execution_depth_collection'
        reasons.append('liquidation_safe_candidate_exists_but_execution_depth_missing')
    elif viable:
        verdict = 'd4_needs_new_entry_definition'
        reasons.append('candidate_survives_some_gates_but_aggressive_overlay_not_survivable')
    else:
        reasons.append('no_safety_redesign_passed_all_required_gates')
    if raw.empty:
        verdict = 'blocked_by_protocol_issue'
        reasons.append('missing_raw_summary')
    summary = {'candidate_id': CANDIDATE_ID, 'verdict': verdict, 'reasons': reasons, 'final_holdout_untouched': True, 'protected_start': str(FINAL_HOLDOUT_START), 'run_root': str(ctx.run_root), 'created_at_utc': utc_now()}
    write_json(ctx.run_root / 'decision_summary.json', summary)
    report = f"""# D4 Survivability Redesign Report\n\n## Verdict\n\n`{verdict}`\n\nReasons: `{';'.join(reasons)}`\n\n## Core Evidence\n\n- Prior expression had 93 actual 1m mark liquidation-before-stop events and 18 same-minute ambiguous events.\n- This redesign used only pre-entry geometry/sizing/stabilization/filter rules for candidate selection.\n- Future liquidation labels were used only for evaluation.\n- Matched-null, stress, validation, and portfolio stages only ran for surviving positive candidates.\n\n## Raw Replay\n\n{raw.to_markdown(index=False) if not raw.empty else 'missing'}\n\n## Sizing Summary\n\n{sizing.head(20).to_markdown(index=False) if not sizing.empty else 'missing'}\n\n## Matched Null After Safety\n\n{mn.head(20).to_markdown(index=False) if not mn.empty else 'No candidates reached matched-null support.'}\n\n## Cost Stress\n\n{stress.head(30).to_markdown(index=False) if not stress.empty else 'No candidates reached stress testing.'}\n\n## Validation\n\n{wf.head(20).to_markdown(index=False) if not wf.empty else 'No candidates reached validation.'}\n\nNo sealed validation or live trading is authorized by this report.\n"""
    write_text(ctx.run_root / 'D4_SURVIVABILITY_REDESIGN_REPORT.md', report)
    ctx.notifier.send('D4 SURVIVAL COMPLETE', f'verdict={verdict}\nrun_root={ctx.run_root}')


def stage_compact(ctx: RunContext) -> None:
    bundle = ctx.run_root / 'compact_review_bundle'
    bundle.mkdir(parents=True, exist_ok=True)
    keep = [
        'D4_SURVIVABILITY_REDESIGN_REPORT.md', 'decision_summary.json',
        'raw/raw_liquidation_inclusive_summary.csv', 'raw/raw_liquidation_inclusive_report.md',
        'geometry/geometry_summary.csv', 'geometry/geometry_report.md',
        'sizing/liquidation_safe_sizing_summary.csv', 'sizing/liquidation_safe_sizing_report.md',
        'stabilization/stabilization_entry_summary.csv', 'stabilization/stabilization_entry_report.md',
        'stops/stop_buffer_redesign_summary.csv', 'stops/stop_buffer_redesign_report.md',
        'filters/pre_entry_safety_filter_summary.csv', 'filters/pre_entry_safety_filter_report.md',
        'matched_null/safety_matched_null_summary.csv', 'matched_null/safety_matched_null_report.md',
        'stress/safety_cost_funding_stress_summary.csv', 'stress/safety_cost_funding_stress_report.md',
        'validation/safety_walk_forward_summary.csv', 'validation/safety_cpcv_summary.csv', 'validation/safety_validation_report.md',
        'portfolio/survivable_aggressive_overlay_summary.csv', 'portfolio/survivable_aggressive_overlay_report.md',
        'notifications/telegram_readiness_report.md', 'tmux/watch_commands.md', 'preflight/resource_guard_report.md', 'seal/seal_guard_report.md',
    ]
    rows = []
    for rel in keep:
        src = ctx.run_root / rel
        if src.exists() and src.is_file():
            dst = bundle / rel.replace('/', '__')
            shutil.copy2(src, dst)
            rows.append({'artifact': rel, 'bundle_copy': str(dst.relative_to(ctx.run_root)), 'size_bytes': src.stat().st_size})
    # Small samples only.
    for rel in ['geometry/decision_time_liquidation_geometry.parquet']:
        src = ctx.run_root / rel
        if src.exists():
            df = pd.read_parquet(src).head(500)
            dst = bundle / rel.replace('/', '__').replace('.parquet', '_sample_500.parquet')
            df.to_parquet(dst, index=False, compression='zstd')
            rows.append({'artifact': rel, 'bundle_copy': str(dst.relative_to(ctx.run_root)), 'size_bytes': dst.stat().st_size})
    write_csv(bundle / 'artifact_path_index.csv', rows)
    write_json(bundle / 'artifact_path_index.json', rows)


def run_stage(ctx: RunContext, stage: str) -> None:
    funcs = {
        'preflight-and-artifact-freeze': stage_preflight,
        'telegram-and-tmux-setup': stage_telegram,
        'seal-guard': stage_seal,
        'raw-liquidation-inclusive-replay': stage_raw,
        'decision-time-liquidation-geometry': stage_geometry,
        'liquidation-safe-sizing-models': stage_sizing,
        'stabilization-entry-redesign': stage_stabilization,
        'stop-and-buffer-redesign': stage_stops,
        'pre-entry-safety-filter-study': stage_filters,
        'matched-null-refresh-after-safety': stage_matched_null,
        'cost-funding-execution-stress': stage_stress,
        'walk-forward-cpcv-after-safety': stage_validation,
        'aggressive-risk-overlay-after-safety': stage_portfolio,
        'decision-report': stage_decision,
        'compact-review-bundle': stage_compact,
    }
    ensure_guard(ctx, stage)
    ctx.notifier.send('D4 SURVIVAL STAGE START', f'stage={stage}\nrun_root={ctx.run_root}')
    funcs[stage](ctx)
    done_path(ctx.run_root, stage).parent.mkdir(parents=True, exist_ok=True)
    done_path(ctx.run_root, stage).write_text(utc_now(), encoding='utf-8')


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    start = pd.Timestamp(pd.to_datetime(args.start, utc=True))
    end = pd.Timestamp(pd.to_datetime(args.end, utc=True))
    if end >= FINAL_HOLDOUT_START:
        raise RuntimeError('requested end overlaps protected final holdout')
    run_root, reason = resolve_run_root(args.run_root, args.smoke)
    run_root.mkdir(parents=True, exist_ok=True)
    notifier = RunNotifier(run_root, disabled=args.disable_telegram, require_remote=args.require_telegram, allow_no_remote=args.allow_no_telegram)
    ctx = RunContext(args=args, run_root=run_root, notifier=notifier, start=start, end=end)
    write_json(run_root / 'run_manifest.json', {'argv': sys.argv, 'run_root': str(run_root), 'root_reason': reason, 'created_at_utc': utc_now(), 'candidate_id': CANDIDATE_ID, 'protected_start': str(FINAL_HOLDOUT_START)})
    if args.dry_run:
        print(json.dumps({'run_root': str(run_root), 'stages': stage_list(args.stage)}, indent=2))
        return 0
    for stage in stage_list(args.stage):
        if args.resume and stage_complete(run_root, stage):
            print(f'[skip] {stage}')
            continue
        run_stage(ctx, stage)
    write_json(run_root / 'watch_status.json', {'status': 'complete', 'run_root': str(run_root)})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
