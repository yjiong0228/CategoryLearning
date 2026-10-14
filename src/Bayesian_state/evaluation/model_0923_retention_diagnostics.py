"""Rule residence and counterevidence diagnostics independent of the exit policy.

A rule below the prespecified geometric accuracy threshold is incompatible for
this retrospective diagnostic. Active occupancy and contribution to choices are
reported separately. Multiple evidence windows are scoring sensitivities only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def retention_diagnostics(a: dict, rules: list[int], condition: int, late_window: int,
                          windows: list[int], horizons: list[int]) -> tuple[list[dict], list[dict], list[dict]]:
    """Return per-path metrics, incompatible residences and clearance summaries."""
    n, paths, _ = a['rule'].shape
    if not windows or not horizons or min(windows+horizons) < 1:
        raise ValueError('Evidence windows and residence horizons must be positive')
    active = a['rule'] >= 0
    incompatible = active & ~np.isin(a['rule'], rules)
    categories = a['prediction'].shape[-1]
    neutral = (np.where(a['feedback']==0,.5,.25) if condition==3 else
               np.where(a['feedback']==1,1/categories,1-1/categories))
    evidence = np.log(np.maximum(a['evidence'],np.finfo(float).tiny))-np.log(neutral[:,:,None])
    late = slice(max(0,n-late_window),n)
    per_path=[]
    episodes=[]
    for path in range(paths):
        per_path.append(dict(
            mean_active_candidates=float(active[:,path].sum(axis=1).mean()),
            late_incompatible_candidates=float(incompatible[late,path].sum(axis=1).mean()),
            late_incompatible_contribution=float(np.where(incompatible[late,path],a['contribution'][late,path],0.).sum(axis=1).mean()),
            total_autonomous_exits=int((a['exits'][:,path]>=0).sum()),
            total_capacity_replacements=int((a['capacity_dropped'][:,path]>=0).sum()),
            total_old_rule_exposure=int(active[:-1,path].sum())))
        opened={}
        for trial in range(n+1):
            ids={int(a['rule'][trial,path,slot]):int(slot) for slot in np.flatnonzero(incompatible[trial,path])} if trial<n else {}
            for h in set(opened)-set(ids):
                state=opened.pop(h)
                cause=('right_censored' if trial==n else 'autonomous_exit' if h in a['exits'][trial,path] else
                       'capacity' if a['capacity_dropped'][trial,path]==h else 'unexplained')
                if cause=='unexplained':
                    raise ValueError('A rule disappeared without a logged cause')
                row=dict(path=path,rule=h,start_trial=state['start']+1,last_trial=trial,
                         duration=trial-state['start'],right_censored=trial==n,cause=cause)
                for window in windows:
                    first=state['first'][window]
                    row[f'first_negative_w{window}']=first+1 if first is not None else np.nan
                episodes.append(row)
            for h,slot in ids.items():
                state=opened.setdefault(h,dict(start=trial,history=[],first={w:None for w in windows}))
                state['history']=(state['history']+[float(evidence[trial,path,slot])])[-max(windows):]
                for w in windows:
                    if state['first'][w] is None and sum(state['history'][-w:])<0:
                        state['first'][w]=trial
    summary=[]
    frame=pd.DataFrame(episodes)
    for window in windows:
        for horizon in horizons:
            first=frame[f'first_negative_w{window}'] if len(frame) else pd.Series(dtype=float)
            eligible=first.notna() & (first+horizon<=n)
            # last_trial is 1-based last presence; the next transition is last+1.
            cleared=(frame.last_trial+1-first<=horizon)&~frame.right_censored if len(frame) else pd.Series(dtype=bool)
            count=int(eligible.sum())
            summary.append(dict(window=window,horizon=horizon,eligible=count,
                cleared=int((eligible&cleared).sum()),fraction=float(cleared[eligible].mean()) if count else np.nan))
    for horizon in horizons:
        eligible=(frame.start_trial+horizon-1<=n) if len(frame) else pd.Series(dtype=bool)
        retained=(frame.duration>=horizon) if len(frame) else pd.Series(dtype=bool)
        for path,row in enumerate(per_path):
            selected=eligible&(frame.path==path) if len(frame) else eligible
            row[f'incompatible_residences_eligible_{horizon}']=int(selected.sum())
            row[f'incompatible_residences_retained_{horizon}']=int((selected&retained).sum())
    return per_path,episodes,summary
