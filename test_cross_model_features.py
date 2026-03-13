"""
Cross-model feature test: does adding 4h + daily model probabilities improve the 1h model?

Aanpak (stacking):
  - Voor elke 1h candle op T: gebruik de 4h-modelproba van de meest recente afgesloten
    4h-candle vóór T (geen look-ahead bias — shift(1) op 4h index).
  - Zelfde voor dagmodel: meest recente dag-candle vóór T.
  - Voeg toe als signal_4h_proba en signal_daily_proba.
  - Hertrain 1h model en vergelijk AUC + Sharpe.

Gebruik: python test_cross_model_features.py [BTCUSDT|ETHUSDT]
"""

import sys
import json
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

import config
import config_4h
import config_daily

symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
print(f"\n{'='*60}")
print(f"Cross-model feature test — {symbol}")
print(f"{'='*60}")

# ── Laad bestaande 1h features ─────────────────────────────────────────────
feat_path = config.DATA_DIR / f"{symbol}_features.parquet"
print(f"\n1. Laad 1h feature matrix: {feat_path}")
df_1h = pd.read_parquet(feat_path)
print(f"   {len(df_1h)} rijen ({df_1h.index[0].date()} → {df_1h.index[-1].date()})")

# ── Genereer 4h model probabilities ────────────────────────────────────────
print("\n2. Genereer 4h model probabilities...")
from src.data_fetcher import load_ohlcv
from src.features_4h import build_features_4h

df_4h_ohlcv = load_ohlcv(symbol=symbol, interval="4h")
df_4h_feat  = build_features_4h(df_4h_ohlcv, symbol=symbol)

model_4h_path = config_4h.symbol_path_4h(symbol, "model.pkl")
if model_4h_path.exists():
    model_4h   = joblib.load(model_4h_path)
    feat_4h    = [c for c in config_4h.FEATURE_COLS_4H_MODEL if c in df_4h_feat.columns]
    proba_4h   = model_4h.predict_proba(df_4h_feat[feat_4h])[:, 1]
    sr_4h      = pd.Series(proba_4h, index=df_4h_feat.index, name="signal_4h_proba")
    # Shift 1 candle: T=04:00 gebruikt de 4h-proba van T=00:00 (al afgesloten)
    sr_4h_shifted = sr_4h.shift(1)
    print(f"   4h proba: {len(sr_4h)} waarden, mean={sr_4h.mean():.3f}")
else:
    print(f"   GEEN 4h model gevonden: {model_4h_path}")
    sr_4h_shifted = pd.Series(dtype=float, name="signal_4h_proba")

# ── Genereer daily model probabilities ─────────────────────────────────────
print("\n3. Genereer daily model probabilities...")
from src.features_daily import build_features_daily

df_1d_ohlcv = load_ohlcv(symbol=symbol, interval="1d")
df_1d_feat  = build_features_daily(df_1d_ohlcv, symbol=symbol)

model_1d_path = config_daily.symbol_path_daily(symbol, "model.pkl")
if model_1d_path.exists():
    model_1d   = joblib.load(model_1d_path)
    feat_1d    = [c for c in config_daily.FEATURE_COLS_DAILY if c in df_1d_feat.columns]
    proba_1d   = model_1d.predict_proba(df_1d_feat[feat_1d])[:, 1]
    sr_1d      = pd.Series(proba_1d, index=df_1d_feat.index, name="signal_daily_proba")
    # Shift 1 dag: T=2024-06-03 00:00 (1h) gebruikt de dagproba van 2024-06-02
    sr_1d_shifted = sr_1d.shift(1)
    print(f"   daily proba: {len(sr_1d)} waarden, mean={sr_1d.mean():.3f}")
else:
    print(f"   GEEN daily model gevonden: {model_1d_path}")
    sr_1d_shifted = pd.Series(dtype=float, name="signal_daily_proba")

# ── Voeg toe aan 1h feature matrix (forward-fill) ──────────────────────────
print("\n4. Merge cross-model features met 1h matrix...")

# 4h: merge_asof zodat elke 1h-rij de meest recente (niet-lookahead) 4h proba krijgt
idx_name = df_1h.index.name or "index"

if len(sr_4h_shifted.dropna()) > 0:
    sr_4h_df = sr_4h_shifted.dropna().reset_index()
    sr_4h_df.columns = ["timestamp", "signal_4h_proba"]
    df_1h_reset = df_1h.reset_index()
    df_1h_reset = pd.merge_asof(
        df_1h_reset.sort_values(idx_name),
        sr_4h_df.sort_values("timestamp"),
        left_on=idx_name, right_on="timestamp",
        direction="backward",
    )
    df_1h_reset = df_1h_reset.drop(columns=["timestamp"], errors="ignore")
    df_1h_merged = df_1h_reset.set_index(idx_name)
else:
    df_1h_merged = df_1h.copy()
    df_1h_merged["signal_4h_proba"] = np.nan

# Daily: merge_asof
if len(sr_1d_shifted.dropna()) > 0:
    sr_1d_df = sr_1d_shifted.dropna().reset_index()
    sr_1d_df.columns = ["timestamp", "signal_daily_proba"]
    df_1h_merged_reset = df_1h_merged.reset_index()
    idx_name2 = df_1h_merged_reset.columns[0]  # eerste kolom na reset_index
    df_1h_merged_reset = pd.merge_asof(
        df_1h_merged_reset.sort_values(idx_name2),
        sr_1d_df.sort_values("timestamp"),
        left_on=idx_name2, right_on="timestamp",
        direction="backward",
    )
    df_1h_merged_reset = df_1h_merged_reset.drop(columns=["timestamp"], errors="ignore")
    df_1h_merged = df_1h_merged_reset.set_index(idx_name2)
else:
    df_1h_merged["signal_daily_proba"] = np.nan

# Verwijder rijen waar cross-model features NaN zijn
n_before = len(df_1h_merged)
df_1h_merged = df_1h_merged.dropna(subset=["signal_4h_proba", "signal_daily_proba"])
print(f"   Rijen voor merge: {n_before} | Na dropna: {len(df_1h_merged)}")
print(f"   signal_4h_proba: {df_1h_merged['signal_4h_proba'].mean():.3f} ± "
      f"{df_1h_merged['signal_4h_proba'].std():.3f}")
print(f"   signal_daily_proba: {df_1h_merged['signal_daily_proba'].mean():.3f} ± "
      f"{df_1h_merged['signal_daily_proba'].std():.3f}")

# ── Basislijn: 1h model ZONDER cross-model features ───────────────────────
print("\n5. Vergelijk baseline vs. cross-model features...")

from src.model import load_model, time_split
from src.backtest import run_backtest, compute_metrics, load_optimal_threshold
import lightgbm as lgb

# Gebruik dezelfde train/val/test split als de baseline
base_df     = df_1h_merged  # zelfde rijen als na filtering
feat_base   = config.FEATURE_COLS
feat_cross  = config.FEATURE_COLS + ["signal_4h_proba", "signal_daily_proba"]

# Time-based split (aanpassing voor df_1h_merged)
n           = len(base_df)
test_rows   = config.TEST_SIZE_DAYS * 24
val_rows    = config.VALIDATION_SIZE_DAYS * 24
train_end   = n - test_rows - val_rows
val_end     = n - test_rows

train_df    = base_df.iloc[:train_end]
val_df      = base_df.iloc[train_end:val_end]
test_df     = base_df.iloc[val_end:]

print(f"   Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# Laad lgb params
lgb_params_path = config.DATA_DIR / "lgb_best_params.json"
with open(lgb_params_path) as f:
    lgb_params = json.load(f)

def _train_lgb(train, val, feats):
    """Train LightGBM en retourneer model + val AUC."""
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score

    X_tr, y_tr = train[feats], train["target"]
    X_v,  y_v  = val[feats],   val["target"]

    model = LGBMClassifier(
        n_estimators=lgb_params.get("n_estimators", 407),
        max_depth=lgb_params.get("max_depth", 5),
        learning_rate=lgb_params.get("learning_rate", 0.028),
        subsample=lgb_params.get("subsample", 0.65),
        colsample_bytree=lgb_params.get("colsample_bytree", 0.477),
        min_child_samples=lgb_params.get("min_child_samples", 175),
        reg_alpha=lgb_params.get("reg_alpha", 0.1),
        reg_lambda=lgb_params.get("reg_lambda", 1.0),
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    model.fit(X_tr, y_tr)
    val_auc = roc_auc_score(y_v, model.predict_proba(X_v)[:, 1])
    return model, val_auc

def _eval_model(model, test, feats, label):
    """Evalueer model op test set en druk resultaten af."""
    from sklearn.metrics import roc_auc_score

    X_te, y_te = test[feats], test["target"]
    probas = model.predict_proba(X_te)[:, 1]
    auc    = roc_auc_score(y_te, probas)

    long_thr, short_thr = load_optimal_threshold(symbol=symbol)
    results = run_backtest(test, probas, threshold=long_thr,
                           threshold_short=short_thr, use_short=(short_thr > 0))
    metrics = compute_metrics(results)

    print(f"\n   [{label}]")
    print(f"   Test AUC  : {auc:.4f}")
    print(f"   Sharpe    : {metrics.get('sharpe_ratio', 0):.3f}")
    print(f"   Return    : {metrics.get('total_return', 0)*100:.1f}%")
    print(f"   Trades    : {metrics.get('n_trades', 0)}")
    return auc, metrics

# ── Baseline ───────────────────────────────────────────────────────────────
feats_base_avail = [c for c in feat_base if c in train_df.columns]
model_base, val_auc_base = _train_lgb(train_df, val_df, feats_base_avail)
auc_base, met_base = _eval_model(model_base, test_df, feats_base_avail, "BASELINE (geen cross-model)")

# ── Met cross-model features ───────────────────────────────────────────────
feats_cross_avail = [c for c in feat_cross if c in train_df.columns]
model_cross, val_auc_cross = _train_lgb(train_df, val_df, feats_cross_avail)
auc_cross, met_cross = _eval_model(model_cross, test_df, feats_cross_avail, "MET cross-model features")

# ── Samenvatting ───────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"SAMENVATTING — {symbol}")
print(f"{'='*60}")
delta_auc    = auc_cross - auc_base
delta_sharpe = met_cross.get("sharpe_ratio",0) - met_base.get("sharpe_ratio",0)
delta_ret    = (met_cross.get("total_return",0) - met_base.get("total_return",0)) * 100

print(f"{'':20s}  {'Baseline':>12s}  {'+ Cross-model':>14s}  {'Delta':>8s}")
print(f"{'Val AUC':20s}  {val_auc_base:12.4f}  {val_auc_cross:14.4f}  {val_auc_cross-val_auc_base:+8.4f}")
print(f"{'Test AUC':20s}  {auc_base:12.4f}  {auc_cross:14.4f}  {delta_auc:+8.4f}")
print(f"{'Sharpe':20s}  {met_base.get('sharpe_ratio',0):12.3f}  {met_cross.get('sharpe_ratio',0):14.3f}  {delta_sharpe:+8.3f}")
print(f"{'Return':20s}  {met_base.get('total_return',0)*100:11.1f}%  {met_cross.get('total_return',0)*100:13.1f}%  {delta_ret:+7.1f}%")
print(f"{'Trades':20s}  {met_base.get('n_trades',0):12.0f}  {met_cross.get('n_trades',0):14.0f}")

verdict = "✅ VERBETERING" if delta_auc > 0.002 or delta_sharpe > 0.5 else "❌ GEEN verbetering"
print(f"\nVerdict: {verdict}")
if delta_auc > 0.002:
    print("  → Overweeg signal_4h_proba + signal_daily_proba toe te voegen aan FEATURE_COLS")
