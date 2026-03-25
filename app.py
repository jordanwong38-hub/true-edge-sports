import math
from datetime import date
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# =========================================================
# PAGE SETUP
# =========================================================
st.set_page_config(page_title="True Edge Sports — MLB", layout="wide")

st.title("True Edge Sports — MLB (2026)")
st.caption(
    "Public MLB analytics dashboard (V1). "
    "Hitters + Pitchers + Trends + Matchups + Props + Profiles + Live Standings + Today’s Schedule."
)

# =========================================================
# PATHS
# =========================================================
HITTERS_PATH = Path("data/mlb/clean/hitters_2025_clean.csv")
PITCHERS_PATH = Path("data/mlb/clean/pitchers_2025_clean.csv")
TRENDS_DIR = Path("data/mlb/clean")

# =========================================================
# HELPERS
# =========================================================
def fmt(val, kind="num", decimals=1):
    if pd.isna(val):
        return "—"
    if kind == "pct":
        return f"{float(val):.{decimals}f}%"
    if kind == "avg":
        return f"{float(val):.3f}"
    if kind == "int":
        return f"{int(val)}"
    if kind == "float1":
        return f"{float(val):.1f}"
    if kind == "float2":
        return f"{float(val):.2f}"
    if kind == "float3":
        return f"{float(val):.3f}"
    return str(val)


def ordinal(n):
    if pd.isna(n):
        return "—"
    n = int(round(float(n)))
    if 10 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def delta_num(player_val, league_val):
    if pd.isna(player_val) or league_val is None:
        return None
    return float(player_val) - float(league_val)


def fmt_delta(d, kind="float3", pct_points=False):
    if d is None:
        return None
    if pct_points:
        return f"{d:+.1f} pp"
    if kind == "float2":
        return f"{d:+.2f}"
    return f"{d:+.3f}"


def safe_mean(df: pd.DataFrame, col: str):
    if col not in df.columns or df.empty:
        return None
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        return None
    return float(s.mean())


def top_leader(df: pd.DataFrame, metric_col: str, higher_better: bool = True):
    if df.empty or metric_col not in df.columns:
        return ("—", None)
    tmp = df.copy()
    tmp["_metric"] = pd.to_numeric(tmp[metric_col], errors="coerce")
    tmp = tmp.dropna(subset=["_metric"])
    if tmp.empty:
        return ("—", None)
    best = tmp.sort_values("_metric", ascending=not higher_better).iloc[0]
    return best.get("player_name", "—"), best.get("_metric")


def rank_and_percentile(df: pd.DataFrame, col: str, higher_better: bool = True):
    s = pd.to_numeric(df[col], errors="coerce")
    rank = s.rank(ascending=not higher_better, method="min")
    pct = s.rank(pct=True, ascending=not higher_better) * 100
    return rank, pct


def make_rank_table(df: pd.DataFrame, player_row: pd.Series, cols: list[str], lower_better_cols: set[str]):
    rows = []
    n = len(df)

    label_map = {
        "pa": "PA",
        "batting_avg": "AVG",
        "hit": "H",
        "home_run": "HR",
        "rbi": "RBI",
        "strikeout": "SO",
        "walk": "BB",
        "slg_percent": "SLG",
        "on_base_percent": "OBP",
        "on_base_plus_slg": "OPS",
        "woba": "wOBA",
        "hard_hit_percent": "Hard Hit%",
        "k_percent": "K%",
        "bb_percent": "BB%",
        "ip": "IP",
        "era": "ERA",
        "whip": "WHIP",
        "k_minus_bb_percent": "K-BB%",
        "whiff_percent": "Whiff%",
    }

    for c in cols:
        if c not in df.columns:
            continue
        val = player_row.get(c)
        if pd.isna(val):
            continue

        higher_better = c not in lower_better_cols
        r, p = rank_and_percentile(df, c, higher_better=higher_better)

        idx = player_row.name
        rank_val = r.loc[idx] if idx in r.index else None
        pct_val = p.loc[idx] if idx in p.index else None

        if "percent" in c or c.endswith("_percent"):
            value_str = fmt(val, "pct", 1)
        elif c in {"pa", "hit", "home_run", "rbi", "strikeout", "walk"}:
            value_str = fmt(val, "int")
        elif c == "ip":
            value_str = fmt(val, "float1")
        elif c in {"era", "whip"}:
            value_str = fmt(val, "float2")
        else:
            value_str = fmt(val, "float3")

        rows.append(
            {
                "Stat": label_map.get(c, c),
                "Value": value_str,
                "Rank": f"{int(rank_val)} / {n}" if pd.notna(rank_val) else "—",
                "Percentile": ordinal(pct_val) if pd.notna(pct_val) else "—",
            }
        )

    return pd.DataFrame(rows)


def format_df_for_display(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    out = df.copy()

    rename_hitters = {
        "player_name": "Player",
        "team": "Team",
        "pa": "PA",
        "batting_avg": "AVG",
        "hit": "H",
        "home_run": "HR",
        "rbi": "RBI",
        "strikeout": "SO",
        "walk": "BB",
        "on_base_percent": "OBP",
        "slg_percent": "SLG",
        "on_base_plus_slg": "OPS",
        "woba": "wOBA",
        "hard_hit_percent": "Hard Hit%",
        "k_percent": "K%",
        "bb_percent": "BB%",
    }

    rename_pitchers = {
        "player_name": "Player",
        "team": "Team",
        "ip": "IP",
        "era": "ERA",
        "whip": "WHIP",
        "strikeout": "SO",
        "walk": "BB",
        "k_percent": "K%",
        "bb_percent": "BB%",
        "k_minus_bb_percent": "K-BB%",
        "whiff_percent": "Whiff%",
        "hard_hit_percent": "Hard Hit% Allowed",
    }

    if kind.startswith("hitters"):
        out = out.rename(columns={k: v for k, v in rename_hitters.items() if k in out.columns})
        for c in ["AVG", "OBP", "SLG", "OPS", "wOBA"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
        for c in ["Hard Hit%", "K%", "BB%"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").map(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
        for c in ["PA", "H", "HR", "RBI", "SO", "BB"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").map(lambda x: f"{int(x)}" if pd.notna(x) else "—")

    elif kind.startswith("pitchers"):
        out = out.rename(columns={k: v for k, v in rename_pitchers.items() if k in out.columns})
        if "IP" in out.columns:
            out["IP"] = pd.to_numeric(out["IP"], errors="coerce").map(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
        for c in ["ERA", "WHIP"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        for c in ["SO", "BB"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").map(lambda x: f"{int(x)}" if pd.notna(x) else "—")
        for c in ["K%", "BB%", "K-BB%", "Whiff%", "Hard Hit% Allowed"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").map(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")

    return out


def format_trends_hitters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {
        "player_name": "Player",
        "team": "Team",
        "plateAppearances": "PA",
        "hits": "H",
        "homeRuns": "HR",
        "rbi": "RBI",
        "baseOnBalls": "BB",
        "strikeOuts": "SO",
        "avg": "AVG",
        "obp": "OBP",
        "slg": "SLG",
        "ops": "OPS",
        "ops_delta_vs_season": "ΔOPS vs Season",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})

    for c in ["PA", "H", "HR", "RBI", "BB", "SO"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").map(lambda x: f"{int(x)}" if pd.notna(x) else "—")

    for c in ["AVG", "OBP", "SLG", "OPS", "ΔOPS vs Season"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").map(
                lambda x: f"{x:+.3f}" if c == "ΔOPS vs Season" and pd.notna(x)
                else (f"{x:.3f}" if pd.notna(x) else "—")
            )
    return out


def format_trends_pitchers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {
        "player_name": "Player",
        "team": "Team",
        "inningsPitched": "IP",
        "strikeOuts": "SO",
        "baseOnBalls": "BB",
        "era": "ERA",
        "whip": "WHIP",
        "era_delta_vs_season": "ΔERA vs Season",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})

    if "IP" in out.columns:
        out["IP"] = pd.to_numeric(out["IP"], errors="coerce").map(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
    for c in ["SO", "BB"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").map(lambda x: f"{int(x)}" if pd.notna(x) else "—")
    for c in ["ERA", "WHIP"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    if "ΔERA vs Season" in out.columns:
        out["ΔERA vs Season"] = pd.to_numeric(out["ΔERA vs Season"], errors="coerce").map(
            lambda x: f"{x:+.2f}" if pd.notna(x) else "—"
        )
    return out


def top_delta_list(df: pd.DataFrame, metric: str, league_avg: float | None, n: int = 5, higher_better: bool = True):
    if df.empty or league_avg is None or metric not in df.columns:
        return pd.DataFrame(columns=["Player", "Value", "Delta"])
    tmp = df[["player_name", metric]].copy()
    tmp[metric] = pd.to_numeric(tmp[metric], errors="coerce")
    tmp = tmp.dropna(subset=[metric])
    if tmp.empty:
        return pd.DataFrame(columns=["Player", "Value", "Delta"])

    if higher_better:
        tmp["delta"] = tmp[metric] - league_avg
    else:
        tmp["delta"] = league_avg - tmp[metric]

    tmp = tmp.sort_values("delta", ascending=False).head(n)
    return pd.DataFrame({"Player": tmp["player_name"].values, "Value": tmp[metric].values, "Delta": tmp["delta"].values})


def format_hot_list(df: pd.DataFrame, value_kind: str, delta_pp: bool = False):
    out = df.copy()
    if out.empty:
        return out

    if value_kind == "pct":
        out["Value"] = out["Value"].map(lambda x: fmt(x, "pct", 1))
    elif value_kind == "float2":
        out["Value"] = out["Value"].map(lambda x: fmt(x, "float2"))
    else:
        out["Value"] = out["Value"].map(lambda x: fmt(x, "float3"))

    if delta_pp:
        out["Delta"] = out["Delta"].map(lambda x: fmt_delta(x, pct_points=True) if pd.notna(x) else "—")
    else:
        decimals = 2 if value_kind == "float2" else 3
        out["Delta"] = out["Delta"].map(lambda x: f"{x:+.{decimals}f}" if pd.notna(x) else "—")
    return out


def percentile_rank(series, higher_is_better=True):
    s = pd.to_numeric(series, errors="coerce")
    
    if higher_is_better:
        return s.rank(pct=True)
    else:
        return 1 - s.rank(pct=True)


def add_hitter_prop_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    hh = percentile_rank(out["hard_hit_percent"], True) if "hard_hit_percent" in out.columns else 0
    slg = percentile_rank(out["slg_percent"], True) if "slg_percent" in out.columns else 0
    woba = percentile_rank(out["woba"], True) if "woba" in out.columns else 0
    hr = percentile_rank(out["home_run"], True) if "home_run" in out.columns else 0
    avg = percentile_rank(out["batting_avg"], True) if "batting_avg" in out.columns else 0
    obp = percentile_rank(out["on_base_percent"], True) if "on_base_percent" in out.columns else 0
    k_inv = percentile_rank(out["k_percent"], False) if "k_percent" in out.columns else 0

    out["hr_signal"] = (0.35 * hh + 0.30 * slg + 0.20 * woba + 0.15 * hr) * 100
    out["hits_signal"] = (0.30 * avg + 0.25 * obp + 0.20 * woba + 0.15 * hh + 0.10 * k_inv) * 100
    return out


def add_pitcher_prop_scores(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    kbb = percentile_rank(out["k_minus_bb_percent"], True) if "k_minus_bb_percent" in out.columns else 0
    whiff = percentile_rank(out["whiff_percent"], True) if "whiff_percent" in out.columns else 0
    k = percentile_rank(out["strikeout"], True) if "strikeout" in out.columns else 0
    bb_inv = percentile_rank(out["bb_percent"], False) if "bb_percent" in out.columns else 0
    hh_inv = percentile_rank(out["hard_hit_percent"], False) if "hard_hit_percent" in out.columns else 0
    out["k_signal"] = (0.35 * kbb + 0.30 * whiff + 0.15 * k + 0.10 * bb_inv + 0.10 * hh_inv) * 100
    return out


def zscore_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    std = s.std()
    if pd.isna(std) or std == 0:
        return pd.Series([0] * len(s), index=s.index)
    return (s - s.mean()) / std


def build_team_strengths(hitters_df: pd.DataFrame, pitchers_df: pd.DataFrame) -> pd.DataFrame:
    if "team" not in hitters_df.columns or "team" not in pitchers_df.columns:
        return pd.DataFrame()

    hit = hitters_df.groupby("team", dropna=True).agg(
        team_ops=("on_base_plus_slg", "mean"),
        team_woba=("woba", "mean"),
        team_hh=("hard_hit_percent", "mean"),
    ).reset_index()

    pitch_aggs = {
        "team_kbb": ("k_minus_bb_percent", "mean"),
        "team_whiff": ("whiff_percent", "mean"),
        "team_hha": ("hard_hit_percent", "mean"),
    }
    if "era" in pitchers_df.columns:
        pitch_aggs["team_era"] = ("era", "mean")
    if "whip" in pitchers_df.columns:
        pitch_aggs["team_whip"] = ("whip", "mean")

    pitch = pitchers_df.groupby("team", dropna=True).agg(**pitch_aggs).reset_index()
    team = hit.merge(pitch, on="team", how="outer")

    team["offense_score"] = (
        zscore_series(team["team_ops"]) +
        zscore_series(team["team_woba"]) +
        zscore_series(team["team_hh"])
    ) / 3

    pitch_parts = [
        zscore_series(team["team_kbb"]),
        zscore_series(team["team_whiff"]),
        -zscore_series(team["team_hha"]),
    ]
    if "team_era" in team.columns:
        pitch_parts.append(-zscore_series(team["team_era"]))
    if "team_whip" in team.columns:
        pitch_parts.append(-zscore_series(team["team_whip"]))

    team["pitching_score"] = sum(pitch_parts) / len(pitch_parts)
    team["team_score"] = 0.55 * team["offense_score"] + 0.45 * team["pitching_score"]
    return team


def compute_pitcher_bonus(pitchers_df: pd.DataFrame) -> pd.DataFrame:
    df = pitchers_df.copy()
    if df.empty:
        return pd.DataFrame(columns=["player_name", "pitcher_bonus"])

    parts = [
        percentile_rank(df["k_minus_bb_percent"], True) if "k_minus_bb_percent" in df.columns else 0,
        percentile_rank(df["whiff_percent"], True) if "whiff_percent" in df.columns else 0,
        percentile_rank(df["hard_hit_percent"], False) if "hard_hit_percent" in df.columns else 0,
    ]
    if "era" in df.columns:
        parts.append(percentile_rank(df["era"], False))
    if "whip" in df.columns:
        parts.append(percentile_rank(df["whip"], False))

    bonus = sum(parts) / len(parts)
    df["pitcher_bonus"] = (bonus - 0.5) * 0.60
    return df[["player_name", "pitcher_bonus"]]


def logistic(x):
    return 1 / (1 + math.exp(-x))


def get_biggest_movers(df: pd.DataFrame, stat: str, top_n: int = 10, ascending: bool = False) -> pd.DataFrame:
    out = df.copy()
    if stat not in out.columns:
        return pd.DataFrame(columns=["player_name", "team", stat])

    out[stat] = pd.to_numeric(out[stat], errors="coerce")
    out = out.dropna(subset=[stat]).sort_values(stat, ascending=ascending)

    cols = [c for c in ["player_name", "team", stat] if c in out.columns]
    return out[cols].head(top_n)

def get_teams_playing_today(schedule_df: pd.DataFrame) -> set[str]:
    if schedule_df.empty:
        return set()
    teams = set(schedule_df["Away"].dropna().tolist()) | set(schedule_df["Home"].dropna().tolist())
    return teams

def project_strikeouts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    BASE_K = 5.0
    out["proj_k"] = BASE_K + ((out["k_signal"] - 50) / 10) * 0.5
    return out


def get_k_lean(row) -> str:
    sig = row.get("k_signal")
    if pd.isna(sig):
        return "Pass"
    if sig >= 80:
        return "🔥 Strong Over"
    elif sig >= 70:
        return "Lean Over"
    elif sig >= 60:
        return "Slight Lean"
    return "Pass"


def project_home_run_props(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Probability-style estimate, not literal projected HR total
    # 50 signal ~= 10% HR chance
    # Every 10 points above 50 adds ~4%
    out["proj_hr_chance"] = 0.10 + ((out["hr_signal"] - 50) / 10) * 0.04
    out["proj_hr_chance"] = out["proj_hr_chance"].clip(lower=0.02, upper=0.35)

    return out


def get_hr_lean(row) -> str:
    sig = row.get("hr_signal")
    if pd.isna(sig):
        return "Pass"
    if sig >= 85:
        return "🔥 Strong HR Look"
    elif sig >= 75:
        return "Good HR Look"
    elif sig >= 65:
        return "Slight HR Lean"
    return "Pass"


def project_hits_props(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Average everyday hitter baseline
    BASE_HITS = 1.0
    out["proj_hits"] = BASE_HITS + ((out["hits_signal"] - 50) / 10) * 0.12
    out["proj_hits"] = out["proj_hits"].clip(lower=0.6, upper=2.2)

    return out


def get_hits_lean(row) -> str:
    sig = row.get("hits_signal")
    if pd.isna(sig):
        return "Pass"
    if sig >= 85:
        return "🔥 Strong Over"
    elif sig >= 75:
        return "Lean Over"
    elif sig >= 65:
        return "Slight Lean"
    return "Pass"


# =========================================================
# LOADERS
# =========================================================
@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.resolve()}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


# =========================================================
# MLB API
# =========================================================
@st.cache_data(ttl=3600)
def fetch_standings(season: int) -> pd.DataFrame:
    url = "https://statsapi.mlb.com/api/v1/standings"
    params = {"leagueId": "103,104", "season": season, "standingsTypes": "regularSeason"}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    rows = []
    for record in data.get("records", []):
        division = record.get("division", {}).get("name", "")
        league = record.get("league", {}).get("name", "")
        for tr in record.get("teamRecords", []):
            rows.append({
                "League": league,
                "Division": division,
                "Team": tr.get("team", {}).get("name", ""),
                "W": tr.get("wins"),
                "L": tr.get("losses"),
                "Win%": float(tr.get("winningPercentage")) if tr.get("winningPercentage") is not None else None,
                "GB": tr.get("gamesBack"),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["League", "Division", "Win%"], ascending=[True, True, False])
    return df


@st.cache_data(ttl=900)
def fetch_schedule_raw(game_date: str) -> pd.DataFrame:
    url = "https://statsapi.mlb.com/api/v1/schedule"
    params = {"sportId": 1, "date": game_date, "hydrate": "team,linescore,probablePitcher"}
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()

    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            away_full = g.get("teams", {}).get("away", {}).get("team", {}).get("name", "")
            home_full = g.get("teams", {}).get("home", {}).get("team", {}).get("name", "")
            games.append({
                "Start (ISO)": g.get("gameDate", ""),
                "Away": away_full,
                "Home": home_full,
                "Status": g.get("status", {}).get("detailedState", ""),
                "Venue": g.get("venue", {}).get("name", ""),
                "Away Probable": g.get("teams", {}).get("away", {}).get("probablePitcher", {}).get("fullName", "—"),
                "Home Probable": g.get("teams", {}).get("home", {}).get("probablePitcher", {}).get("fullName", "—"),
            })
    return pd.DataFrame(games)


# =========================================================
# LOAD DATA
# =========================================================
try:
    hitters = load_csv(HITTERS_PATH)
    pitchers = load_csv(PITCHERS_PATH)
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

hitters = coerce_numeric(hitters, [
    "player_id", "pa", "batting_avg", "hit", "home_run", "rbi", "strikeout", "walk",
    "slg_percent", "on_base_percent", "on_base_plus_slg", "woba",
    "hard_hit_percent", "k_percent", "bb_percent"
])

pitchers = coerce_numeric(pitchers, [
    "player_id", "ip", "era", "whip", "strikeout", "walk", "k_percent",
    "bb_percent", "hard_hit_percent", "whiff_percent", "k_minus_bb_percent"
])

STAT_HELP = {
    "AVG": "Batting average.",
    "H": "Hits.",
    "HR": "Home runs.",
    "RBI": "Runs batted in.",
    "SO": "Strikeouts.",
    "BB": "Walks.",
    "OBP": "On-base percentage.",
    "SLG": "Slugging percentage.",
    "OPS": "OBP + SLG.",
    "wOBA": "Weighted on-base average.",
    "Hard Hit%": "Percent of batted balls hit hard.",
    "K%": "Strikeout rate.",
    "BB%": "Walk rate.",
    "PA": "Plate appearances.",
    "IP": "Innings pitched.",
    "ERA": "Earned run average.",
    "WHIP": "Walks + hits per inning pitched.",
    "K-BB%": "Strikeout rate minus walk rate.",
    "Whiff%": "Swing-and-miss rate.",
    "Hard Hit% Allowed": "Hard-hit contact allowed.",
}

with st.expander("What do these stats mean? (Quick guide)", expanded=False):
    st.markdown("""
**Hitters**
- **wOBA**: Weighted on-base. Better overall hitter quality than AVG.
- **OPS**: OBP + SLG.
- **Hard Hit%**: Quality of contact.
- **K% / BB%**: Plate discipline.

**Pitchers**
- **ERA / WHIP**: Familiar run prevention and baserunner metrics.
- **K-BB%**: One of the best quick skill indicators.
- **Whiff%**: Swing-and-miss ability.
- **Hard Hit% Allowed**: Contact quality allowed.
""")
st.info("Tip: Use Min PA / Min IP thresholds to reduce small-sample noise.")

# =========================================================
# TABS
# =========================================================
tab_hitters, tab_pitchers, tab_trends, tab_matchups, tab_best_props, tab_props, tab_profile, tab_standings, tab_schedule, tab_about = st.tabs(
    ["Hitters", "Pitchers", "Trends", "Matchups", "Best Props", "Props", "Player Profile", "Standings", "Today's Schedule", "About"]
)

# =========================================================
# HITTERS
# =========================================================
with tab_hitters:
    st.header("Hitters — Leaderboard (2026)")

    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    with c1:
        view_mode_h = st.radio("View", ["Basic", "Advanced"], horizontal=True, index=0, key="view_mode_h")
    with c2:
        max_pa = int(hitters["pa"].max()) if "pa" in hitters.columns and pd.notna(hitters["pa"].max()) else 700
        min_pa = st.slider("Min PA", 0, max_pa, 300, 10, key="min_pa_h")
    with c3:
        if "team" in hitters.columns:
            team_vals_h = [t for t in hitters["team"].dropna().unique().tolist() if str(t).strip() != ""]
            team_options_h = ["All"] + sorted(team_vals_h)
            team_filter_h = st.selectbox("Team", team_options_h, key="team_filter_h")
        else:
            team_filter_h = "All"
            st.selectbox("Team", ["All"], key="team_filter_h_disabled", disabled=True)
    with c4:
        search_h = st.text_input("Search", "", key="search_h").strip().lower()
    with c5:
        top_n_h = st.select_slider("Show top N", options=[10, 25, 50, 100], value=25, key="top_n_h")

    sort_options_basic = {
        "OPS": "on_base_plus_slg",
        "HR": "home_run",
        "RBI": "rbi",
        "H": "hit",
        "AVG": "batting_avg",
        "OBP": "on_base_percent",
        "SLG": "slg_percent",
        "BB": "walk",
        "SO": "strikeout",
    }
    sort_options_adv = {
        "wOBA": "woba",
        "OPS": "on_base_plus_slg",
        "Hard Hit %": "hard_hit_percent",
        "K% (lower better)": "k_percent",
        "BB%": "bb_percent",
    }

    sort_dict = sort_options_basic if view_mode_h == "Basic" else sort_options_adv
    sort_label_h = st.selectbox("Sort hitters by", list(sort_dict.keys()), index=0, key="sort_h")
    sort_col_h = sort_dict[sort_label_h]

    view = hitters.copy()
    view = view[view["pa"] >= min_pa]
    if team_filter_h != "All" and "team" in view.columns:
        view = view[view["team"] == team_filter_h]
    leaders_h = view.copy()

    if search_h:
        view = view[view["player_name"].astype(str).str.lower().str.contains(search_h)]

    ascending = True if sort_label_h.startswith("K%") else False
    view = view.sort_values(sort_col_h, ascending=ascending, na_position="last")

    basic_cols = ["player_name", "team", "pa", "batting_avg", "hit", "home_run", "rbi", "strikeout", "walk", "slg_percent", "on_base_percent", "on_base_plus_slg"]
    adv_cols = ["player_name", "team", "pa", "woba", "hard_hit_percent", "k_percent", "bb_percent", "on_base_plus_slg"]
    display_cols = basic_cols if view_mode_h == "Basic" else adv_cols
    display_cols = [c for c in display_cols if c in view.columns]

    st.subheader("Leaders (updates with your filters)")
    l1, l2, l3 = st.columns(3)
    name1, val1 = top_leader(leaders_h, "woba", True)
    name2, val2 = top_leader(leaders_h, "hard_hit_percent", True)
    name3, val3 = top_leader(leaders_h, "on_base_plus_slg", True)
    l1.metric("wOBA Leader", f"{name1}", fmt(val1, "float3") if val1 is not None else "—")
    l2.metric("Hard Hit% Leader", f"{name2}", fmt(val2, "pct", 1) if val2 is not None else "—")
    l3.metric("OPS Leader", f"{name3}", fmt(val3, "float3") if val3 is not None else "—")

    avg_ops = safe_mean(leaders_h, "on_base_plus_slg")
    avg_woba = safe_mean(leaders_h, "woba")
    avg_hh = safe_mean(leaders_h, "hard_hit_percent")

    st.caption(
        "League Averages (filtered): "
        f"OPS {fmt(avg_ops, 'float3') if avg_ops is not None else '—'} · "
        f"wOBA {fmt(avg_woba, 'float3') if avg_woba is not None else '—'} · "
        f"Hard Hit% {fmt(avg_hh, 'pct', 1) if avg_hh is not None else '—'}"
    )

    st.markdown("### Hot List (vs league avg, updates with Min PA)")
    h1, h2 = st.columns(2)
    hot_woba = top_delta_list(leaders_h, "woba", avg_woba, n=5, higher_better=True)
    hot_hh = top_delta_list(leaders_h, "hard_hit_percent", avg_hh, n=5, higher_better=True)

    with h1:
        st.write("**Top wOBA above league avg**")
        st.dataframe(format_hot_list(hot_woba, value_kind="float3"), use_container_width=True, hide_index=True)
    with h2:
        st.write("**Top Hard Hit% above league avg**")
        st.dataframe(format_hot_list(hot_hh, value_kind="pct", delta_pp=True), use_container_width=True, hide_index=True)

    st.caption(f"Note: Deltas are computed vs league averages within your current filter (Min PA: {min_pa}).")

    left, right = st.columns([2, 1], gap="large")

    with left:
        st.subheader(f"Leaderboard — {view_mode_h} (sorted by {sort_label_h})")
        st.write(f"Players shown: **{min(top_n_h, len(view))}** / **{len(view)}** (Min PA: **{min_pa}**)")
        kind = "hitters_basic" if view_mode_h == "Basic" else "hitters_adv"
        table_df = format_df_for_display(view[display_cols].head(top_n_h), kind)
        st.dataframe(table_df, use_container_width=True, hide_index=True)

        with st.expander("Stat Glossary (for the columns shown above)", expanded=False):
            shown_labels = list(table_df.columns)
            glossary_rows = [{"Stat": lab, "Meaning": STAT_HELP[lab]} for lab in shown_labels if lab in STAT_HELP]
            if glossary_rows:
                st.dataframe(pd.DataFrame(glossary_rows), use_container_width=True, hide_index=True)

        csv = view[display_cols].head(top_n_h).to_csv(index=False).encode("utf-8")
        st.download_button("Download current hitters table as CSV", data=csv, file_name="true_edge_hitters_2025.csv", mime="text/csv", key="hitters_download")

    with right:
        st.subheader("Hitter Spotlight")
        if len(view) == 0:
            st.info("No hitters match your filters.")
        else:
            table_players = view["player_name"].head(top_n_h).tolist()
            chosen = st.selectbox("Select a hitter", table_players, key="spotlight_h")

            rank_pop = leaders_h.copy()
            p = rank_pop[rank_pop["player_name"] == chosen].iloc[0]

            woba_val = p.get("woba")
            ops_val = p.get("on_base_plus_slg")
            hh_val = p.get("hard_hit_percent")

            s1, s2, s3 = st.columns(3)
            s1.metric("wOBA", fmt(woba_val, "float3"), delta=fmt_delta(delta_num(woba_val, avg_woba), "float3"))
            s2.metric("OPS", fmt(ops_val, "float3"), delta=fmt_delta(delta_num(ops_val, avg_ops), "float3"))
            s3.metric("Hard Hit %", fmt(hh_val, "pct", 1), delta=fmt_delta(delta_num(hh_val, avg_hh), pct_points=True))

            st.caption(
                f"League Avg (Min PA: {min_pa}): "
                f"wOBA {fmt(avg_woba, 'float3') if avg_woba is not None else '—'} · "
                f"OPS {fmt(avg_ops, 'float3') if avg_ops is not None else '—'} · "
                f"Hard Hit% {fmt(avg_hh, 'pct', 1) if avg_hh is not None else '—'}"
            )

            st.markdown("---")
            st.subheader("2025 Snapshot")
            a, b = st.columns(2)
            with a:
                st.markdown(f"**PA:** {fmt(p.get('pa'), 'int')}")
                st.markdown(f"**AVG:** {fmt(p.get('batting_avg'), 'avg')}")
                st.markdown(f"**OBP:** {fmt(p.get('on_base_percent'), 'avg')}")
                st.markdown(f"**RBI:** {fmt(p.get('rbi'), 'int')}")
            with b:
                st.markdown(f"**SLG:** {fmt(p.get('slg_percent'), 'avg')}")
                st.markdown(f"**HR:** {fmt(p.get('home_run'), 'int')}")
                st.markdown(f"**SO / BB:** {fmt(p.get('strikeout'), 'int')} / {fmt(p.get('walk'), 'int')}")

            st.markdown("---")
            st.subheader("Ranks & Percentiles (among filtered population)")
            lower_better_hitters = {"strikeout", "k_percent"}
            cols_for_rank = basic_cols if view_mode_h == "Basic" else adv_cols
            cols_for_rank = [c for c in cols_for_rank if c not in {"player_name", "team"}]
            st.dataframe(make_rank_table(rank_pop, p, cols_for_rank, lower_better_hitters), use_container_width=True, hide_index=True)

# =========================================================
# PITCHERS
# =========================================================
with tab_pitchers:
    st.header("Pitchers — Leaderboard (2026)")

    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    with c1:
        view_mode_p = st.radio("View", ["Basic", "Advanced"], horizontal=True, index=1, key="view_mode_p")
    with c2:
        max_ip = float(pitchers["ip"].max()) if "ip" in pitchers.columns and pd.notna(pitchers["ip"].max()) else 200.0
        min_ip = st.slider("Min IP", 0.0, max_ip, 40.0, 1.0, key="min_ip_p")
    with c3:
        if "team" in pitchers.columns:
            team_vals_p = [t for t in pitchers["team"].dropna().unique().tolist() if str(t).strip() != ""]
            team_options_p = ["All"] + sorted(team_vals_p)
            team_filter_p = st.selectbox("Team", team_options_p, key="team_filter_p")
        else:
            team_filter_p = "All"
            st.selectbox("Team", ["All"], key="team_filter_p_disabled", disabled=True)
    with c4:
        search_p = st.text_input("Search", "", key="search_p").strip().lower()
    with c5:
        top_n_p = st.select_slider("Show top N", options=[10, 25, 50, 100], value=25, key="top_n_p")

    sort_options_p_basic = {
        "ERA (lower better)": "era",
        "WHIP (lower better)": "whip",
        "IP": "ip",
        "SO": "strikeout",
        "BB (lower better)": "walk",
    }
    sort_options_p_adv = {
        "K-BB%": "k_minus_bb_percent",
        "K%": "k_percent",
        "Whiff %": "whiff_percent",
        "Hard Hit % (allowed) (lower better)": "hard_hit_percent",
        "BB% (lower better)": "bb_percent",
    }

    sort_dict_p = sort_options_p_basic if view_mode_p == "Basic" else sort_options_p_adv
    sort_label_p = st.selectbox("Sort pitchers by", list(sort_dict_p.keys()), index=0, key="sort_p")
    sort_col_p = sort_dict_p[sort_label_p]

    viewp = pitchers.copy()
    viewp = viewp[viewp["ip"] >= min_ip]
    if team_filter_p != "All" and "team" in viewp.columns:
        viewp = viewp[viewp["team"] == team_filter_p]
    leaders_p = viewp.copy()

    if search_p:
        viewp = viewp[viewp["player_name"].astype(str).str.lower().str.contains(search_p)]

    lower_better = "lower better" in sort_label_p
    viewp = viewp.sort_values(sort_col_p, ascending=lower_better, na_position="last")

    p_basic_cols = ["player_name", "team", "ip", "era", "whip", "strikeout", "walk"]
    p_adv_cols = ["player_name", "team", "ip", "k_percent", "bb_percent", "k_minus_bb_percent", "whiff_percent", "hard_hit_percent"]
    display_cols_p = p_basic_cols if view_mode_p == "Basic" else p_adv_cols
    display_cols_p = [c for c in display_cols_p if c in viewp.columns]

    st.subheader("Leaders (updates with your filters)")
    l1, l2, l3 = st.columns(3)
    name1, val1 = top_leader(leaders_p, "k_minus_bb_percent", True)
    name2, val2 = top_leader(leaders_p, "whiff_percent", True)
    name3, val3 = top_leader(leaders_p, "hard_hit_percent", False)
    l1.metric("K-BB% Leader", f"{name1}", fmt(val1, "pct", 1) if val1 is not None else "—")
    l2.metric("Whiff% Leader", f"{name2}", fmt(val2, "pct", 1) if val2 is not None else "—")
    l3.metric("Lowest Hard Hit% Allowed", f"{name3}", fmt(val3, "pct", 1) if val3 is not None else "—")

    avg_kbb = safe_mean(leaders_p, "k_minus_bb_percent")
    avg_whiff = safe_mean(leaders_p, "whiff_percent")
    avg_hha = safe_mean(leaders_p, "hard_hit_percent")

    st.caption(
        "League Averages (filtered): "
        f"K-BB% {fmt(avg_kbb, 'pct', 1) if avg_kbb is not None else '—'} · "
        f"Whiff% {fmt(avg_whiff, 'pct', 1) if avg_whiff is not None else '—'} · "
        f"Hard Hit% Allowed {fmt(avg_hha, 'pct', 1) if avg_hha is not None else '—'}"
    )

    st.markdown("### Hot List (vs league avg, updates with Min IP)")
    p_hot1, p_hot2 = st.columns(2)
    hot_kbb = top_delta_list(leaders_p, "k_minus_bb_percent", avg_kbb, n=5, higher_better=True)
    hot_hha = top_delta_list(leaders_p, "hard_hit_percent", avg_hha, n=5, higher_better=False)

    with p_hot1:
        st.write("**Top K-BB% above league avg**")
        st.dataframe(format_hot_list(hot_kbb, value_kind="pct", delta_pp=True), use_container_width=True, hide_index=True)
    with p_hot2:
        st.write("**Lowest Hard Hit% Allowed vs league avg**")
        st.dataframe(format_hot_list(hot_hha, value_kind="pct", delta_pp=True), use_container_width=True, hide_index=True)

    st.caption(f"Note: Deltas are computed vs league averages within your current filter (Min IP: {min_ip:.0f}).")

    leftp, rightp = st.columns([2, 1], gap="large")

    with leftp:
        st.subheader(f"Leaderboard — {view_mode_p} (sorted by {sort_label_p})")
        st.write(f"Pitchers shown: **{min(top_n_p, len(viewp))}** / **{len(viewp)}** (Min IP: **{min_ip}**)")
        kind = "pitchers_basic" if view_mode_p == "Basic" else "pitchers_adv"
        table_dfp = format_df_for_display(viewp[display_cols_p].head(top_n_p), kind)
        st.dataframe(table_dfp, use_container_width=True, hide_index=True)

        with st.expander("Stat Glossary (for the columns shown above)", expanded=False):
            shown_labels = list(table_dfp.columns)
            glossary_rows = [{"Stat": lab, "Meaning": STAT_HELP[lab]} for lab in shown_labels if lab in STAT_HELP]
            if glossary_rows:
                st.dataframe(pd.DataFrame(glossary_rows), use_container_width=True, hide_index=True)

        csvp = viewp[display_cols_p].head(top_n_p).to_csv(index=False).encode("utf-8")
        st.download_button("Download current pitchers table as CSV", data=csvp, file_name="true_edge_pitchers_2025.csv", mime="text/csv", key="pitchers_download")

    with rightp:
        st.subheader("Pitcher Spotlight")
        if len(viewp) == 0:
            st.info("No pitchers match your filters.")
        else:
            table_pitchers = viewp["player_name"].head(top_n_p).tolist()
            chosen_p = st.selectbox("Select a pitcher", table_pitchers, key="spotlight_p")

            rank_pop_p = leaders_p.copy()
            pp = rank_pop_p[rank_pop_p["player_name"] == chosen_p].iloc[0]

            kbb_val = pp.get("k_minus_bb_percent")
            whiff_val = pp.get("whiff_percent")
            hha_val = pp.get("hard_hit_percent")

            s1, s2, s3 = st.columns(3)
            s1.metric("K-BB%", fmt(kbb_val, "pct", 1), delta=fmt_delta(delta_num(kbb_val, avg_kbb), pct_points=True))
            s2.metric("Whiff %", fmt(whiff_val, "pct", 1), delta=fmt_delta(delta_num(whiff_val, avg_whiff), pct_points=True))
            s3.metric("Hard Hit %", fmt(hha_val, "pct", 1), delta=fmt_delta(delta_num(hha_val, avg_hha), pct_points=True), delta_color="inverse")

            st.caption(
                f"League Avg (Min IP: {min_ip:.0f}): "
                f"K-BB% {fmt(avg_kbb, 'pct', 1) if avg_kbb is not None else '—'} · "
                f"Whiff% {fmt(avg_whiff, 'pct', 1) if avg_whiff is not None else '—'} · "
                f"Hard Hit% Allowed {fmt(avg_hha, 'pct', 1) if avg_hha is not None else '—'}"
            )

            st.markdown("---")
            st.subheader("2025 Snapshot")
            a, b = st.columns(2)
            with a:
                st.markdown(f"**IP:** {fmt(pp.get('ip'), 'float1')}")
                st.markdown(f"**ERA:** {fmt(pp.get('era'), 'float2')}")
                st.markdown(f"**SO:** {fmt(pp.get('strikeout'), 'int')}")
            with b:
                st.markdown(f"**WHIP:** {fmt(pp.get('whip'), 'float2')}")
                st.markdown(f"**BB:** {fmt(pp.get('walk'), 'int')}")
                st.markdown(f"**K% / BB%:** {fmt(pp.get('k_percent'), 'pct', 1)} / {fmt(pp.get('bb_percent'), 'pct', 1)}")

            st.markdown("---")
            st.subheader("Ranks & Percentiles (among filtered population)")
            lower_better_pitchers = {"walk", "bb_percent", "hard_hit_percent", "era", "whip"}
            cols_for_rank_p = p_basic_cols if view_mode_p == "Basic" else p_adv_cols
            cols_for_rank_p = [c for c in cols_for_rank_p if c not in {"player_name", "team"}]
            st.dataframe(make_rank_table(rank_pop_p, pp, cols_for_rank_p, lower_better_pitchers), use_container_width=True, hide_index=True)

# =========================================================
# TRENDS
# =========================================================
with tab_trends:
    game_type_label = st.radio(
        "Game Type",
        ["Regular Season", "Spring Training", "Playoffs"],
        horizontal=True,
        index=0,
        key="trends_game_type"
    )
    game_type = {"Regular Season": "R", "Spring Training": "S", "Playoffs": "P"}[game_type_label]
    window = st.radio("Window", [7, 14, 30], horizontal=True, index=0, key="trends_window")

    st.header(f"Trends — Last {window} Days ({game_type_label})")
    st.caption("Pulled from MLB Stats API by date range. Run scripts/pull_trends.py to refresh.")
    st.caption("If Regular Season trends are empty, switch to Spring Training until regular season games populate.")

    h_path = TRENDS_DIR / f"hitters_trends_{game_type}_last{window}d.csv"
    p_path = TRENDS_DIR / f"pitchers_trends_{game_type}_last{window}d.csv"

    ht = safe_read_csv(h_path)
    pt = safe_read_csv(p_path)

    if ht.empty and pt.empty:
        st.info("No games found for this window/type yet. Try a different window or switch game type.")
    else:
        for c in ht.columns:
            if c not in ["player_name", "team"]:
                ht[c] = pd.to_numeric(ht[c], errors="coerce")
        for c in pt.columns:
            if c not in ["player_name", "team"]:
                pt[c] = pd.to_numeric(pt[c], errors="coerce")

        season_ops = hitters[["player_name", "on_base_plus_slg"]].copy()
        season_ops["on_base_plus_slg"] = pd.to_numeric(season_ops["on_base_plus_slg"], errors="coerce")
        if "ops" in ht.columns:
            ht = ht.merge(season_ops, on="player_name", how="left")
            ht["ops_delta_vs_season"] = ht["ops"] - ht["on_base_plus_slg"]

        if ("era" in pitchers.columns) and ("whip" in pitchers.columns):
            season_pitch = pitchers[["player_name", "era", "whip"]].copy()
            season_pitch["era"] = pd.to_numeric(season_pitch["era"], errors="coerce")
            season_pitch["whip"] = pd.to_numeric(season_pitch["whip"], errors="coerce")
            pt = pt.merge(season_pitch, on="player_name", how="left", suffixes=("", "_season"))

            if "era" in pt.columns and "era_season" in pt.columns:
                pt["era_delta_vs_season"] = pt["era_season"] - pt["era"]
            if "whip" in pt.columns and "whip_season" in pt.columns:
                pt["whip_delta_vs_season"] = pt["whip_season"] - pt["whip"]

        min_pa_tr = st.slider("Min PA", 0, 120, 20, 1, key="min_pa_tr")
        min_ip_tr = st.slider("Min IP", 0.0, 60.0, 10.0, 1.0, key="min_ip_tr")

        ht_view = ht.copy()
        if "plateAppearances" in ht_view.columns:
            ht_view = ht_view[ht_view["plateAppearances"] >= min_pa_tr]

        pt_view = pt.copy()
        if "inningsPitched" in pt_view.columns:
            pt_view = pt_view[pt_view["inningsPitched"] >= min_ip_tr]

        st.markdown("## 🔥 Biggest Movers")
        left_m, right_m = st.columns(2, gap="large")

        with left_m:
            st.subheader("Hitters — Biggest ΔOPS vs Season")
            if "ops_delta_vs_season" in ht_view.columns:
                movers_hit = ht_view.dropna(subset=["ops_delta_vs_season"]).sort_values("ops_delta_vs_season", ascending=False).head(5)
                if not movers_hit.empty:
                    movers_hit_display = movers_hit[[c for c in ["player_name", "team", "ops", "on_base_plus_slg", "ops_delta_vs_season"] if c in movers_hit.columns]].copy()
                    movers_hit_display = movers_hit_display.rename(columns={
                        "player_name": "Player",
                        "team": "Team",
                        "ops": "OPS",
                        "on_base_plus_slg": "Season OPS",
                        "ops_delta_vs_season": "ΔOPS",
                    })
                    for c in ["OPS", "Season OPS", "ΔOPS"]:
                        if c in movers_hit_display.columns:
                            movers_hit_display[c] = pd.to_numeric(movers_hit_display[c], errors="coerce").map(
                                lambda x: f"{x:+.3f}" if c == "ΔOPS" and pd.notna(x)
                                else (f"{x:.3f}" if pd.notna(x) else "—")
                            )
                    st.dataframe(movers_hit_display, use_container_width=True, hide_index=True)
                else:
                    st.info("No hitter movers available.")
            else:
                st.info("No OPS delta data available.")

        with right_m:
            st.subheader("Pitchers — Biggest ΔERA vs Season")
            if "era_delta_vs_season" in pt_view.columns:
                movers_pitch = pt_view.dropna(subset=["era_delta_vs_season"]).sort_values("era_delta_vs_season", ascending=False).head(5)
                if not movers_pitch.empty:
                    movers_pitch_display = movers_pitch[[c for c in ["player_name", "team", "era", "era_season", "era_delta_vs_season"] if c in movers_pitch.columns]].copy()
                    movers_pitch_display = movers_pitch_display.rename(columns={
                        "player_name": "Player",
                        "team": "Team",
                        "era": "ERA",
                        "era_season": "Season ERA",
                        "era_delta_vs_season": "ΔERA",
                    })
                    for c in ["ERA", "Season ERA", "ΔERA"]:
                        if c in movers_pitch_display.columns:
                            movers_pitch_display[c] = pd.to_numeric(movers_pitch_display[c], errors="coerce").map(
                                lambda x: f"{x:+.2f}" if c == "ΔERA" and pd.notna(x)
                                else (f"{x:.2f}" if pd.notna(x) else "—")
                            )
                    st.dataframe(movers_pitch_display, use_container_width=True, hide_index=True)
                else:
                    st.info("No pitcher movers available.")
            else:
                st.info("ERA improvement movers will appear after season ERA data is available.")

        st.caption(f"Note: Deltas are computed within the selected {window}-day window and current game type.")

        st.markdown("## 🔥 Biggest Movers (Raw Window Stats)")
        c1, c2 = st.columns(2, gap="large")

        with c1:
            stat_choice = st.selectbox(
                "Choose hitter mover stat",
                ["ops", "woba", "hard_hit_percent", "homeRuns", "hits"],
                key="movers_stat"
            )
            up = get_biggest_movers(ht_view, stat_choice, top_n=10, ascending=False)
            down = get_biggest_movers(ht_view, stat_choice, top_n=10, ascending=True)

            st.markdown("### 🚀 Trending Up")
            if not up.empty:
                st.dataframe(format_trends_hitters(up), use_container_width=True, hide_index=True)
            else:
                st.info("No hitter up movers available.")

            st.markdown("### ❄️ Trending Down")
            if not down.empty:
                st.dataframe(format_trends_hitters(down), use_container_width=True, hide_index=True)
            else:
                st.info("No hitter down movers available.")

        with c2:
            p_stat_choice = st.selectbox(
                "Choose pitcher mover stat",
                ["strikeOuts", "era", "whip"],
                key="p_movers_stat"
            )

            if p_stat_choice in {"era", "whip"}:
                p_up = get_biggest_movers(pt_view, p_stat_choice, top_n=10, ascending=True)
                p_down = get_biggest_movers(pt_view, p_stat_choice, top_n=10, ascending=False)
            else:
                p_up = get_biggest_movers(pt_view, p_stat_choice, top_n=10, ascending=False)
                p_down = get_biggest_movers(pt_view, p_stat_choice, top_n=10, ascending=True)

            st.markdown("### 🚀 Trending Up")
            if not p_up.empty:
                st.dataframe(format_trends_pitchers(p_up), use_container_width=True, hide_index=True)
            else:
                st.info("No pitcher up movers available.")

            st.markdown("### ❄️ Trending Down")
            if not p_down.empty:
                st.dataframe(format_trends_pitchers(p_down), use_container_width=True, hide_index=True)
            else:
                st.info("No pitcher down movers available.")

        left, right = st.columns(2, gap="large")

        with left:
            st.subheader("Hitters — Hot in this window")
            sort_opt = st.selectbox(
                "Sort hitters by",
                ["OPS", "ΔOPS vs Season", "HR", "Hits"],
                index=0,
                key="trends_hit_sort"
            )

            if sort_opt == "OPS":
                sort_col, asc = "ops", False
            elif sort_opt == "ΔOPS vs Season":
                sort_col, asc = "ops_delta_vs_season", False
            elif sort_opt == "HR":
                sort_col, asc = "homeRuns", False
            else:
                sort_col, asc = "hits", False

            show_cols = [c for c in [
                "player_name", "team", "plateAppearances", "avg", "obp", "slg", "ops",
                "ops_delta_vs_season", "homeRuns", "hits", "rbi", "baseOnBalls", "strikeOuts"
            ] if c in ht_view.columns]

            ht_sorted = ht_view.sort_values(sort_col, ascending=asc, na_position="last")
            st.dataframe(format_trends_hitters(ht_sorted[show_cols].head(25)), use_container_width=True, hide_index=True)

        with right:
            st.subheader("Pitchers — Hot in this window")
            sort_p = st.selectbox(
                "Sort pitchers by",
                ["ERA", "WHIP", "SO", "ΔERA vs Season"],
                index=0,
                key="trends_pitch_sort"
            )

            if sort_p == "ERA":
                sort_col, asc = "era", True
            elif sort_p == "WHIP":
                sort_col, asc = "whip", True
            elif sort_p == "ΔERA vs Season":
                sort_col, asc = "era_delta_vs_season", False
            else:
                sort_col, asc = "strikeOuts", False

            show_cols_p = [c for c in [
                "player_name", "team", "inningsPitched", "era", "whip",
                "era_delta_vs_season", "strikeOuts", "baseOnBalls"
            ] if c in pt_view.columns]

            pt_sorted = pt_view.sort_values(sort_col, ascending=asc, na_position="last")
            st.dataframe(format_trends_pitchers(pt_sorted[show_cols_p].head(25)), use_container_width=True, hide_index=True)

# =========================================================
# MATCHUPS
# =========================================================
with tab_matchups:
    st.header("Matchups — Today")
    st.caption("Small V1 version: team strength + probable pitcher adjustment + simple win probability.")

    today = date.today().isoformat()
    matchup_date = st.text_input("Date (YYYY-MM-DD)", value=today, key="matchup_date")

    sched = fetch_schedule_raw(matchup_date)
    if sched.empty:
        st.info("No games found for that date.")
    else:
        team_strengths = build_team_strengths(hitters, pitchers)
        pitcher_bonus_df = compute_pitcher_bonus(pitchers)

        if team_strengths.empty:
            st.warning("Team strengths are unavailable because the team field is missing from hitters or pitchers.")
        else:
            bonus_map = dict(zip(pitcher_bonus_df["player_name"], pitcher_bonus_df["pitcher_bonus"]))
            score_map = dict(zip(team_strengths["team"], team_strengths["team_score"]))

            rows = []
            for _, g in sched.iterrows():
                away_key = g["Away"]
                home_key = g["Home"]

                away_score = score_map.get(away_key, 0)
                home_score = score_map.get(home_key, 0)

                away_sp = g.get("Away Probable", "—")
                home_sp = g.get("Home Probable", "—")

                away_bonus = bonus_map.get(away_sp, 0)
                home_bonus = bonus_map.get(home_sp, 0)

                diff = (home_score + home_bonus + 0.12) - (away_score + away_bonus)
                home_prob = logistic(diff * 1.5)
                away_prob = 1 - home_prob

                if home_prob >= 0.55:
                    pick = g["Home"]
                    confidence = "Lean" if home_prob < 0.60 else ("Medium" if home_prob < 0.67 else "Strong")
                elif away_prob >= 0.55:
                    pick = g["Away"]
                    confidence = "Lean" if away_prob < 0.60 else ("Medium" if away_prob < 0.67 else "Strong")
                else:
                    away_prob = 0.5
                    home_prob = 0.5
                    pick = "Coin Flip"
                    confidence = "Low"
                
                if pd.isna(away_prob) or pd.isna(home_prob):
                    away_prob = 0.5
                    home_prob = 0.5
                    pick = "Coin Flip"
                    confiedence = "Low"
                
                rows.append({
                    "Away": g["Away"],
                    "Home": g["Home"],
                    "Away SP": away_sp,
                    "Home SP": home_sp,
                    "Away Win %": f"{away_prob * 100:.1f}%",
                    "Home Win %": f"{home_prob * 100:.1f}%",
                    "Pick": pick,
                    "Confidence": confidence,
                })

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            with st.expander("How Matchups works", expanded=False):
                st.write("""
This small V1 matchup model combines:
- team offense quality (OPS, wOBA, Hard Hit%)
- team pitching quality (K-BB%, Whiff%, Hard Hit% Allowed, and if available ERA/WHIP)
- probable pitcher bonus
- small home-field advantage
""")

# =========================================================
# BEST PROPS
# =========================================================
with tab_best_props:
    st.header("🔥 Best Props of the Day")
    st.caption("Early version powered by current signal scores and today's schedule.")
    st.caption("Best props filtered to teams playing on the selected date.")
    st.markdown("---")

    today = date.today().isoformat()
    props_date = st.text_input("Date (YYYY-MM-DD)", value=today, key="best_props_date")

    sched = fetch_schedule_raw(props_date)
    teams_today = set()
    if not sched.empty:
        teams_today = set(sched["Away"].dropna().tolist()) | set(sched["Home"].dropna().tolist())

    if not teams_today:
        st.info("No games found for that date.")
    else:
        hitters_props = add_hitter_prop_scores(hitters.copy())
        pitchers_props = add_pitcher_prop_scores(pitchers.copy())

        hitters_props = project_home_run_props(hitters_props)
        hitters_props = project_hits_props(hitters_props)
        pitchers_props = project_strikeouts(pitchers_props)

        hitters_props["HR Lean"] = hitters_props.apply(get_hr_lean, axis=1)
        hitters_props["Hits Lean"] = hitters_props.apply(get_hits_lean, axis=1)
        pitchers_props["K Lean"] = pitchers_props.apply(get_k_lean, axis=1)

        if "team" in hitters_props.columns:
            hitters_today = hitters_props[hitters_props["team"].isin(teams_today)].copy()
        else:
            hitters_today = hitters_props.copy()

        if "team" in pitchers_props.columns:
            pitchers_today = pitchers_props[pitchers_props["team"].isin(teams_today)].copy()
        else:
            pitchers_today = pitchers_props.copy()

        if "pa" in hitters_today.columns:
            hitters_today = hitters_today[hitters_today["pa"] >= 50]
        if "ip" in pitchers_today.columns:
            pitchers_today = pitchers_today[pitchers_today["ip"] >= 10]

        c1, c2, c3 = st.columns(3, gap="large")

        with c1:
            st.subheader("Top 3 HR Props")
            hr_cols = [c for c in [
                "player_name", "team", "home_run", "hard_hit_percent", "slg_percent",
                "woba", "hr_signal", "proj_hr_chance", "HR Lean"
            ] if c in hitters_today.columns]

            hr_df = hitters_today.sort_values("hr_signal", ascending=False)[hr_cols].head(3).copy()

            if not hr_df.empty:
                hr_df = hr_df.rename(columns={
                    "player_name": "Player",
                    "team": "Team",
                    "home_run": "HR",
                    "hard_hit_percent": "Hard Hit%",
                    "slg_percent": "SLG",
                    "woba": "wOBA",
                    "hr_signal": "Signal",
                    "proj_hr_chance": "HR Chance",
                })

                if "Hard Hit%" in hr_df.columns:
                    hr_df["Hard Hit%"] = pd.to_numeric(hr_df["Hard Hit%"], errors="coerce").map(
                        lambda x: f"{x:.1f}%" if pd.notna(x) else "—"
                    )
                for c in ["SLG", "wOBA"]:
                    if c in hr_df.columns:
                        hr_df[c] = pd.to_numeric(hr_df[c], errors="coerce").map(
                            lambda x: f"{x:.3f}" if pd.notna(x) else "—"
                        )
                if "Signal" in hr_df.columns:
                    hr_df["Signal"] = pd.to_numeric(hr_df["Signal"], errors="coerce").map(
                        lambda x: f"{x:.1f}" if pd.notna(x) else "—"
                    )
                if "HR Chance" in hr_df.columns:
                    hr_df["HR Chance"] = pd.to_numeric(hr_df["HR Chance"], errors="coerce").map(
                        lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—"
                    )

                st.dataframe(hr_df, use_container_width=True, hide_index=True)
            else:
                st.info("No HR props available.")

        with c2:
            st.subheader("Top 3 Hits Props")
            hit_cols = [c for c in [
                "player_name", "team", "hit", "batting_avg", "on_base_percent",
                "woba", "hits_signal", "proj_hits", "Hits Lean"
            ] if c in hitters_today.columns]

            hits_df = hitters_today.sort_values("hits_signal", ascending=False)[hit_cols].head(3).copy()

            if not hits_df.empty:
                hits_df = hits_df.rename(columns={
                    "player_name": "Player",
                    "team": "Team",
                    "hit": "H",
                    "batting_avg": "AVG",
                    "on_base_percent": "OBP",
                    "woba": "wOBA",
                    "hits_signal": "Signal",
                    "proj_hits": "Projected Hits",
                })

                for c in ["AVG", "OBP", "wOBA"]:
                    if c in hits_df.columns:
                        hits_df[c] = pd.to_numeric(hits_df[c], errors="coerce").map(
                            lambda x: f"{x:.3f}" if pd.notna(x) else "—"
                        )
                if "Signal" in hits_df.columns:
                    hits_df["Signal"] = pd.to_numeric(hits_df["Signal"], errors="coerce").map(
                        lambda x: f"{x:.1f}" if pd.notna(x) else "—"
                    )
                if "Projected Hits" in hits_df.columns:
                    hits_df["Projected Hits"] = pd.to_numeric(hits_df["Projected Hits"], errors="coerce").map(
                        lambda x: f"{x:.2f}" if pd.notna(x) else "—"
                    )

                st.dataframe(hits_df, use_container_width=True, hide_index=True)
            else:
                st.info("No hits props available.")

        with c3:
            st.subheader("Top 3 Strikeout Props")
            k_cols = [c for c in [
                "player_name", "team", "strikeout", "k_minus_bb_percent",
                "whiff_percent", "k_signal", "proj_k", "K Lean"
            ] if c in pitchers_today.columns]

            k_df = pitchers_today.sort_values("k_signal", ascending=False)[k_cols].head(3).copy()

            if not k_df.empty:
                k_df = k_df.rename(columns={
                    "player_name": "Player",
                    "team": "Team",
                    "strikeout": "SO",
                    "k_minus_bb_percent": "K-BB%",
                    "whiff_percent": "Whiff%",
                    "k_signal": "Signal",
                    "proj_k": "Projected Ks",
                })

                for c in ["K-BB%", "Whiff%"]:
                    if c in k_df.columns:
                        k_df[c] = pd.to_numeric(k_df[c], errors="coerce").map(
                            lambda x: f"{x:.1f}%" if pd.notna(x) else "—"
                        )
                if "Signal" in k_df.columns:
                    k_df["Signal"] = pd.to_numeric(k_df["Signal"], errors="coerce").map(
                        lambda x: f"{x:.1f}" if pd.notna(x) else "—"
                    )
                if "Projected Ks" in k_df.columns:
                    k_df["Projected Ks"] = pd.to_numeric(k_df["Projected Ks"], errors="coerce").map(
                        lambda x: f"{x:.1f}" if pd.notna(x) else "—"
                    )

                st.dataframe(k_df, use_container_width=True, hide_index=True)
            else:
                st.info("No strikeout props available.")

        st.markdown("---")
        st.subheader("How these work")
        st.write(
            "These are early prop leans based on signal scores and filtered to teams playing on the selected date. "
            "HR uses a probability-style estimate, Hits uses a projected hits model, and Ks uses a projected strikeout model."
        )

# =========================================================
# PROPS
# =========================================================
with tab_props:
    st.header("Props (Beta)")
    st.caption("Season-long signal scores built from current player skill indicators. Good for early filtering, not final picks.")

    hitters_props = add_hitter_prop_scores(hitters.copy())
    pitchers_props = add_pitcher_prop_scores(pitchers.copy())

    h_controls, p_controls = st.columns(2, gap="large")

    with h_controls:
        st.subheader("Hitter Props")
        max_pa_props = int(hitters_props["pa"].max()) if "pa" in hitters_props.columns and pd.notna(hitters_props["pa"].max()) else 700
        min_pa_props = st.slider("Min PA", 0, max_pa_props, 300, 10, key="props_min_pa")
        if "team" in hitters_props.columns:
            team_vals_hp = [t for t in hitters_props["team"].dropna().unique().tolist() if str(t).strip() != ""]
            team_options_hp = ["All"] + sorted(team_vals_hp)
            team_filter_hp = st.selectbox("Team", team_options_hp, key="props_team_h")
        else:
            team_filter_hp = "All"
        hitter_signal = st.selectbox("Signal", ["HR Signal", "Hits Signal"], index=0, key="props_h_signal")

    with p_controls:
        st.subheader("Pitcher Props")
        max_ip_props = float(pitchers_props["ip"].max()) if "ip" in pitchers_props.columns and pd.notna(pitchers_props["ip"].max()) else 200.0
        min_ip_props = st.slider("Min IP", 0.0, max_ip_props, 40.0, 1.0, key="props_min_ip")
        if "team" in pitchers_props.columns:
            team_vals_pp = [t for t in pitchers_props["team"].dropna().unique().tolist() if str(t).strip() != ""]
            team_options_pp = ["All"] + sorted(team_vals_pp)
            team_filter_pp = st.selectbox("Team", team_options_pp, key="props_team_p")
        else:
            team_filter_pp = "All"

    hitters_props = hitters_props[hitters_props["pa"] >= min_pa_props]
    if team_filter_hp != "All" and "team" in hitters_props.columns:
        hitters_props = hitters_props[hitters_props["team"] == team_filter_hp]

    pitchers_props = pitchers_props[pitchers_props["ip"] >= min_ip_props]
    if team_filter_pp != "All" and "team" in pitchers_props.columns:
        pitchers_props = pitchers_props[pitchers_props["team"] == team_filter_pp]

    left, right = st.columns(2, gap="large")

    with left:
        if hitter_signal == "HR Signal":
            show = hitters_props.sort_values("hr_signal", ascending=False)[[c for c in ["player_name", "team", "pa", "home_run", "hard_hit_percent", "slg_percent", "woba", "hr_signal"] if c in hitters_props.columns]].head(25)
            show = show.rename(columns={"player_name": "Player", "team": "Team", "pa": "PA", "home_run": "HR", "hard_hit_percent": "Hard Hit%", "slg_percent": "SLG", "woba": "wOBA", "hr_signal": "HR Signal"})
        else:
            show = hitters_props.sort_values("hits_signal", ascending=False)[[c for c in ["player_name", "team", "pa", "hit", "batting_avg", "on_base_percent", "woba", "k_percent", "hits_signal"] if c in hitters_props.columns]].head(25)
            show = show.rename(columns={"player_name": "Player", "team": "Team", "pa": "PA", "hit": "H", "batting_avg": "AVG", "on_base_percent": "OBP", "woba": "wOBA", "k_percent": "K%", "hits_signal": "Hits Signal"})

        for c in ["AVG", "OBP", "wOBA", "SLG"]:
            if c in show.columns:
                show[c] = pd.to_numeric(show[c], errors="coerce").map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
        for c in ["Hard Hit%", "K%"]:
            if c in show.columns:
                show[c] = pd.to_numeric(show[c], errors="coerce").map(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
        for c in ["HR Signal", "Hits Signal"]:
            if c in show.columns:
                show[c] = pd.to_numeric(show[c], errors="coerce").map(lambda x: f"{x:.1f}" if pd.notna(x) else "—")

        st.dataframe(show, use_container_width=True, hide_index=True)

    with right:
        showp = pitchers_props.sort_values("k_signal", ascending=False)[[c for c in ["player_name", "team", "ip", "strikeout", "k_minus_bb_percent", "whiff_percent", "hard_hit_percent", "k_signal"] if c in pitchers_props.columns]].head(25)
        showp = showp.rename(columns={"player_name": "Player", "team": "Team", "ip": "IP", "strikeout": "SO", "k_minus_bb_percent": "K-BB%", "whiff_percent": "Whiff%", "hard_hit_percent": "Hard Hit% Allowed", "k_signal": "K Signal"})

        if "IP" in showp.columns:
            showp["IP"] = pd.to_numeric(showp["IP"], errors="coerce").map(lambda x: f"{x:.1f}" if pd.notna(x) else "—")
        if "SO" in showp.columns:
            showp["SO"] = pd.to_numeric(showp["SO"], errors="coerce").map(lambda x: f"{int(x)}" if pd.notna(x) else "—")
        for c in ["K-BB%", "Whiff%", "Hard Hit% Allowed"]:
            if c in showp.columns:
                showp[c] = pd.to_numeric(showp[c], errors="coerce").map(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
        if "K Signal" in showp.columns:
            showp["K Signal"] = pd.to_numeric(showp["K Signal"], errors="coerce").map(lambda x: f"{x:.1f}" if pd.notna(x) else "—")

        st.dataframe(showp, use_container_width=True, hide_index=True)

    with st.expander("How Props works", expanded=False):
        st.write("""
These are season-long signal scores.

**HR Signal** leans on:
- Hard Hit%
- SLG
- wOBA
- HR totals

**Hits Signal** leans on:
- AVG
- OBP
- wOBA
- lower K%

**K Signal** leans on:
- K-BB%
- Whiff%
- strikeout volume
- lower hard-hit / walk profile
""")

# =========================================================
# PLAYER PROFILE
# =========================================================
with tab_profile:
    st.header("Player Profile (2026)")
    st.caption("One dropdown. The app auto-detects whether you selected a hitter or pitcher.")

    hitters_list = sorted(hitters["player_name"].dropna().unique().tolist())
    pitchers_list = sorted(pitchers["player_name"].dropna().unique().tolist())
    options = (["Hitter — " + n for n in hitters_list] + ["Pitcher — " + n for n in pitchers_list])

    chosen = st.selectbox("Choose a player", options)

    if chosen.startswith("Hitter — "):
        name = chosen.replace("Hitter — ", "", 1)
        row = hitters[hitters["player_name"] == name].iloc[0]

        st.subheader(f"{name} (Hitter)")
        c1, c2, c3 = st.columns(3)
        c1.metric("wOBA", fmt(row.get("woba"), "float3"))
        c2.metric("OPS", fmt(row.get("on_base_plus_slg"), "float3"))
        c3.metric("Hard Hit %", fmt(row.get("hard_hit_percent"), "pct", 1))

        st.markdown("---")
        st.subheader("Snapshot")
        s1, s2 = st.columns(2)
        with s1:
            st.markdown(f"**PA:** {fmt(row.get('pa'), 'int')}")
            st.markdown(f"**AVG:** {fmt(row.get('batting_avg'), 'avg')}")
            st.markdown(f"**OBP:** {fmt(row.get('on_base_percent'), 'avg')}")
            st.markdown(f"**RBI:** {fmt(row.get('rbi'), 'int')}")
        with s2:
            st.markdown(f"**SLG:** {fmt(row.get('slg_percent'), 'avg')}")
            st.markdown(f"**HR:** {fmt(row.get('home_run'), 'int')}")
            st.markdown(f"**SO / BB:** {fmt(row.get('strikeout'), 'int')} / {fmt(row.get('walk'), 'int')}")

    else:
        name = chosen.replace("Pitcher — ", "", 1)
        row = pitchers[pitchers["player_name"] == name].iloc[0]

        st.subheader(f"{name} (Pitcher)")
        c1, c2, c3 = st.columns(3)
        c1.metric("K-BB%", fmt(row.get("k_minus_bb_percent"), "pct", 1))
        c2.metric("Whiff %", fmt(row.get("whiff_percent"), "pct", 1))
        c3.metric("Hard Hit %", fmt(row.get("hard_hit_percent"), "pct", 1))

        st.markdown("---")
        st.subheader("Snapshot")
        s1, s2 = st.columns(2)
        with s1:
            st.markdown(f"**IP:** {fmt(row.get('ip'), 'float1')}")
            st.markdown(f"**ERA:** {fmt(row.get('era'), 'float2')}")
            st.markdown(f"**SO:** {fmt(row.get('strikeout'), 'int')}")
        with s2:
            st.markdown(f"**WHIP:** {fmt(row.get('whip'), 'float2')}")
            st.markdown(f"**BB:** {fmt(row.get('walk'), 'int')}")
            st.markdown(f"**K% / BB%:** {fmt(row.get('k_percent'), 'pct', 1)} / {fmt(row.get('bb_percent'), 'pct', 1)}")

# =========================================================
# STANDINGS
# =========================================================
with tab_standings:
    st.header("MLB Standings")
    season = st.number_input("Season", value=2026, step=1)

    try:
        standings = fetch_standings(int(season))

        division_order = [
            "American League East",
            "American League Central",
            "American League West",
            "National League East",
            "National League Central",
            "National League West",
        ]

        # Create consistent labels
        if not standings.empty:
            standings = standings.copy()
            standings["Division Label"] = (
                standings["League"].astype(str) + " " +
                standings["Division"].astype(str).str.replace(" Division", "", regex=False)
            )

        for div in division_order:
            st.subheader(div)

            if not standings.empty:
                group = standings[standings["Division Label"].str.contains(div.split()[-1], na=False)]
            else:
                group = pd.DataFrame()

            if group.empty:
                st.info("Season just started - standings will update after games begin.")
            else:
                st.dataframe(
                    group[["Team", "W", "L", "Win%", "GB"]],
                    use_container_width=True,
                    hide_index=True,
                )

    except Exception as e:
        st.error(f"Could not load standings: {e}")

# =========================================================
# SCHEDULE
# =========================================================
with tab_schedule:
    st.header("Today's MLB Schedule")

    today = date.today().isoformat()
    game_date = st.text_input("Date (YYYY-MM-DD)", value=today)

    try:
        sched = fetch_schedule_raw(game_date)
        if sched.empty:
            st.info("No games found for that date.")
        else:
            display_sched = sched[["Start (ISO)", "Away", "Home", "Status", "Venue", "Away Probable", "Home Probable"]].copy()
            st.dataframe(display_sched, use_container_width=True, hide_index=True)

            csvs = display_sched.to_csv(index=False).encode("utf-8")
            st.download_button("Download schedule as CSV", data=csvs, file_name=f"true_edge_schedule_{game_date}.csv", mime="text/csv", key="schedule_download")
    except Exception as e:
        st.error(f"Could not load schedule: {e}")

# =========================================================
# ABOUT
# =========================================================
with tab_about:
    st.header("About True Edge Sports")
    st.write("""
True Edge Sports is an MLB analytics dashboard built with Python, pandas, Streamlit, and the MLB Stats API.

**V1 includes**
- Hitters leaderboard (Basic + Advanced)
- Pitchers leaderboard (Basic + Advanced)
- Team filters
- Trends tab with Regular Season / Spring Training / Playoffs
- Biggest Movers
- Matchups (small V1)
- Props (Beta)
- Player profiles
- Live standings + today’s schedule

**Planned**
- stronger matchup model
- rolling trend engine
- team pages
- premium betting analytics layer
- expansion to NHL / NFL / NBA / Soccer
""")