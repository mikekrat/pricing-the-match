"""pricing-the-match: how a betting operator prices a match and what drives its profit.

Three layers, each explained step by step in notebooks/:
  1. probabilities  - Elo ratings and a logistic model, scored against the bookmaker
  2. prices         - odds with a margin, and how exposed those prices are
  3. profit         - a season simulated many times with casual and sharp bettors
  4. improvements   - form and goal features, tuned Elo, walk-forward testing, goals markets
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REQUIRED = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
ODDS = ["B365H", "B365D", "B365A"]
OU_ODDS = ["B365>2.5", "B365<2.5"]      # over / under 2.5 goals (optional columns)


def load_matches(folder: str | Path) -> pd.DataFrame:
    """Read every CSV in `folder`, one file per season, into one table sorted by date."""
    files = sorted(Path(folder).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {folder}")
    frames = []
    for f in files:
        df = pd.read_csv(f, encoding="latin-1")
        missing = [c for c in REQUIRED + ODDS if c not in df.columns]
        if missing:
            raise ValueError(f"{f.name} is missing columns {missing}")
        for c in OU_ODDS:                      # not every season file has them
            if c not in df.columns:
                df[c] = np.nan
        df = df[REQUIRED + ODDS + OU_ODDS].dropna(subset=REQUIRED)
        # football-data.co.uk mixes dd/mm/yy and dd/mm/yyyy across seasons
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, format="mixed")
        df["season"] = f.stem
        frames.append(df)
    out = pd.concat(frames, ignore_index=True).sort_values("Date", kind="stable")
    out = out.dropna(subset=ODDS).reset_index(drop=True)
    bad = ~out["FTR"].isin(["H", "D", "A"])
    if bad.any():
        raise ValueError(f"{bad.sum()} rows have a result other than H/D/A")
    return out


def make_synthetic(folder: str | Path, seasons: int = 6, teams: int = 20,
                   margin: float = 0.05, seed: int = 7) -> None:
    """Write `seasons` fake league seasons in football-data.co.uk format.

    Each team has a hidden strength that drifts between seasons. Goals are
    Poisson. The fake bookmaker knows the true probabilities up to a little
    noise and adds `margin` on top - so on this data the bookmaker should
    beat any model, which is a useful sanity check.
    """
    rng = np.random.default_rng(seed)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    names = [f"Team{i:02d}" for i in range(teams)]
    strength = rng.normal(0, 0.35, teams)
    openness = rng.normal(0, 0.10, teams)          # some teams' matches have more goals
    goals = np.arange(0, 11)
    next_id = teams
    for s in range(seasons):
        rows = []
        day = pd.Timestamp(2019 + s, 8, 10)
        fixtures = [(h, a) for h in range(teams) for a in range(teams) if h != a]
        rng.shuffle(fixtures)
        for i, (h, a) in enumerate(fixtures):
            lam_h = np.exp(0.30 + strength[h] - strength[a] + openness[h] + openness[a])   # 0.30 = home edge
            lam_a = np.exp(0.05 + strength[a] - strength[h] + openness[h] + openness[a])
            ph = _pois(goals, lam_h)
            pa = _pois(goals, lam_a)
            grid = np.outer(ph, pa)
            p = np.array([np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()])
            p = p / p.sum()
            gh, ga = rng.poisson(lam_h), rng.poisson(lam_a)
            noisy = p * np.exp(rng.normal(0, 0.03, 3))
            noisy = noisy / noisy.sum()
            odds = np.round(1 / (noisy * (1 + margin)), 2)
            total = np.add.outer(goals, goals)
            p_over = grid[total > 2.5].sum() / grid.sum()
            p_ou = np.array([p_over, 1 - p_over]) * np.exp(rng.normal(0, 0.02, 2))
            ou = np.round(1 / (p_ou / p_ou.sum() * (1 + margin)), 2)
            rows.append({
                "Date": (day + pd.Timedelta(days=i // 10 * 7)).strftime("%d/%m/%Y"),
                "HomeTeam": names[h], "AwayTeam": names[a],
                "FTHG": gh, "FTAG": ga,
                "FTR": "H" if gh > ga else ("D" if gh == ga else "A"),
                "B365H": odds[0], "B365D": odds[1], "B365A": odds[2],
                "B365>2.5": ou[0], "B365<2.5": ou[1],
            })
        pd.DataFrame(rows).to_csv(folder / f"synthetic_{2019 + s}.csv", index=False)
        strength = 0.85 * strength + rng.normal(0, 0.12, teams)
        for t in np.argsort(strength)[:3]:         # bottom three go down, weaker new teams come up
            names[t] = f"Team{next_id:02d}"
            next_id += 1
            strength[t], openness[t] = rng.normal(-0.35, 0.15), rng.normal(0, 0.10)


def _pois(k, lam):
    return poisson.pmf(k, lam)


START = 1500.0


ELO_V1 = dict(k=20.0, home_adv=60.0, carry=1.0, promoted_start=None)   # the step 1 settings


def add_elo(matches: pd.DataFrame, k: float = 20.0, home_adv: float = 60.0,
            carry: float = 1.0, promoted_start: float | None = None) -> pd.DataFrame:
    """Return a copy of `matches` (must be date-sorted) with pre-match
    elo_home, elo_away and elo_diff (home minus away, before home advantage).

    At the start of each new season:
      - a team that also played the previous season keeps `carry` of its distance
        from 1500 (carry=1 keeps everything, carry=0.8 pulls it 20% back to average);
      - a team that did not (promoted, or back after relegation) starts at `promoted_start`.
        With promoted_start=None it keeps whatever rating it last had, or 1500 if it is new.
    The defaults (carry=1, promoted_start=None) are the plain step 1 ratings: nothing happens between seasons.
    """
    if not matches["Date"].is_monotonic_increasing:
        raise ValueError("matches must be sorted by date")
    seasons = matches["season"] if "season" in matches.columns else pd.Series("all", index=matches.index)
    squads = {}
    for se, h, a in zip(seasons, matches["HomeTeam"], matches["AwayTeam"]):
        squads.setdefault(se, set()).update((h, a))
    rating: dict[str, float] = {}
    eh, ea = [], []
    current = None
    for se, home, away, res in zip(seasons, matches["HomeTeam"], matches["AwayTeam"], matches["FTR"]):
        if se != current:
            if current is not None:
                for t in squads[se]:
                    if t in squads[current] and t in rating:
                        rating[t] = START + carry * (rating[t] - START)
                    elif promoted_start is not None:
                        rating[t] = promoted_start
            current = se
        rh, ra = rating.get(home, START), rating.get(away, START)
        eh.append(rh)                      # stored BEFORE the update: no look-ahead
        ea.append(ra)
        expected_home = 1.0 / (1.0 + 10 ** (-(rh + home_adv - ra) / 400.0))
        score_home = {"H": 1.0, "D": 0.5, "A": 0.0}[res]
        delta = k * (score_home - expected_home)
        rating[home], rating[away] = rh + delta, ra - delta
    out = matches.copy()
    out["elo_home"], out["elo_away"] = eh, ea
    out["elo_diff"] = out["elo_home"] - out["elo_away"]
    out.attrs["final_ratings"] = rating
    return out


CLASSES = ["H", "D", "A"]
ODDS = ["B365H", "B365D", "B365A"]


def implied_probs(odds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """odds (n,3) -> (fair probabilities (n,3), overround (n,))."""
    raw = 1.0 / np.asarray(odds, dtype=float)
    total = raw.sum(axis=1)
    return raw / total[:, None], total - 1.0


def split_by_season(df: pd.DataFrame, test_seasons: int = 1):
    """Train on the past, test on the most recent season(s). Never shuffle:
    a random split would let the model train on matches played after the
    ones it is tested on."""
    seasons = list(dict.fromkeys(df["season"]))
    if len(seasons) <= test_seasons:
        raise ValueError("need more seasons than test_seasons")
    test = df["season"].isin(seasons[-test_seasons:])
    return df[~test], df[test]


_SCALE = {"elo_diff": 100.0}        # Elo points are divided by 100 so all features are of similar size


def _design(df: pd.DataFrame, features) -> np.ndarray:
    return np.column_stack([df[f].to_numpy(dtype=float) / _SCALE.get(f, 1.0) for f in features])


def fit(train: pd.DataFrame, features=("elo_diff",)):
    """Multinomial logistic regression on the chosen pre-match features."""
    model = LogisticRegression(C=1e6, max_iter=1000)  # C large = no regularisation
    model.fit(_design(train, features), train["FTR"])
    model.features_ = list(features)
    return model


def predict(model, df: pd.DataFrame) -> np.ndarray:
    """(n,3) probabilities in H, D, A order."""
    p = model.predict_proba(_design(df, model.features_))
    order = [list(model.classes_).index(c) for c in CLASSES]
    return p[:, order]


def one_hot(results: pd.Series) -> np.ndarray:
    return np.stack([(results == c).to_numpy(dtype=float) for c in CLASSES], axis=1)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error of the probabilities. 0 = perfect, lower is better.
    Always saying 1/3 each scores 0.667."""
    return float(((p - y) ** 2).sum(axis=1).mean())


def log_loss(p: np.ndarray, y: np.ndarray) -> float:
    """Punishes confident wrong answers harder than Brier. Lower is better.
    Always saying 1/3 each scores 1.099."""
    return float(-(y * np.log(np.clip(p, 1e-12, 1))).sum(axis=1).mean())


def evaluate(train: pd.DataFrame, test: pd.DataFrame, model) -> pd.DataFrame:
    """Score three forecasters on the test set: base rates, our model, the bookmaker."""
    y = one_hot(test["FTR"])
    base = np.tile(one_hot(train["FTR"]).mean(axis=0), (len(test), 1))
    ours = predict(model, test)
    book, _ = implied_probs(test[ODDS].to_numpy())
    rows = [(name, brier(p, y), log_loss(p, y))
            for name, p in [("base rates", base), ("our model", ours), ("bookmaker", book)]]
    return pd.DataFrame(rows, columns=["forecaster", "brier", "log_loss"]).round(4)


# ---- step 2: pricing ----
def _solve_power(p, target):
    """Find k per row so that sum_i p_i**k == target (bisection; the sum falls as k rises)."""
    p = np.atleast_2d(np.asarray(p, dtype=float))
    lo, hi = np.full(len(p), 0.01), np.full(len(p), 10.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        too_big = (p ** mid[:, None]).sum(axis=1) > target
        lo, hi = np.where(too_big, mid, lo), np.where(too_big, hi, mid)
    return (lo + hi) / 2


def booked_probs(p, margin, method="proportional"):
    """Fair probabilities (n,3) -> inflated probabilities summing to 1 + margin."""
    p = np.atleast_2d(np.asarray(p, dtype=float))
    if margin < 0:
        raise ValueError("margin cannot be negative")
    if method == "proportional":
        return p * (1 + margin)
    if method == "power":
        return p ** _solve_power(p, 1 + margin)[:, None]
    raise ValueError(f"unknown method {method!r}")


def price(p, margin, method="proportional", tick=0.01):
    """Fair probabilities -> decimal odds offered to the customer.
    Odds are rounded DOWN to the tick, so rounding never gives margin away."""
    odds = 1.0 / booked_probs(p, margin, method)
    if tick:
        odds = np.floor(odds / tick + 1e-9) * tick
    return np.maximum(odds, 1.01)


def remove_margin(odds, method="proportional"):
    """The inverse: offered odds -> the fair probabilities behind them."""
    raw = 1.0 / np.atleast_2d(np.asarray(odds, dtype=float))
    if method == "proportional":
        return raw / raw.sum(axis=1, keepdims=True)
    if method == "power":
        return raw ** _solve_power(raw, 1.0)[:, None]
    raise ValueError(f"unknown method {method!r}")


def edge(odds, q):
    """Operator's expected profit per unit staked, if the true probability is q."""
    return 1.0 - np.asarray(q) * np.asarray(odds)


# ---- step 1 (continued): calibration ----
def calibration(p, y, bins=5):
    """Bucket predicted probabilities and compare with how often they happened."""
    flat_p, flat_y = p.ravel(), y.ravel()
    edges = np.quantile(flat_p, np.linspace(0, 1, bins + 1))
    idx = np.clip(np.searchsorted(edges, flat_p, side="right") - 1, 0, bins - 1)
    t = pd.DataFrame({"bin": idx, "predicted": flat_p, "observed": flat_y})
    return t.groupby("bin").agg(predicted=("predicted", "mean"),
                                observed=("observed", "mean"),
                                n=("observed", "size")).round(3)


# ---- step 2 (continued): exposure ----
def exposure(p_ours, p_ref, margin, method="proportional"):
    """Share of offered prices a bettor holding p_ref could bet with positive
    expected value, and the bettor's average edge on those prices."""
    o = price(p_ours, margin, method)
    bettor_ev = p_ref * o - 1.0                 # the bettor's expected profit per unit
    exposed = bettor_ev > 0
    return exposed.mean(), (bettor_ev[exposed].mean() if exposed.any() else 0.0)


# ---- step 3: operator profit ----
BASE = dict(
    margin=0.06,            # our overround
    method="power",         # how the margin is spread (step 2)
    casual_turnover=10_000, # casual money per match at the market margin
    casual_info=0.0,        # share of casual money that follows true probabilities
    casual_noise=0.3,       # how unevenly casual money lands (0 = perfectly balanced book)
    sharp_limit=200,        # max sharp stake we accept per price (0 = no sharps)
    min_edge=0.02,          # sharps only bet if expected value beats this
    elasticity=8.0,         # ASSUMED: how fast casual turnover falls as margin rises
    market_margin=0.055,    # the margin customers can get elsewhere
)


def build_book(p_op, p_true, margin, method, casual_turnover, casual_info, casual_noise,
               sharp_limit, min_edge, elasticity, market_margin, tick=0.01, seed=0):
    """Prices and stakes for every match. Returns (odds, casual stakes, sharp stakes), each (n, 3)."""
    odds = price(p_op, margin, method, tick=tick)
    shown = (1 / odds) / (1 / odds).sum(axis=1, keepdims=True)       # what the odds imply
    split = (1 - casual_info) * shown + casual_info * p_true
    if casual_noise:
        split = split * np.exp(np.random.default_rng(seed).normal(0, casual_noise, split.shape))
        split = split / split.sum(axis=1, keepdims=True)
    volume = casual_turnover * np.exp(-elasticity * (margin - market_margin))
    casual = volume * split
    sharp = np.where(p_true * odds - 1 > min_edge, float(sharp_limit), 0.0)
    return odds, casual, sharp


def expected_profit(odds, stakes, p_true):
    """Exact expected profit: each unit staked earns the operator 1 - (true prob * odds)."""
    return float((stakes * (1 - p_true * odds)).sum())


def simulate(p_op, p_true, n_sims=5000, seed=0, tick=0.01, **params):
    """Play the season n_sims times. Returns per-season operator profit and a summary."""
    cfg = {**BASE, **params}
    odds, casual, sharp = build_book(p_op, p_true, tick=tick, seed=seed, **cfg)
    stakes = casual + sharp
    rng = np.random.default_rng(seed + 1)
    cum = np.cumsum(p_true, axis=1)
    u = rng.random((n_sims, len(p_true)))
    winner = (u[:, :, None] > cum[None, :, :]).sum(axis=2).clip(max=p_true.shape[1] - 1)   # (sims, matches)
    payout = np.take_along_axis((stakes * odds)[None, :, :], winner[:, :, None], axis=2)[:, :, 0]
    profit = stakes.sum() - payout.sum(axis=1)                         # per simulated season
    turnover = stakes.sum()
    return {
        "profit": profit,
        "turnover": turnover,
        "sharp turnover share": sharp.sum() / turnover,
        "expected profit": expected_profit(odds, stakes, p_true),
        "from casuals": expected_profit(odds, casual, p_true),
        "from sharps": expected_profit(odds, sharp, p_true),
        "hold %": 100 * profit.mean() / turnover,
        "worst 5% season": float(np.quantile(profit, 0.05)),
        "P(losing season)": float((profit < 0).mean()),
    }


def summary(label, r):
    return {"scenario": label, "turnover": round(r["turnover"]),
            "profit from casuals": round(r["from casuals"]), "profit from sharps": round(r["from sharps"]),
            "expected profit": round(r["expected profit"]), "hold %": round(r["hold %"], 2),
            "worst 5% season": round(r["worst 5% season"]), "P(losing season)": round(r["P(losing season)"], 3)}


def settle(p_op, p_true, y, **params):
    """Profit of the book on the results that actually occurred (y is one-hot, (n, 3))."""
    odds, casual, sharp = build_book(p_op, p_true, **{**BASE, **params})
    stakes = casual + sharp
    return float(stakes.sum() - (stakes * odds * y).sum()), float(stakes.sum())


def break_even_limit(p_op, p_true, **params):
    """Sharp stake limit per price at which expected season profit reaches zero.
    The loss to sharps is linear in the limit, so this is exact. inf if sharps find nothing."""
    cfg = {**BASE, **params}
    odds, casual, sharp1 = build_book(p_op, p_true, **{**cfg, "sharp_limit": 1})
    cost = -expected_profit(odds, sharp1, p_true)
    return expected_profit(odds, casual, p_true) / cost if cost > 0 else float("inf")


def find_data(root="."):
    """(folder, is_real). Real season CSVs in data/real if present, else synthetic ones."""
    real, synth = Path(root) / "data" / "real", Path(root) / "data" / "synthetic"
    if list(real.glob("*.csv")):
        return real, True
    if not list(synth.glob("*.csv")):
        make_synthetic(synth)
    return synth, False


# ---- step 4: improvements ----------------------------------------------------
# 4a. more features, all computed from matches played BEFORE the one being predicted
F_ELO = ["elo_diff"]
F_FORM = ["elo_diff", "form_diff"]
F_GD = ["elo_diff", "gd_diff"]
F_ALL = ["elo_diff", "form_diff", "gd_diff"]
GOAL_FEATURES = ["elo_diff", "gd_diff", "tot_goals"]


def add_features(matches: pd.DataFrame, form_n: int = 5, goals_n: int = 10) -> pd.DataFrame:
    """Add rolling pre-match features:
      form_diff  points per game over each team's last `form_n` matches, home minus away
      gd_diff    goal difference per game over the last `goals_n` matches, home minus away
      tot_goals  goals scored + conceded per game over the last `goals_n`, averaged over the two teams
    A team with no history yet gets the league average so far."""
    if not matches["Date"].is_monotonic_increasing:
        raise ValueError("matches must be sorted by date")
    hist = {}                                   # team -> list of (points, goals for, goals against)
    all_pts, all_tot, n_seen = 0.0, 0.0, 0
    form, gd, tot = [], [], []

    def stats(team):
        h = hist.get(team, [])
        if not h:
            avg_pts = all_pts / n_seen if n_seen else 1.35
            avg_tot = all_tot / n_seen if n_seen else 2.7
            return avg_pts, 0.0, avg_tot
        f, g = h[-form_n:], h[-goals_n:]
        return (sum(x[0] for x in f) / len(f),
                sum(x[1] - x[2] for x in g) / len(g),
                sum(x[1] + x[2] for x in g) / len(g))

    for home, away, gh, ga in zip(matches["HomeTeam"], matches["AwayTeam"], matches["FTHG"], matches["FTAG"]):
        ph, dh, th = stats(home)
        pa, da, ta = stats(away)
        form.append(ph - pa); gd.append(dh - da); tot.append((th + ta) / 2)
        pts_h = 3 if gh > ga else (1 if gh == ga else 0)
        pts_a = 3 if ga > gh else (1 if gh == ga else 0)
        hist.setdefault(home, []).append((pts_h, gh, ga))
        hist.setdefault(away, []).append((pts_a, ga, gh))
        all_pts += pts_h + pts_a; all_tot += 2 * (gh + ga); n_seen += 2
    out = matches.copy()
    out["form_diff"], out["gd_diff"], out["tot_goals"] = form, gd, tot
    return out


def prepare(raw: pd.DataFrame, **elo_params) -> pd.DataFrame:
    """Raw matches -> matches with Elo and rolling features."""
    return add_features(add_elo(raw, **elo_params))


# 4b. tune the Elo settings, using training seasons only
GRID = dict(k=[10.0, 20.0, 30.0], home_adv=[40.0, 70.0, 100.0],
            carry=[0.6, 0.8, 1.0], promoted_start=[1400.0, 1450.0, 1500.0])


def rolling_log_loss(df: pd.DataFrame, features=F_ALL, min_train: int = 1) -> float:
    """Average log loss when each season is predicted from the seasons before it."""
    seasons = list(dict.fromkeys(df["season"]))
    total, n = 0.0, 0
    for i in range(min_train, len(seasons)):
        tr, te = df[df["season"].isin(seasons[:i])], df[df["season"] == seasons[i]]
        total += log_loss(predict(fit(tr, features), te), one_hot(te["FTR"])) * len(te)
        n += len(te)
    if n == 0:
        raise ValueError("need at least two seasons to tune on")
    return total / n


def tune_elo(train_raw: pd.DataFrame, features=F_ALL, grid=GRID):
    """Try every combination in `grid` and keep the one with the lowest rolling log loss.
    Pass TRAINING seasons only: the test season must never influence these choices.
    Returns (best settings, table of all combinations sorted best first)."""
    import itertools
    rows = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), values))
        rows.append({**params, "rolling log loss": rolling_log_loss(prepare(train_raw, **params), features)})
    table = pd.DataFrame(rows).sort_values("rolling log loss").reset_index(drop=True)
    best = {k_: float(table.loc[0, k_]) for k_ in grid}
    return best, table


FEATURE_SETS = {"elo": F_ELO, "elo + form": F_FORM, "elo + goal difference": F_GD, "elo + form + goal difference": F_ALL}


def tune_model(train_raw: pd.DataFrame, grid=GRID):
    """Choose the Elo settings, then choose which extra features earn their place,
    both by rolling log loss on TRAINING seasons only.
    Returns (elo settings, feature list, table with one row per feature set)."""
    best, _ = tune_elo(train_raw, F_ELO, grid)
    df = prepare(train_raw, **best)
    table = pd.DataFrame([{"features": name, "rolling log loss": rolling_log_loss(df, feats)}
                          for name, feats in FEATURE_SETS.items()]).sort_values("rolling log loss").reset_index(drop=True)
    return best, FEATURE_SETS[table.loc[0, "features"]], table


# 4c. walk-forward: is the result stable from season to season?
def walk_forward(raw: pd.DataFrame, min_train: int = 2, grid=GRID) -> pd.DataFrame:
    """For every season after the first `min_train`: tune and train on all earlier seasons,
    then score that season. Nothing from a tested season is used to choose anything."""
    seasons = list(dict.fromkeys(raw["season"]))
    rows = []
    for i in range(min_train, len(seasons)):
        past, upto = raw[raw["season"].isin(seasons[:i])], raw[raw["season"].isin(seasons[:i + 1])]
        best, features, _ = tune_model(past, grid)
        res = {"test season": seasons[i], "train seasons": i,
               "v2 features": next(n_ for n_, f_ in FEATURE_SETS.items() if f_ == features)}
        for name, params, feats in [("elo only (v1)", ELO_V1, F_ELO), ("improved (v2)", best, features)]:
            df = prepare(upto, **params)
            tr, te = df[df["season"] != seasons[i]], df[df["season"] == seasons[i]]
            y = one_hot(te["FTR"])
            res[name] = log_loss(predict(fit(tr, feats), te), y)
        res["base rates"] = log_loss(np.tile(one_hot(tr["FTR"]).mean(axis=0), (len(te), 1)), y)
        res["bookmaker"] = log_loss(implied_probs(te[ODDS].to_numpy())[0], y)
        gap = res["base rates"] - res["bookmaker"]
        res["gap closed v1"] = (res["base rates"] - res["elo only (v1)"]) / gap
        res["gap closed v2"] = (res["base rates"] - res["improved (v2)"]) / gap
        rows.append(res)
    cols = ["test season", "train seasons", "base rates", "elo only (v1)", "improved (v2)", "bookmaker",
            "gap closed v1", "gap closed v2", "v2 features"]
    return pd.DataFrame(rows)[cols]


# 4d. a goals model, and the markets that follow from it
def fit_goals(train: pd.DataFrame, features=GOAL_FEATURES):
    """Two Poisson regressions: expected home goals and expected away goals."""
    X = train[list(features)].to_numpy(dtype=float)
    make = lambda: make_pipeline(StandardScaler(), PoissonRegressor(alpha=0.0, max_iter=1000))
    return {"home": make().fit(X, train["FTHG"]), "away": make().fit(X, train["FTAG"]), "features": list(features)}


def predict_goals(gm, df: pd.DataFrame):
    X = df[gm["features"]].to_numpy(dtype=float)
    return gm["home"].predict(X), gm["away"].predict(X)


def score_grid(lam_h, lam_a, max_goals: int = 10) -> np.ndarray:
    """(n, G, G): probability of every scoreline, home goals on axis 1, away goals on axis 2.
    Assumes the two teams' goals are independent Poisson counts."""
    g = np.arange(max_goals + 1)
    ph = poisson.pmf(g[None, :], np.atleast_1d(lam_h)[:, None])
    pa = poisson.pmf(g[None, :], np.atleast_1d(lam_a)[:, None])
    grid = ph[:, :, None] * pa[:, None, :]
    return grid / grid.sum(axis=(1, 2), keepdims=True)


def _diff_total(grid):
    g = np.arange(grid.shape[1])
    return np.subtract.outer(g, g), np.add.outer(g, g)       # home minus away, total goals


def market_1x2(grid) -> np.ndarray:
    diff, _ = _diff_total(grid)
    return np.stack([grid[:, diff > 0].sum(axis=1), grid[:, diff == 0].sum(axis=1), grid[:, diff < 0].sum(axis=1)], axis=1)


def market_over_under(grid, line: float = 2.5) -> np.ndarray:
    """(n, 2): P(over), P(under). Use a half line so there is no push."""
    _, total = _diff_total(grid)
    over = grid[:, total > line].sum(axis=1)
    return np.stack([over, 1 - over], axis=1)


def over_under_result(df: pd.DataFrame, line: float = 2.5) -> np.ndarray:
    over = ((df["FTHG"] + df["FTAG"]) > line).to_numpy(dtype=float)
    return np.stack([over, 1 - over], axis=1)


def handicap_fair_odds(grid, line: float) -> np.ndarray:
    """(n, 2): fair decimal odds for home and away on an Asian handicap.
    `line` is added to the home score: -0.5 means home must win, +0.5 means home must not lose.
    Whole lines can push (stake returned). Quarter lines split the stake over the two nearest half-steps."""
    diff, _ = _diff_total(grid)
    parts = [line] if (line * 2) == int(line * 2) else [line - 0.25, line + 0.25]
    win_h = np.mean([grid[:, diff + l > 0].sum(axis=1) for l in parts], axis=0)
    push = np.mean([grid[:, diff + l == 0].sum(axis=1) for l in parts], axis=0)
    win_a = 1 - win_h - push
    return np.stack([(1 - push) / win_h, (1 - push) / win_a], axis=1)


def handicap_return(home_goals: int, away_goals: int, line: float, odds: float, side: str = "home") -> float:
    """What a 1-unit handicap bet returns (stake included) for a given final score."""
    parts = [line] if (line * 2) == int(line * 2) else [line - 0.25, line + 0.25]
    total = 0.0
    for l in parts:
        d = (home_goals - away_goals + l) * (1 if side == "home" else -1)
        total += (odds if d > 0 else (1.0 if d == 0 else 0.0)) / len(parts)
    return total


def evaluate_over_under(train: pd.DataFrame, test: pd.DataFrame, gm, line: float = 2.5) -> pd.DataFrame:
    """Base rate, our goals model and the bookmaker on over/under, on matches that have those odds."""
    te = test.dropna(subset=OU_ODDS)
    if te.empty:
        raise ValueError("no over/under odds in the test season")
    y = over_under_result(te, line)
    base = np.tile(over_under_result(train, line).mean(axis=0), (len(te), 1))
    ours = market_over_under(score_grid(*predict_goals(gm, te)), line)
    book = remove_margin(te[OU_ODDS].to_numpy())
    rows = [(name, brier(p, y), log_loss(p, y))
            for name, p in [("base rates", base), ("our goals model", ours), ("bookmaker", book)]]
    return pd.DataFrame(rows, columns=["forecaster", "brier", "log_loss"]).round(4)


# ---- step 5: prices that move when money arrives ---------------------------------
# Until now every bet on a match was struck at one price. Here money arrives in
# `n_clips` rounds and the operator may change its prices between rounds.
#   fixed        prices never move (identical to build_book / simulate)
#   information  a sportsbook that shifts its probability towards money it did not expect
#   lmsr         an automated market maker (logarithmic market scoring rule), as used by prediction markets
RULES = ["fixed", "information", "lmsr"]
DYNAMIC = dict(
    rule="information",
    n_clips=10,          # rounds of betting per match; a sharp's single bet is at most sharp_limit / n_clips
    alpha=1.0,           # information rule: speed. Log-odds shift per 1,000 units of unexpected money
    casual_weight=0.1,   # information rule: weight on casual money (0 = only known sharp accounts move the price,
                         #                   1 = the operator cannot tell customers apart)
    lmsr_b=2000.0,       # lmsr: liquidity. Larger = prices move less per unit of money
)


def _lmsr_prices(q, b):
    e = np.exp((q - q.max(axis=1, keepdims=True)) / b)
    return e / e.sum(axis=1, keepdims=True)


def _lmsr_cost(q, b):
    """C(q) = b * ln(sum_j exp(q_j / b)), computed without overflow."""
    top = q.max(axis=1)
    return top + b * np.log(np.exp((q - top[:, None]) / b).sum(axis=1))


def _lmsr_buy(q, b, i, cost):
    """Spend `cost` (n,) on outcome i at the market maker's prices. Updates q in place
    and returns the shares bought; each share pays 1 if outcome i happens.
    Cost function C(q) = b * ln(sum_j exp(q_j / b)); a purchase costs C(after) - C(before)."""
    shift = q.max(axis=1)
    e = np.exp((q - shift[:, None]) / b)
    tot = e.sum(axis=1)
    new_ei = tot * np.exp(cost / b) - (tot - e[:, i])
    shares = np.where(cost > 0, b * np.log(new_ei) + shift - q[:, i], 0.0)
    q[:, i] += shares
    return shares


def _lmsr_basket(q, b, direction, cost):
    """Shares bought when `cost` (n,) is spent on a basket in the proportions `direction` (n, k).
    Buying moves the price as you go, so the basket is scaled until it costs exactly `cost`."""
    c0 = _lmsr_cost(q, b)
    lo, hi = np.zeros(len(q)), np.ones(len(q))
    for _ in range(50):
        mid = (lo + hi) / 2
        too_much = _lmsr_cost(q + mid[:, None] * direction, b) - c0 > cost
        lo, hi = np.where(too_much, lo, mid), np.where(too_much, mid, hi)
    return ((lo + hi) / 2)[:, None] * direction


def build_book_dynamic(p_op, p_true, seed=0, tick=0.01, **params):
    """Play the betting on every match round by round. Returns a dict with the money taken
    ('paid') and the payout owed per outcome ('liab'), each split by customer type, plus the
    path of displayed odds, shape (n_clips + 1, n, k)."""
    cfg = {**BASE, **DYNAMIC, **params}
    rule, T = cfg["rule"], int(cfg["n_clips"])
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}")
    p_op, p_true = np.asarray(p_op, dtype=float), np.asarray(p_true, dtype=float)
    n, k = p_op.shape
    margin, limit, clip = cfg["margin"], float(cfg["sharp_limit"]), float(cfg["sharp_limit"]) / T
    volume = cfg["casual_turnover"] * np.exp(-cfg["elasticity"] * (margin - cfg["market_margin"]))
    noise = (np.exp(np.random.default_rng(seed).normal(0, cfg["casual_noise"], (n, k)))
             if cfg["casual_noise"] else np.ones((n, k)))
    paid = {"casual": np.zeros((n, k)), "sharp": np.zeros((n, k))}
    liab = {"casual": np.zeros((n, k)), "sharp": np.zeros((n, k))}
    path = []

    def casual_demand(shown_open):
        """Casual money per round at the OPENING prices. The public's view is anchored on the opening line."""
        s = ((1 - cfg["casual_info"]) * shown_open + cfg["casual_info"] * p_true) * noise
        return volume / T * s / s.sum(axis=1, keepdims=True)

    def reprice(base, odds, odds_open):
        """Casual customers walk away from a price that has got worse: if an outcome's odds fall below
        where they opened, less casual money bets it, using the same assumed elasticity as the margin
        (a 1% worse price loses about `elasticity`% of that money). A price that has got better does
        NOT attract extra casual money here, so moving prices can only cost casual turnover."""
        return base * np.minimum((odds / odds_open) ** cfg["elasticity"], 1.0)

    if rule == "lmsr":
        b, fee = float(cfg["lmsr_b"]), margin              # the fee plays the role of the margin
        q = b * np.log(p_op)                               # opening prices = our model's probabilities
        odds_open = 1 / (p_op * (1 + fee))
        base = casual_demand(p_op)
        for _ in range(T):
            path.append(1 / (_lmsr_prices(q, b) * (1 + fee)))
            for i in range(k):                             # sharps buy while the price is worth it to them
                p_i = np.minimum(_lmsr_prices(q, b)[:, i], 1 - 1e-12)
                target = np.minimum(p_true[:, i] / ((1 + fee) * (1 + cfg["min_edge"])), 0.999)
                budget = np.minimum(clip, limit - paid["sharp"][:, i]) / (1 + fee)
                to_target = b * np.log((1 - p_i) / (1 - target))
                cost = np.where(target > p_i, np.clip(np.minimum(to_target, budget), 0, None), 0.0)
                liab["sharp"][:, i] += _lmsr_buy(q, b, i, cost)
                paid["sharp"][:, i] += cost * (1 + fee)
            prices = _lmsr_prices(q, b)                    # casual money arrives as one basket of shares
            spend = reprice(base, 1 / (prices * (1 + fee)), odds_open)
            shares = _lmsr_basket(q, b, spend / prices, spend.sum(axis=1) / (1 + fee))
            q += shares
            liab["casual"] += shares
            paid["casual"] += spend
        final_p = _lmsr_prices(q, b)
        path.append(1 / (final_p * (1 + fee)))
    else:
        p = p_op.copy()
        odds_open = price(p, margin, cfg["method"], tick=tick)
        base = casual_demand((1 / odds_open) / (1 / odds_open).sum(axis=1, keepdims=True))
        for _ in range(T):
            odds = price(p, margin, cfg["method"], tick=tick)
            path.append(odds)
            shown = (1 / odds) / (1 / odds).sum(axis=1, keepdims=True)
            room = limit - paid["sharp"]
            sharp = np.where((p_true * odds - 1 > cfg["min_edge"]) & (room > 1e-9), np.minimum(clip, room), 0.0)
            casual = reprice(base, odds, odds_open)
            for who, stake in (("sharp", sharp), ("casual", casual)):
                paid[who] += stake
                liab[who] += stake * odds
            if rule == "information" and cfg["alpha"]:
                seen = cfg["casual_weight"] * casual + sharp            # money, weighted by how much we trust it
                surprise = seen - shown * seen.sum(axis=1, keepdims=True)
                p = p * np.exp(np.clip(cfg["alpha"] * surprise / 1000.0, -0.3, 0.3))   # at most ~30% per round
                p = p / p.sum(axis=1, keepdims=True)
        final_p = p
        path.append(price(p, margin, cfg["method"], tick=tick))
    return {"paid": paid, "liab": liab, "path": np.array(path), "final_p": final_p}


def _season_draws(p_true, n_sims, seed):
    rng = np.random.default_rng(seed + 1)
    cum = np.cumsum(p_true, axis=1)
    u = rng.random((n_sims, len(p_true)))
    return (u[:, :, None] > cum[None, :, :]).sum(axis=2).clip(max=p_true.shape[1] - 1)


def simulate_dynamic(p_op, p_true, n_sims=5000, seed=0, tick=0.01, **params):
    """Like simulate(), but the betting on each match happens in rounds and prices may move.
    Returns the same summary, plus the share of closing prices that are still exposed."""
    cfg = {**BASE, **DYNAMIC, **params}
    book = build_book_dynamic(p_op, p_true, seed=seed, tick=tick, **params)
    paid_all = book["paid"]["casual"] + book["paid"]["sharp"]
    liab_all = book["liab"]["casual"] + book["liab"]["sharp"]
    turnover = paid_all.sum()
    winner = _season_draws(p_true, n_sims, seed)
    payout = np.take_along_axis(liab_all[None, :, :], winner[:, :, None], axis=2)[:, :, 0]
    profit = turnover - payout.sum(axis=1)
    ev = lambda who: float(book["paid"][who].sum() - (p_true * book["liab"][who]).sum())
    closing = book["path"][-1]
    return {
        "profit": profit,
        "turnover": turnover,
        "sharp turnover share": book["paid"]["sharp"].sum() / turnover,
        "expected profit": ev("casual") + ev("sharp"),
        "from casuals": ev("casual"),
        "from sharps": ev("sharp"),
        "hold %": 100 * profit.mean() / turnover,
        "worst 5% season": float(np.quantile(profit, 0.05)),
        "P(losing season)": float((profit < 0).mean()),
        "exposed at close": float((p_true * closing - 1 > cfg["min_edge"]).mean()),
        "book": book,
    }


def settle_dynamic(p_op, p_true, y, **params):
    """Profit of the round-by-round book on the results that actually occurred."""
    book = build_book_dynamic(p_op, p_true, **params)
    paid = book["paid"]["casual"] + book["paid"]["sharp"]
    liab = book["liab"]["casual"] + book["liab"]["sharp"]
    return float(paid.sum() - (liab * y).sum()), float(paid.sum())
