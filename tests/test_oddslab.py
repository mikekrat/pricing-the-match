import numpy as np
import pandas as pd
import pytest

import oddslab as ol

P = np.array([[0.50, 0.30, 0.20], [0.80, 0.15, 0.05], [0.34, 0.33, 0.33]])


@pytest.fixture(scope="module")
def matches(tmp_path_factory):
    folder = tmp_path_factory.mktemp("d")
    ol.make_synthetic(folder, seasons=4, teams=12)
    return ol.add_elo(ol.load_matches(folder))


@pytest.fixture(scope="module")
def TM():
    rng = np.random.default_rng(1)
    T = rng.dirichlet([4, 2.5, 3], size=200)
    M = T * np.exp(rng.normal(0, 0.15, T.shape))
    return T, M / M.sum(axis=1, keepdims=True)


# ---- step 1: probabilities
def test_loader(matches):
    assert matches["Date"].is_monotonic_increasing and len(matches) == 4 * 12 * 11


def test_elo_zero_sum_and_start(matches):
    assert matches.loc[0, "elo_home"] == 1500
    assert np.isclose(np.mean(list(matches.attrs["final_ratings"].values())), 1500)


def test_no_lookahead(matches):
    flipped = matches[["Date", "HomeTeam", "AwayTeam", "FTR"]].copy()
    flipped.loc[flipped.index[-1], "FTR"] = "A" if flipped["FTR"].iloc[-1] != "A" else "H"
    assert np.allclose(ol.add_elo(flipped)["elo_diff"], matches["elo_diff"])


def test_unsorted_rejected(matches):
    with pytest.raises(ValueError):
        ol.add_elo(matches.iloc[::-1])


def test_model_output(matches):
    train, test = ol.split_by_season(matches)
    assert train["Date"].max() < test["Date"].min()
    m = ol.fit(train)
    p = ol.predict(m, test)
    assert np.allclose(p.sum(axis=1), 1) and (p >= 0).all()
    home = ol.predict(m, pd.DataFrame({"elo_diff": [-200.0, 0.0, 200.0]}))[:, 0]
    assert home[0] < home[1] < home[2]
    ll = ol.evaluate(train, test, m).set_index("forecaster")["log_loss"]
    assert ll["bookmaker"] < ll["our model"] < ll["base rates"]


def test_scores_by_hand():
    y = np.array([[1.0, 0, 0]])
    assert ol.brier(y, y) == 0
    assert ol.brier(np.full((1, 3), 1 / 3), y) == pytest.approx(2 / 3)
    assert ol.log_loss(np.full((1, 3), 1 / 3), y) == pytest.approx(np.log(3))


# ---- step 2: prices
@pytest.mark.parametrize("method", ["proportional", "power"])
def test_margin_methods(method):
    b = ol.booked_probs(P, 0.06, method)
    assert np.allclose(b.sum(axis=1), 1.06)
    assert (np.argsort(b, axis=1) == np.argsort(P, axis=1)).all()
    assert np.allclose(1 / ol.price(P, 0.0, method, tick=None), P)
    assert np.allclose(ol.remove_margin(ol.price(P, 0.06, method, tick=None), method), P, atol=1e-8)


def test_where_the_margin_goes():
    assert np.allclose(ol.edge(ol.price(P, 0.06, "proportional", tick=None), P), 0.06 / 1.06)
    e = ol.edge(ol.price(P, 0.06, "power", tick=None), P)
    assert e[1, 2] > e[1, 0] and e[0, 2] > e[0, 0]


def test_rounding_never_gives_margin_away():
    fine, rounded = ol.price(P, 0.06, tick=None), ol.price(P, 0.06, tick=0.01)
    assert (rounded <= fine + 1e-12).all() and (1 / rounded).sum(axis=1).min() >= 1.06 - 1e-9


def test_negative_margin_rejected():
    with pytest.raises(ValueError):
        ol.booked_probs(P, -0.01)


def test_exposure():
    wrong = P[:, ::-1]
    shares = [ol.exposure(wrong, P, m)[0] for m in (0.0, 0.05, 0.20, 5.0)]
    assert all(a >= b for a, b in zip(shares, shares[1:])) and shares[-1] < shares[0]
    assert ol.exposure(P, P, 0.0001)[0] == 0


# ---- step 3: profit
def test_simulation_matches_formula(TM):
    T, M = TM
    r = ol.simulate(M, T, n_sims=20000, seed=3)
    se = r["profit"].std() / np.sqrt(len(r["profit"]))
    assert abs(r["profit"].mean() - r["expected profit"]) < 4 * se


def test_balanced_book(TM):
    T, M = TM
    r = ol.simulate(M, T, n_sims=200, sharp_limit=0, casual_noise=0, method="proportional",
                    margin=0.06, tick=None, elasticity=0)
    assert r["expected profit"] / r["turnover"] == pytest.approx(0.06 / 1.06)
    assert r["profit"].std() < 1e-6 * r["turnover"]


def test_sharps(TM):
    T, M = TM
    assert ol.simulate(T, T, n_sims=10)["sharp turnover share"] == 0
    prof = [ol.simulate(M, T, n_sims=10, sharp_limit=L)["expected profit"] for L in (0, 100, 500, 2000)]
    assert all(a > b for a, b in zip(prof, prof[1:]))
    assert ol.simulate(M, T, n_sims=10)["from sharps"] < 0


def test_break_even_limit(TM):
    T, M = TM
    be = ol.break_even_limit(M, T)
    assert ol.simulate(M, T, n_sims=10, sharp_limit=be)["expected profit"] == pytest.approx(0, abs=1e-6 * 1e6)
    assert np.isinf(ol.break_even_limit(T, T))


def test_reproducible_and_settle(TM):
    T, M = TM
    assert np.allclose(ol.simulate(M, T, n_sims=100, seed=5)["profit"], ol.simulate(M, T, n_sims=100, seed=5)["profit"])
    y = np.eye(3)[np.random.default_rng(2).integers(0, 3, len(T))]
    odds, cas, shp = ol.build_book(M, T, **ol.BASE)
    by_hand = sum((cas[i] + shp[i]).sum() - ((cas[i] + shp[i]) * odds[i] * y[i]).sum() for i in range(len(T)))
    assert ol.settle(M, T, y)[0] == pytest.approx(by_hand)


# ---- step 4: improvements
@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    folder = tmp_path_factory.mktemp("r")
    ol.make_synthetic(folder, seasons=5, teams=12)
    return ol.load_matches(folder)


def plain_elo(df, k=20.0, home_adv=60.0):
    """The step 1 rating loop, written out again here so the module is checked against something independent."""
    rating, diff = {}, []
    for h, a, r in zip(df["HomeTeam"], df["AwayTeam"], df["FTR"]):
        rh, ra = rating.get(h, 1500.0), rating.get(a, 1500.0)
        diff.append(rh - ra)
        d = k * ({"H": 1.0, "D": 0.5, "A": 0.0}[r] - 1 / (1 + 10 ** (-(rh + home_adv - ra) / 400)))
        rating[h], rating[a] = rh + d, ra - d
    return np.array(diff)


def with_returning_team(raw):
    """Rename a team that is new in the third season to one relegated after the first: it now leaves and comes back."""
    s = list(dict.fromkeys(raw["season"]))
    teams = lambda i: set(raw[raw["season"] == s[i]]["HomeTeam"])
    gone = sorted(teams(0) - teams(1))[0]
    new = sorted(teams(2) - teams(1))[0]
    out = raw.copy()
    later = out["season"].isin(s[2:])
    out.loc[later, ["HomeTeam", "AwayTeam"]] = out.loc[later, ["HomeTeam", "AwayTeam"]].replace(new, gone)
    return out, gone, s


def test_default_elo_is_the_step1_elo_even_for_returning_teams(raw):
    back, gone, s = with_returning_team(raw)
    assert np.allclose(ol.add_elo(back)["elo_diff"], plain_elo(back))
    assert np.allclose(ol.add_elo(back, **ol.ELO_V1)["elo_diff"], plain_elo(back))
    e = ol.add_elo(back, promoted_start=1400.0)                  # with a number, the returning team is reset
    first = e[(e["season"] == s[2]) & ((e["HomeTeam"] == gone) | (e["AwayTeam"] == gone))].iloc[0]
    assert (first["elo_home"] if first["HomeTeam"] == gone else first["elo_away"]) == 1400.0


def test_elo_season_rules(raw):
    seasons = list(dict.fromkeys(raw["season"]))
    first, second = raw[raw["season"] == seasons[0]], raw[raw["season"] == seasons[1]]
    old = set(first["HomeTeam"])
    new = set(second["HomeTeam"]) - old
    assert len(new) == 3
    e = ol.add_elo(raw, carry=0.5, promoted_start=1400.0)
    end = ol.add_elo(first).attrs["final_ratings"]
    for team in set(second["HomeTeam"]):
        row = e[(e["season"] == seasons[1]) & ((e["HomeTeam"] == team) | (e["AwayTeam"] == team))].iloc[0]
        start = row["elo_home"] if row["HomeTeam"] == team else row["elo_away"]
        assert start == pytest.approx(1400.0 if team in new else 1500 + 0.5 * (end[team] - 1500))


def test_features_have_no_lookahead(raw):
    a = ol.prepare(raw)
    b = raw.copy()
    last = b.index[-1]
    b.loc[last, ["FTHG", "FTAG", "FTR"]] = [9, 0, "H"]
    b = ol.prepare(b)
    cols = ["elo_diff", "form_diff", "gd_diff", "tot_goals"]
    assert np.allclose(a[cols], b[cols])
    assert a[cols].notna().all().all()


def test_form_by_hand():
    d = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=3), "HomeTeam": ["A", "A", "A"],
                      "AwayTeam": ["B", "B", "B"], "FTHG": [2, 0, 1], "FTAG": [0, 0, 3], "FTR": ["H", "D", "A"]})
    f = ol.add_features(d)
    assert f.loc[1, "form_diff"] == 3 - 0 and f.loc[1, "gd_diff"] == 2 - (-2)
    assert f.loc[2, "form_diff"] == (3 + 1) / 2 - (0 + 1) / 2 and f.loc[2, "tot_goals"] == 1.0


def test_tuning_and_walk_forward(raw):
    small = dict(k=[10.0, 30.0], home_adv=[60.0], carry=[0.7, 1.0], promoted_start=[1400.0, 1500.0])
    train = raw[raw["season"] != raw["season"].iloc[-1]]
    best, table = ol.tune_elo(train, grid=small)
    assert len(table) == 8 and table["rolling log loss"].is_monotonic_increasing
    assert ol.rolling_log_loss(ol.prepare(train, **best)) == pytest.approx(table.loc[0, "rolling log loss"])
    elo, feats, ftable = ol.tune_model(train, grid=small)
    assert feats in ol.FEATURE_SETS.values() and len(ftable) == 4
    wf = ol.walk_forward(raw, min_train=2, grid=small)
    assert list(wf["test season"]) == list(dict.fromkeys(raw["season"]))[2:]
    assert (wf["bookmaker"] < wf["base rates"]).all()


def test_score_grid_and_markets():
    grid = ol.score_grid([1.6, 0.9], [1.1, 1.4])
    assert np.allclose(grid.sum(axis=(1, 2)), 1)
    assert np.allclose(ol.market_1x2(grid).sum(axis=1), 1)
    ou = ol.market_over_under(grid, 2.5)
    assert np.allclose(ou.sum(axis=1), 1)
    from scipy.stats import poisson
    assert ou[0, 1] == pytest.approx(poisson.cdf(2, 1.6 + 1.1), abs=1e-6)     # total goals is Poisson(2.7)
    assert ol.market_1x2(grid)[0, 0] > ol.market_1x2(grid)[1, 0]


@pytest.mark.parametrize("line", [-1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0])
def test_handicap_odds_are_fair(line):
    grid = ol.score_grid([1.6], [1.1])
    odds = ol.handicap_fair_odds(grid, line)[0]
    g = range(grid.shape[1])
    for side, o in zip(["home", "away"], odds):
        ev = sum(grid[0, i, j] * ol.handicap_return(i, j, line, o, side) for i in g for j in g)
        assert ev == pytest.approx(1.0, abs=1e-9)


def test_handicap_special_cases():
    grid = ol.score_grid([1.6], [1.1])
    h, d, a = ol.market_1x2(grid)[0]
    assert ol.handicap_fair_odds(grid, -0.5)[0, 0] == pytest.approx(1 / h)
    assert ol.handicap_fair_odds(grid, 0.0)[0, 0] == pytest.approx((h + a) / h)        # draw no bet
    assert ol.handicap_fair_odds(grid, -0.25)[0, 0] == pytest.approx((1 - d / 2) / h)


def test_goals_model_and_two_way_simulation(raw):
    df = ol.prepare(raw)
    train, test = ol.split_by_season(df)
    gm = ol.fit_goals(train)
    lam_h, lam_a = ol.predict_goals(gm, test)
    assert (lam_h > 0).all() and (lam_a > 0).all() and lam_h.mean() > lam_a.mean()
    t = ol.evaluate_over_under(train, test, gm).set_index("forecaster")["log_loss"]
    assert t["bookmaker"] < t["base rates"]
    p_op = ol.market_over_under(ol.score_grid(lam_h, lam_a))
    p_true = ol.remove_margin(test[ol.OU_ODDS].to_numpy())
    r = ol.simulate(p_op, p_true, n_sims=4000, seed=1)
    se = r["profit"].std() / np.sqrt(4000)
    assert abs(r["profit"].mean() - r["expected profit"]) < 4 * se


# ---- step 5: prices that move
def test_fixed_rule_equals_static_book(TM):
    T, M = TM
    a = ol.simulate(M, T, n_sims=300, seed=4, sharp_limit=500)
    b = ol.simulate_dynamic(M, T, n_sims=300, seed=4, sharp_limit=500, rule="fixed")
    assert np.allclose(a["profit"], b["profit"])
    for key in ["turnover", "expected profit", "from casuals", "from sharps"]:
        assert a[key] == pytest.approx(b[key])
    c = ol.simulate_dynamic(M, T, n_sims=300, seed=4, sharp_limit=500, rule="information", alpha=0.0)
    assert np.allclose(a["profit"], c["profit"])


@pytest.mark.parametrize("rule", ol.RULES)
def test_dynamic_simulation_matches_formula(TM, rule):
    T, M = TM
    r = ol.simulate_dynamic(M, T, n_sims=20000, seed=3, rule=rule, sharp_limit=500)
    se = r["profit"].std() / np.sqrt(20000)
    assert abs(r["profit"].mean() - r["expected profit"]) < 4 * se
    assert (r["book"]["paid"]["sharp"] <= 500 + 1e-6).all()


def test_moving_prices_limit_sharp_losses(TM):
    T, M = TM
    kw = dict(n_sims=10, sharp_limit=5000, casual_weight=0.0)
    fixed = ol.simulate_dynamic(M, T, rule="fixed", **kw)
    moving = ol.simulate_dynamic(M, T, rule="information", alpha=2.0, n_clips=50, **kw)
    assert moving["from sharps"] > fixed["from sharps"]                    # a smaller loss
    assert moving["book"]["paid"]["sharp"].sum() < fixed["book"]["paid"]["sharp"].sum()
    assert moving["exposed at close"] < fixed["exposed at close"]
    # where sharps bet exactly one outcome of a match, our probability for it went up
    bet = moving["book"]["paid"]["sharp"] > 0
    one = bet & (bet.sum(axis=1, keepdims=True) == 1)
    assert one.any() and (moving["book"]["final_p"][one] > M[one]).all()


def test_lmsr_mechanics():
    q = 2000.0 * np.log(np.array([[0.5, 0.3, 0.2], [0.6, 0.25, 0.15]]))
    assert np.allclose(ol._lmsr_prices(q, 2000.0), [[0.5, 0.3, 0.2], [0.6, 0.25, 0.15]])
    C = lambda z: 2000.0 * np.log(np.exp(z / 2000.0).sum(axis=1))
    before, q2 = C(q), q.copy()
    shares = ol._lmsr_buy(q2, 2000.0, 1, np.array([100.0, 0.0]))
    assert C(q2)[0] - before[0] == pytest.approx(100.0)                    # paid exactly the cost difference
    assert shares[1] == 0 and np.allclose(q2[1], q[1])
    assert ol._lmsr_prices(q2, 2000.0)[0, 1] > 0.3
    assert np.allclose(ol._lmsr_cost(q, 2000.0), C(q))


def test_lmsr_balanced_money_leaves_prices_alone(TM):
    T, M = TM                                                              # money split like the prices buys equal shares
    book = ol.build_book_dynamic(M, T, rule="lmsr", sharp_limit=0, casual_noise=0.0, margin=0.05)
    assert np.allclose(book["path"][-1], book["path"][0])
    paid, liab = book["paid"]["casual"], book["liab"]["casual"]
    assert np.allclose(liab, liab[:, [0]])                                 # same payout whoever wins
    assert ((paid.sum(axis=1) - liab[:, 0]) / paid.sum(axis=1)) == pytest.approx(0.05 / 1.05)


def test_lmsr_loss_is_bounded(TM):
    T, M = TM
    b = 800.0
    book = ol.build_book_dynamic(M, T, rule="lmsr", lmsr_b=b, sharp_limit=1e9, margin=0.0, n_clips=20)
    paid = (book["paid"]["casual"] + book["paid"]["sharp"]).sum(axis=1)
    liab = book["liab"]["casual"] + book["liab"]["sharp"]
    worst = (paid[:, None] - liab).min(axis=1)                             # per match, over every possible winner
    assert (worst >= -b * np.log(1 / M.min(axis=1)) - 1e-6).all()
    path = book["path"]
    assert np.allclose(1 / path[0], M)                                     # opens at our model's prices
