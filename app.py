"""pricing-the-match web app.  Run with:  python -m streamlit run app.py"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import pricing as ol

st.set_page_config(page_title="pricing-the-match", layout="wide")


@st.cache_data(show_spinner="Tuning the model on the training seasons (about ten seconds, once)...")
def load():
    folder, is_real = ol.find_data()
    raw = ol.load_matches(folder)
    seasons = list(dict.fromkeys(raw["season"]))
    train_raw = raw[raw["season"] != seasons[-1]]            # the test season never touches a choice

    elo, feats, feature_table = ol.tune_model(train_raw)
    v1_train, v1_test = ol.split_by_season(ol.prepare(raw, **ol.ELO_V1))
    train, test = ol.split_by_season(ol.prepare(raw, **elo))
    test = test.reset_index(drop=True)

    y = ol.one_hot(test["FTR"])
    p_true = ol.remove_margin(test[ol.ODDS].to_numpy())
    p_v1 = ol.predict(ol.fit(v1_train, ol.F_ELO), v1_test)
    p_v2 = ol.predict(ol.fit(train, feats), test)
    base = np.tile(ol.one_hot(train["FTR"]).mean(axis=0), (len(test), 1))
    scores = pd.DataFrame(
        [(n, ol.brier(p, y), ol.log_loss(p, y)) for n, p in
         [("base rates", base), ("step 1 model (v1)", p_v1), ("improved model (v2)", p_v2), ("bookmaker", p_true)]],
        columns=["forecaster", "brier", "log_loss"]).round(4)

    gm = ol.fit_goals(train)
    lam_h, lam_a = ol.predict_goals(gm, test)
    grid = ol.score_grid(lam_h, lam_a)
    has_ou = test[ol.OU_ODDS].notna().all(axis=1).to_numpy()
    ou = None
    if has_ou.any():
        ou_test = test[has_ou]
        ou = {"rows": np.flatnonzero(has_ou),
              "p_model": ol.market_over_under(grid[has_ou]),
              "p_true": ol.remove_margin(ou_test[ol.OU_ODDS].to_numpy()),
              "y": ol.over_under_result(ou_test),
              "odds": ou_test[ol.OU_ODDS].to_numpy(dtype=float),
              "book_margin": float(ol.implied_probs(ou_test[ol.OU_ODDS].to_numpy())[1].mean()),
              "scores": ol.evaluate_over_under(train, test, gm)}
    return {
        "is_real": is_real, "n_train": len(train), "test": test, "seasons": seasons,
        "elo": elo, "features": feats, "feature_table": feature_table.round(4),
        "p_v1": p_v1, "p_v2": p_v2, "p_true": p_true, "y": y, "scores": scores,
        "book_margin": float(ol.implied_probs(test[ol.ODDS].to_numpy())[1].mean()),
        "lam_h": lam_h, "lam_a": lam_a, "grid": grid, "ou": ou,
    }


@st.cache_data(show_spinner="Walk-forward test: tuning and scoring every season in turn...")
def load_walk_forward():
    folder, _ = ol.find_data()
    return ol.walk_forward(ol.load_matches(folder))


D = load()
test = D["test"]
labels = [f"{d:%d %b %Y}  {h} v {a}" for d, h, a in zip(test["Date"], test["HomeTeam"], test["AwayTeam"])]

st.title("pricing-the-match")
st.caption("How a betting operator prices a football match, and what its profit depends on.")
if not D["is_real"]:
    st.warning("No season files found in data/real, so this is running on SYNTHETIC matches. "
               "The numbers show that the code works; they are not findings about football.")

# ---------------------------------------------------------------- the levers
sb = st.sidebar
sb.header("What is being priced")
markets = ["Match result (1X2)"] + (["Over/under 2.5 goals"] if D["ou"] else [])
market = sb.selectbox("Market", markets)
model_name = sb.selectbox("Match-result model", ["Improved (v2)", "Step 1 (v1)"])
sb.header("Levers")
margin = sb.slider("Margin (overround)", 0.0, 0.25, 0.06, 0.01)
method = sb.selectbox("Where the margin goes", ["power", "proportional"])
quality = sb.slider("Model quality (0 = our model, 1 = as good as the bookmaker)", 0.0, 1.0, 0.0, 0.05)
sharp_limit = sb.slider("Sharp stake accepted per price", 0, 5000, 200, 50)
sb.header("How prices move")
RULE_NAMES = {"Fixed (never move)": "fixed", "Move on unexpected money": "information", "Market maker (LMSR)": "lmsr"}
rule = RULE_NAMES[sb.selectbox("Pricing rule", list(RULE_NAMES))]
n_clips = sb.slider("Betting rounds per match", 1, 50, 10, 1)
alpha = sb.slider("Speed of price moves (unexpected-money rule)", 0.0, 4.0, 1.0, 0.1)
casual_weight = sb.slider("Weight on casual money (0 = we know who is sharp)", 0.0, 1.0, 0.1, 0.05)
lmsr_b = sb.slider("Liquidity b (market maker)", 250, 20000, 2000, 250)
sb.header("Assumptions")
casual_info = sb.slider("Share of casual money that is informed", 0.0, 1.0, 0.0, 0.05)
elasticity = sb.slider("Price sensitivity of casual customers", 0.0, 20.0, 8.0, 1.0)
sb.caption("The two assumptions are not estimated from data. Move them to see how much the answer depends on them.")

p_1x2 = D["p_v2"] if model_name.startswith("Improved") else D["p_v1"]
if market.startswith("Match"):
    M = {"names": ["home", "draw", "away"], "rows": np.arange(len(test)), "p_model": p_1x2, "p_true": D["p_true"],
         "y": D["y"], "odds": test[ol.ODDS].to_numpy(dtype=float), "book_margin": D["book_margin"], "extra": {}}
else:
    M = {**D["ou"], "names": ["over 2.5", "under 2.5"], "extra": {"market_margin": round(D["ou"]["book_margin"], 3)}}
p_op = (1 - quality) * M["p_model"] + quality * M["p_true"]
params = dict(margin=margin, method=method, sharp_limit=sharp_limit,
              casual_info=casual_info, elasticity=elasticity, **M["extra"])
dyn = dict(rule=rule, n_clips=n_clips, alpha=alpha, casual_weight=casual_weight, lmsr_b=float(lmsr_b), **params)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["1. Probabilities", "2. Prices", "3. Profit", "4. Goals markets", "Assumptions and limits"])

# ---------------------------------------------------------------- 1. model
with tab1:
    st.subheader("Can past results be turned into match probabilities?")
    st.markdown(
        f"Trained on {D['n_train']} earlier matches and tested on the {len(test)} matches of season "
        f"`{test['season'].iloc[0]}`. Lower scores are better. The step 1 model is one Elo rating per team with "
        f"hand-picked settings. The improved model tunes those settings and tests extra features, "
        f"using the training seasons only.")
    left, right = st.columns(2)
    with left:
        st.dataframe(D["scores"], hide_index=True)
        s = D["scores"].set_index("forecaster")["log_loss"]
        gap = s["base rates"] - s["bookmaker"]
        if gap > 0:
            a, b = st.columns(2)
            a.metric("Gap to bookmaker closed, v1", f"{(s['base rates'] - s['step 1 model (v1)']) / gap:.0%}")
            b.metric("Gap to bookmaker closed, v2", f"{(s['base rates'] - s['improved model (v2)']) / gap:.0%}")
        st.caption(f"The bookmaker's average margin on this season was {D['book_margin']:.2%}.")
        st.markdown(f"**Tuned Elo settings:** k = {D['elo']['k']:g}, home advantage = {D['elo']['home_adv']:g}, "
                    f"carry between seasons = {D['elo']['carry']:g}, promoted teams start at {D['elo']['promoted_start']:g}.")
        st.markdown("**Feature sets, scored on training seasons** (the first row is the one used):")
        st.dataframe(D["feature_table"], hide_index=True)
    with right:
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
        for name, p in [("step 1 (v1)", D["p_v1"]), ("improved (v2)", D["p_v2"]), ("bookmaker", D["p_true"])]:
            c = ol.calibration(p, D["y"])
            ax.plot(c["predicted"], c["observed"], "o-", label=name)
        ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
        ax.set_xlim(0, 0.9); ax.set_ylim(0, 0.9); ax.legend(); ax.grid(alpha=0.3)
        st.pyplot(fig)
    if st.checkbox("Run the walk-forward test (every season predicted from the ones before; takes up to a minute)"):
        wf = load_walk_forward()
        st.caption(f"Average gap closed: v1 {wf['gap closed v1'].mean():.0%}, v2 {wf['gap closed v2'].mean():.0%}. "
                   f"v2 beat v1 in {(wf['improved (v2)'] < wf['elo only (v1)']).sum()} of {len(wf)} seasons.")
        st.dataframe(wf.round(4), hide_index=True)

# ---------------------------------------------------------------- 2. prices
with tab2:
    st.subheader(f"From probabilities to odds: {market}")
    rows = list(M["rows"])
    k = st.selectbox("Match", list(range(len(rows))), index=len(rows) - 1, format_func=lambda j: labels[rows[j]],
                     key="price_match")
    ours = ol.price(p_op[[k]], margin, method)[0]
    table = pd.DataFrame({
        "outcome": M["names"],
        "our probability": p_op[k].round(3),
        "bookmaker's probability": M["p_true"][k].round(3),
        "our odds": ours,
        "bookmaker's odds": M["odds"][k],
        "our edge if the bookmaker is right": (1 - M["p_true"][k] * ours).round(3),
    })
    table["exposed"] = np.where(table["our edge if the bookmaker is right"] < 0, "yes", "")
    st.dataframe(table, hide_index=True)
    st.caption("A price is exposed when someone holding the bookmaker's probability could bet it at a profit.")

    share, bettor_edge = ol.exposure(p_op, M["p_true"], margin, method)
    a, b = st.columns(2)
    a.metric("Share of our prices that are exposed this season", f"{share:.0%}")
    b.metric("Bettor's average edge on those prices", f"{bettor_edge:.0%}")

    grid_m = np.arange(0, 26) / 100
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for m_ in ["proportional", "power"]:
        ax.plot(grid_m, [ol.exposure(p_op, M["p_true"], g, m_)[0] for g in grid_m], label=m_)
    ax.axvline(margin, color="k", lw=1); ax.axvline(M["book_margin"], color="k", ls=":", lw=1)
    ax.set_xlabel("margin we charge (solid line: ours, dotted: the bookmaker's)")
    ax.set_ylabel("share of prices exposed"); ax.legend(); ax.grid(alpha=0.3)
    st.pyplot(fig)

# ---------------------------------------------------------------- 3. profit
with tab3:
    st.subheader(f"One season, played 5,000 times: {market}")
    r = ol.simulate_dynamic(p_op, M["p_true"], **dyn)      # with the fixed rule this is exactly the step 3 book
    st.caption({"fixed": "Prices never move: every bet on a match is struck at the opening price.",
                "information": "Prices move between betting rounds, towards money the operator did not expect.",
                "lmsr": "An automated market maker: the price is a formula of the money already bet. "
                        "The margin slider is charged as a fee and 'where the margin goes' has no effect."}[rule])
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Expected profit", f"{r['expected profit']:,.0f}")
    c2.metric("Turnover", f"{r['turnover']:,.0f}")
    c3.metric("Hold", f"{r['hold %']:.2f}%")
    c4.metric("Worst 5% season", f"{r['worst 5% season']:,.0f}")
    c5.metric("Chance of a losing season", f"{r['P(losing season)']:.1%}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Earned from casual bettors", f"{r['from casuals']:,.0f}")
    c2.metric("Lost to sharp bettors", f"{-r['from sharps']:,.0f}")
    c3.metric("Sharp share of turnover", f"{r['sharp turnover share']:.1%}")
    if rule == "fixed":
        be = ol.break_even_limit(p_op, M["p_true"], **params)
        c4.metric("Break-even sharp limit", "none" if np.isinf(be) else f"{be:,.0f}")
    else:
        c4.metric("Prices still exposed at close", f"{r['exposed at close']:.0%}")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(r["profit"], bins=60)
    ax.axvline(0, color="k", lw=1)
    realised, turnover = ol.settle_dynamic(p_op, M["p_true"], M["y"], **dyn)
    ax.axvline(realised, color="C1", lw=2, label="settled on the actual results")
    ax.set_xlabel("operator profit over the season"); ax.set_ylabel("simulated seasons")
    ax.legend(); ax.grid(alpha=0.3)
    st.pyplot(fig)
    st.caption(f"Money is in arbitrary units: casual turnover is {ol.BASE['casual_turnover']:,} per match at the "
               f"market margin. Read the ratios, not the amounts. On the results that actually occurred this "
               f"book made {realised:,.0f} ({realised / turnover:.2%} of turnover): one draw from the distribution.")

    st.markdown(f"**Price path for the match chosen in tab 2:** {labels[rows[k]]}")
    path = r["book"]["path"][:, k, :]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    for c_, name in enumerate(M["names"]):
        line, = ax.plot(path[:, c_], "o-", ms=3, label=name)
        ax.axhline(1 / M["p_true"][k, c_], color=line.get_color(), ls="--", lw=1)
    ax.set_yscale("log")
    ax.set_xlabel("betting round (dashed: the bookmaker's fair odds)"); ax.set_ylabel("odds offered")
    ax.legend(); ax.grid(alpha=0.3)
    st.pyplot(fig)
    sharp_here = r["book"]["paid"]["sharp"][k]
    st.caption("Sharp money taken on this match: " + ", ".join(f"{name} {v:,.0f}" for name, v in zip(M["names"], sharp_here)))

# ---------------------------------------------------------------- 4. goals markets
with tab4:
    st.subheader("A goals model, and the markets that follow from it")
    st.markdown("Two Poisson regressions predict expected goals for each side. Treating the two as independent gives a "
                "probability for every scoreline, and each score-based market is a sum over that grid.")
    j = st.selectbox("Match", list(range(len(test))), index=len(test) - 1, format_func=lambda q: labels[q],
                     key="goals_match")
    g = D["grid"][j]
    a, b, c = st.columns(3)
    a.metric("Expected home goals", f"{D['lam_h'][j]:.2f}")
    b.metric("Expected away goals", f"{D['lam_a'][j]:.2f}")
    c.metric("Actual score", f"{int(test.loc[j, 'FTHG'])}-{int(test.loc[j, 'FTAG'])}")
    top = np.dstack(np.unravel_index(np.argsort(g.ravel())[::-1][:6], g.shape))[0]
    st.caption("Most likely scores: " + ", ".join(f"{h}-{w} ({g[h, w]:.0%})" for h, w in top))

    left, right = st.columns(2)
    with left:
        st.markdown("**Asian handicap** (the line is added to the home score)")
        hc = []
        for line in [-1.5, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5]:
            fair = ol.handicap_fair_odds(D["grid"][[j]], line)[0]
            offered = np.maximum(1.01, np.floor(fair / (1 + margin) * 100) / 100)
            hc.append({"home line": f"{line:+.2f}", "fair home": round(fair[0], 2), "fair away": round(fair[1], 2),
                       "offered home": offered[0], "offered away": offered[1]})
        st.dataframe(pd.DataFrame(hc), hide_index=True)
        st.caption(f"Offered odds apply the margin slider ({margin:.0%}). Whole lines can push; quarter lines split the stake.")
    with right:
        st.markdown("**Over/under 2.5 against the bookmaker** (whole test season)")
        if D["ou"]:
            st.dataframe(D["ou"]["scores"], hide_index=True)
            st.caption(f"Bookmaker's average over/under margin: {D['ou']['book_margin']:.2%}. "
                       "Choose 'Over/under 2.5 goals' in the sidebar to see this market's prices and profit in tabs 2 and 3.")
        else:
            st.info("The season files have no over/under odds, so there is nothing to compare against.")

# ---------------------------------------------------------------- assumptions
with tab5:
    st.subheader("What this rests on")
    st.markdown("""
- **Truth.** Outcomes are drawn from the bookmaker's probabilities with the margin removed. The bookmaker is
  not the truth either; what matters is that somebody knows more than our model.
- **Casual bettors** split their money roughly like the odds on display and do not shop for value.
- **Sharp bettors** know the reference probabilities, bet every price worth at least 2% to them, and stake the limit.
- **Sharps are always right.** Their probabilities are the truth of the simulation, so a rule that follows
  them looks better here than it would against real, sometimes-wrong sharp customers.
- **Casual customers leave a worse price but never chase a better one.** With moving prices this makes the
  profit levels optimistic; the ordering of the rules is the result to trust.
- **Price sensitivity** of casual customers and the **share of casual money that is informed** are assumed, not measured.
- **Goals are independent Poisson counts.** Real scores have slightly more draws and low scores than that implies.

**Not modelled:** in-play betting, and customers who are neither casual nor sharp.

Data: football-data.co.uk. Code, notebooks and build log are in the repository.
""")
