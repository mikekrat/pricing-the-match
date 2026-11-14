# odds-lab

How a betting operator prices a football match, and what its profit depends on.
Built in five steps on six Premier League seasons (2020/21 to 2025/26, 2,280 matches, football-data.co.uk),
with the 2025/26 season held out for testing.

**Try the app:** https://pricing-the-match.streamlit.app

| Step | Question | Where |
|---|---|---|
| 1 | Can past results be turned into match probabilities? | `notebooks/odds_lab_step1.ipynb` |
| 2 | How do probabilities become odds with a margin? | `notebooks/odds_lab_step2.ipynb` |
| 3 | What does the operator's profit depend on? | `notebooks/odds_lab_step3.ipynb` |
| 4 | Which improvements to the model are worth making? | `notebooks/odds_lab_step4.ipynb` |
| 5 | What changes when prices move as money arrives? | `notebooks/odds_lab_step5.ipynb` |
| App | All five steps with the levers as sliders | `app.py` |

## Findings

**1. A very simple model gets most of the way to the bookmaker.**
One Elo rating per team and a logistic regression close 82% of the log-loss gap between always guessing
the base rates and Bet365's own prices.

| Forecaster | Brier | Log loss |
|---|---|---|
| Base rates | 0.6571 | 1.0861 |
| Elo + logistic | 0.6211 | 1.0304 |
| Bookmaker (Bet365, margin removed) | 0.6115 | 1.0185 |

**2. "Most of the way" is not good enough to set prices.**
At a 6% margin (Bet365 averaged 5.5%), a bettor holding the bookmaker's probabilities could profitably
bet 40 to 45% of the model's prices. Getting that under 10% would need a margin of about 22%.
Loading the margin onto longshots (the power method) cuts longshot exposure from 47% to 35%.

**3. What that costs depends on who is allowed to bet, and how much.**
Simulating the 2025/26 season 5,000 times with casual and sharp bettors:

| Book | Hold | Expected profit | Worst 5% season |
|---|---|---|---|
| Casual bettors only | 6.07% | 222,010 | 156,519 |
| Sharps limited to 200 per price | 5.53% | 206,535 | 140,899 |
| Sharps limited to 1,000 per price | 3.56% | 144,635 | 56,089 |
| Sharps limited to 5,000 per price | -2.93% | -164,864 | -487,046 |

(Units are arbitrary: casual turnover is 10,000 per match.)

- At a tight limit, sharps are 2.2% of turnover and take 7% of the profit. Expected profit reaches zero
  at a limit of roughly 2,900 per price.
- A better model is worth little when limits are tight and a lot when they are loose: at a limit of 1,000,
  moving from the Elo model to one as good as the bookmaker raises expected profit from 144,635 to 216,316.
- So margin, stake limits and model accuracy are substitutes. A weak model survives on a wide margin and low
  limits; a strong one can offer better prices and take bigger bets.

**4. The two largest sensitivities are assumptions, not measurements.**
If half of the casual money followed the true probabilities instead of the displayed odds, expected profit
would fall by a third (206,535 to 140,350). The profit-maximising margin is 6%, 11% or 20% depending on the
assumed price sensitivity of casual customers. An operator would estimate both from its own betting data;
this project cannot.

## Version 2: improvements (step 4)

The findings above are for the step 1 model. Step 4 adds five things, each chosen or tested without
the test season influencing a choice:

- **Elo season rules.** Promoted teams start below the league average, and ratings drift back towards
  average between seasons.
- **Tuned Elo constants.** k, home advantage and the two season settings are picked by a grid search
  scored on training seasons only.
- **Extra features.** Recent form and goal difference are tested, and kept only if they beat plain Elo
  on the training seasons.
- **Walk-forward testing.** Every season from the third onward is predicted from the ones before it,
  so the result is several seasons and not one.
- **A goals model.** Two Poisson regressions give a probability for every scoreline, which prices
  over/under and Asian handicap as well as match result. Over/under is scored against Bet365.

Results on the real seasons:

**The tuned model is not clearly better than the step 1 model.** The grid search picked k = 20, home
advantage 40, carry-over 0.6 and a starting rating of 1400 for promoted teams. On the 2025/26 test season
that model (v2) scores worse than step 1 (v1), and over four walk-forward seasons the two are level.

| Test season | v1: share of gap closed | v2: share of gap closed |
|---|---|---|
| 2022/23 | 65% | 74% |
| 2023/24 | 84% | 78% |
| 2024/25 | 81% | 87% |
| 2025/26 | 82% | 75% |
| Average | 78% | 78% |

("Gap" is the log-loss distance between base rates and Bet365. v2 wins two seasons of four.)

**It is still slightly safer to price with.** On 2025/26, at a 6% margin:

| Model | Log loss | Prices a sharp can beat | Profit, limit 200 | Profit, limit 1,000 |
|---|---|---|---|---|
| Step 1 (v1) | 1.0304 | 40% | 206,535 | 144,635 |
| Improved (v2) | 1.0356 | 37% | 209,188 | 164,501 |

A worse log loss with fewer exposed prices is possible because log loss rewards being right on average,
while exposure depends on how often the model is far enough from the bookmaker to clear the margin.

**The goals model matches v2 on match result and adds nothing on over/under 2.5.** Its match-result log
loss is 1.0358 (v2: 1.0356). On over/under it is worse than always quoting the base rate, and so, by a
hair, is Bet365 (whose over/under margin averaged 5.1%):

| Over/under 2.5 forecaster | Brier | Log loss |
|---|---|---|
| Base rate | 0.4951 | 0.6882 |
| Goals model | 0.5027 | 0.6962 |
| Bookmaker (Bet365, margin removed) | 0.4954 | 0.6887 |

Total goals in one season are close to unpredictable from past results; the value of the goals model is
that it prices new markets consistently, not that it beats anyone.

## Step 5: prices that move

Steps 2 and 3 strike every bet on a match at one price, which is why the loss to sharps grows in a
straight line with the stake limit. Step 5 lets money arrive in rounds and compares three ways of
running the book on the same season:

- **Fixed.** The step 3 book, reproduced exactly as a check.
- **Move on unexpected money.** A sportsbook that shifts its probability towards money it did not
  expect, weighted by how far it trusts the customer. Two settings: how fast, and how well it can tell
  sharp customers from casual ones.
- **LMSR market maker.** The mechanism prediction markets use: the price is a formula of the money
  already bet, and the loss on any one match has a hard ceiling.

Results on the real seasons (step 1 model, 6% margin, 10 betting rounds, expected profit over 5,000
simulated seasons):

| Sharp limit per price | Fixed | Move on unexpected money | LMSR market maker |
|---|---|---|---|
| 0 (casual only) | 222,010 | 212,064 | 174,913 |
| 200 | 206,535 | 251,376 | 182,032 |
| 1,000 | 144,635 | 232,890 | 201,470 |
| 5,000 | -164,864 | 16,514 | 210,554 |

- **Moving prices turn sharp money from a cost into information.** The fixed book loses more with every
  increase in the limit. The moving book still makes money at a limit of 5,000, and the market maker's
  profit rises with the limit.
- **Moving costs something when nobody is sharp.** With casual bettors only, both moving rules earn less
  than the fixed book: they react to noise.
- **Only the sportsbook rule ends with better prices.** At a limit of 1,000, the share of prices a sharp
  could still beat at the close is 35% fixed, 8% moving and 44% for the market maker.
- **Reacting more often matters more than reacting hard.** At a limit of 5,000, profit goes from -164,864
  with one round to 16,514 with 10 and 213,052 with 50. At a limit of 1,000, a speed of 0.5 is best
  (236,099); a speed of 4 does worse than not moving at all (130,277 against 144,635).
- **It depends on telling sharp money from casual money.** If casual money is trusted as much as sharp
  money, the best achievable profit at a limit of 1,000 falls 18%, to 194,367, and a fast-moving book
  loses money (-21,881 at speed 4).

The profit levels with moving prices are optimistic (the notebook says why); the ordering of the rules
and the shape of the curves are the result.

## Limits

- "Truth" is the bookmaker's probabilities with the margin removed. The bookmaker is wrong sometimes too,
  so the sharp bettor's edge here is an upper bound.
- One test season. Differences in the third decimal place of a score are noise.
- There is no in-play betting.
- The goals model treats the two teams' goals as independent, which slightly understates draws.

## Run it

    pip install -r requirements.txt
    python -m pytest -q          # 42 tests
    python -m streamlit run app.py

Put the season files in `data/real` (see the note there). Without them the app runs on synthetic matches
and says so.

## How it was built

See `BUILD_LOG.md`.
