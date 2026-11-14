# odds-lab: User guide to the app

14 November 2026

## What we built

odds-lab shows how a betting operator prices a football match and what its profit depends on. It was built in five steps on 2,280 Premier League matches (seasons 2020/21 to 2025/26, from football-data.co.uk). The first five seasons train the model and 2025/26 is held out for testing.

| Step | Question | What the code does | What we found |
| --- | --- | --- | --- |
| 1. Probabilities | Can past results be turned into probabilities for home win, draw and away win? | Gives every team an Elo rating and passes the difference between the two teams through a logistic regression | The model closes 82% of the gap between a "blind" forecast and Bet365 in the 2025/26 season (log loss 1.0304 against 1.0185) |
| 2. Prices | How do probabilities become odds with a margin? | Adds a margin with two methods and counts how many of our prices are "exposed" | At a 6% margin, 40 to 45% of our prices can be bet at a profit by someone who knows Bet365's probabilities |
| 3. Profit | What does the operator's profit depend on? | Plays the season 5,000 times with casual and sharp bettors | With a low limit on sharps, profit stays close to the margin. With a high limit, the weak model turns into a loss |

The conclusion that ties the first three steps together: margin, stake limits and model accuracy are substitutes. A weak model survives on a wide margin and low limits. A strong model can offer better prices and take bigger bets.

The app (`app.py`) puts all the steps on one page, with the levers as sliders. It is online at [pricing-the-match.streamlit.app](https://pricing-the-match.streamlit.app) and the code is at [github.com/mikekrat/pricing-the-match](https://github.com/mikekrat/pricing-the-match). The same calculations live in `oddslab.py` and are explained in detail in the five notebooks.

## What was added in version 2

Two more steps were added after the first three, both on the same data. The main finding: the model improvements did not change the average, while moving prices removes most of the loss to sharps.

| Step | Question | What it does | What we found |
| --- | --- | --- | --- |
| 4. Improvements | Which improvements to the model are worth making? | Tunes the Elo constants, tests form and goal difference, tests on four seasons instead of one, and adds a goals model for over/under and handicap | The tuned model closes 78% of the gap to Bet365 on average, the same as the simple one. It is steadier (74 to 87% against 65 to 84%), not better. Form and goal difference were rejected |
| 5. Moving prices | What changes when prices move as money arrives? | Bets arrive in rounds and three rules are compared: fixed prices, moving towards unexpected money, and an LMSR market maker | At a sharp limit of 1,000: fixed 144,635, moving 232,890, LMSR 201,470. At a limit of 5,000: -164,864, 16,514 and 210,554 |

The app opens with the "Improved (v2)" model. The numbers in "The levers" and "Experiments to try" were produced with the step 1 model: to see them exactly, choose "Step 1 (v1)" under "Match-result model" and "Fixed (never move)" under "Pricing rule" in the sidebar.

## Getting started

The quickest way in is the link: the app opens in the browser at [pricing-the-match.streamlit.app](https://pricing-the-match.streamlit.app), with nothing to install. If nobody has opened it for a few days, the first load takes a minute or two.

To run it on your own computer, at `http://localhost:8501`, follow three steps:

1. Put the six season files (`E0_2020.csv` to `E0_2025.csv`) in the `data\real` folder of the project. They are already there if you cloned the repo.
2. Open a Command Prompt in the project folder.
3. Run the three commands in order:

```
python -m pip install -r requirements.txt
python -m pytest -q
python -m streamlit run app.py
```

The second command runs the 42 tests and should print `42 passed`. If `python` is not recognised, type `py` in its place.

The screen has two parts:

- **Left column (sidebar).** Four groups: "What is being priced" (market and model), "Levers" (the operator's decisions), "How prices move" (whether and how prices change) and "Assumptions" (assumptions about customers).
- **Main area.** Five tabs: "1. Probabilities", "2. Prices", "3. Profit", "4. Goals markets" and "Assumptions and limits".

Every slider move recalculates the whole page in about a second. If you see a yellow warning about "SYNTHETIC matches" at the top, the app did not find the files in `data\real` and is showing fake data: the numbers are then not findings.

## The tabs

The first three tabs answer the three original questions, in the order of the story you will tell. The fourth shows the goals model and the fifth the assumptions.

### 1. Probabilities

Shows how good the two models are. The sidebar controls change nothing here.

- **Scores table.** Four forecasters: base rates (always the historical frequencies), the step 1 model (v1), the improved model (v2) and Bet365. The Brier and log loss columns measure error, so lower is better.
- **Two "Gap to bookmaker closed" metrics.** How much of the distance from base rates to Bet365 each model covers. In the 2025/26 season it is 82% for v1 and 75% for v2 (log loss 1.0304 and 1.0356). So v2 is worse in this season, and this is where it shows first.
- **Tuned Elo settings.** The constants the search picked on the training seasons: k = 20, home advantage 40, carry 0.6 (each summer the ratings move 40% of the way back to the average), promoted teams at 1400. Step 1 used 20, 60, 1 and 1500.
- **Feature sets table.** Four combinations of features with their log loss on the training seasons. The first row is the one in use: plain Elo (0.9743). Goal difference (0.9753), form (0.9758) and both together (0.9760) came out worse and were rejected.
- **Calibration chart.** One line for v1, one for v2 and one for Bet365. When a model says 40%, does it happen about 40% of the time? Points on the dashed diagonal mean yes.
- **"Run the walk-forward test" checkbox.** Predicts every season from the third onward using only the seasons before it. It takes up to a minute and shows the table below.

| Test season | Gap closed by v1 | Gap closed by v2 |
| --- | --- | --- |
| 2022/23 | 65% | 74% |
| 2023/24 | 84% | 78% |
| 2024/25 | 81% | 87% |
| 2025/26 | 82% | 75% |

The average is 78% for both, and v2 wins 2 of the 4 seasons. The 82% of step 1 was a good season, not the rule.

### 2. Prices

Shows the odds we would offer and how vulnerable they are.

- **Title.** Names the market you chose in the sidebar: "Match result (1X2)" or "Over/under 2.5 goals".
- **Match selector ("Match").** Pick any of the 380 matches of the 2025/26 season. The same match also appears in the price path chart on the Profit tab.
- **Match table.** One row per outcome (home, draw, away, or over and under): our probability, Bet365's probability, our odds, Bet365's odds, and our margin if Bet365 is right. The "exposed" column says "yes" when that margin is negative.
- **Two season metrics.** The share of our prices that are exposed, and the bettor's average edge on them.
- **Exposure chart.** The share of exposed prices for every margin from 0% to 25%, with one line per method. The solid vertical line is our margin and the dotted one is Bet365's (5.5% on 1X2, 5.1% on over/under).

The prices here are always the opening prices. The pricing rule affects only the Profit tab.

### 3. Profit

Shows what we would earn running the book for one season, played 5,000 times. Amounts are in arbitrary units (10,000 of casual turnover per match), so read the percentages, not the amounts.

| Metric | What it means |
| --- | --- |
| Expected profit | The average profit of the season, computed exactly from a formula |
| Turnover | Total money staked over the season, by casual and sharp bettors together. At the starting values with the v1 model it is about 3.73 million |
| Hold | Profit as a share of turnover. With no sharp bettors it is close to the margin |
| Worst 5% season | Profit in the worst 1 season in 20: the measure of risk |
| Chance of a losing season | The share of simulations in which the season ends in a loss |
| Earned from casual bettors | What we make from casual bettors |
| Lost to sharp bettors | What we lose to those who know more than we do |
| Sharp share of turnover | The sharps' share of total turnover |
| Break-even sharp limit | The stake limit per price above which expected profit turns negative. "none" means sharps find nothing to bet. Shown only with the "Fixed" rule. With the other two rules you see "Prices still exposed at close" in its place: the share of final prices a sharp would still bet |

The histogram shows the profit in the 5,000 simulated seasons. The black vertical line is zero. The orange one is the profit on the season's actual results: it is a single observation from the distribution, not a forecast.

Under the title, one line says which pricing rule is active. Under the histogram there are two more items for the match you picked on the Prices tab:

- **"Price path" chart.** The odds of each outcome round by round, on a log axis. The dashed lines are Bet365's fair odds. With the "Fixed" rule the lines are flat. With moving prices the exposed odds should fall towards their dashed line.
- **"Sharp money taken on this match".** How much sharp money we accepted on each outcome of the match.

An example with the v1 model and a sharp limit of 1,000, on the home win in West Ham v Leeds (us 37%, Bet365 51%):

| Rule | Opening odds | Closing odds | Sharp money on the price |
| --- | --- | --- | --- |
| Fixed | 2.55 | 2.55 | 1,000 |
| Move on unexpected money | 2.55 | 2.01 | 800 |
| Market maker (LMSR) | 2.54 | 2.61 | 1,000 |

### 4. Goals markets

Shows the goals model and the markets that come out of it. Two Poisson regressions predict each team's expected goals, and from those comes a probability for every scoreline.

- **Match selector.** Separate from the one on the Prices tab.
- **Three metrics.** Expected home goals, expected away goals, and the actual score. The six most likely scores are listed underneath. For West Ham v Leeds the model gives 1.21 and 1.35 goals with 1-1 the most likely score (13%), while the match ended 3-0.
- **Asian handicap table.** Ten lines from -1.5 to +1.5, with fair odds and offered odds. The line is added to the home team's score. Offered odds follow the margin slider and never go below 1.01. On whole-number lines the bet can be refunded, and on quarter lines the stake is split in two.
- **Over/under 2.5 table against Bet365.** Scores for the whole test season, as on tab 1.

| Over/under 2.5 forecaster | Brier | Log loss |
| --- | --- | --- |
| Base rates | 0.4951 | 0.6882 |
| The goals model | 0.5027 | 0.6962 |
| Bet365 | 0.4954 | 0.6887 |

The goals model has no skill on over/under: it does worse than the base rates. Bet365 did not beat them this season either. So do not present the profits of the over/under market as a finding. On 1X2 the goals model is equivalent to v2 (log loss 1.0358 against 1.0356).

### Assumptions and limits

The fifth tab. Text only: the assumptions of the simulation and what is not modelled. Read it before you present any number.

## The levers: what each slider changes

The sidebar has four groups: what is being priced, the operator's decisions ("Levers"), the pricing rule ("How prices move") and two assumptions that were not measured ("Assumptions"). All numbers below come from runs on the real data, with the other settings at their starting values.

**Starting values:** margin 6%, method power, model quality 0, sharp limit 200, informed casual money 0, price sensitivity 8. At that point, with the "Step 1 (v1)" model and the "Fixed (never move)" rule, expected profit is 206,535 and hold is 5.53%. With the default "Improved (v2)" you will see 209,188 and 5.61%.

### What is being priced: market and model

Two menus that set what the Prices and Profit tabs price.

- **Market.** "Match result (1X2)" or "Over/under 2.5 goals". On over/under the probabilities come from the goals model, which has no skill in that market (see tab 4). Use it to show that the same mechanism works with two outcomes, not for numbers.
- **Match-result model.** "Improved (v2)" is the tuned Elo and "Step 1 (v1)" the original. It applies only to the 1X2 market.

| Model, 2025/26 season | Log loss | Exposed prices at 6% | Profit at limit 200 | Profit at limit 1,000 |
| --- | --- | --- | --- | --- |
| Step 1 (v1) | 1.0304 | 40% | 206,535 | 144,635 |
| Improved (v2) | 1.0356 | 37% | 209,188 | 164,501 |

v2 has a worse log loss in this season but makes more profit, because it leaves fewer prices exposed. Forecast score and profit are not the same thing.

### Margin (overround)

How far above 100% the probabilities hidden in our odds add up. It is the operator's commission.

- **When you raise it:** odds get shorter, we earn more per bet and fewer prices stay exposed, but casual customers leave.
- **Numbers:** at 4% profit falls to 163,074 (hold 3.71%). At 10% it rises to 242,840 (hold 8.96%), with turnover falling from 3.73 to 2.71 million.
- **Caution:** the "best" margin depends almost entirely on the price sensitivity assumption (see below). Getting exposure under 10% needs a margin of about 22%, while Bet365 charges 5.5%.

### Where the margin goes: the method

Where the margin is placed. **Proportional** spreads it evenly across all outcomes. **Power** loads more of it onto longshots.

- **When you switch to proportional:** exposure rises from 40% to 45% and profit falls from 206,535 to 199,364.
- **Why:** on longshots a mistake in the probability costs more. With power, exposure on longshots is 35% instead of 47%, at the cost of slightly more exposure on favourites (33% instead of 29%).

### Model quality

Blends our probabilities with Bet365's. At 0 our own Elo model is pricing and at 1 a model as good as Bet365. It is not something you set in practice: it shows what a better model would be worth.

- **When you raise it:** the prices sharps bet fall from 404 (at 0) to 194 (at 0.5) and to 0 (at 1).
- **With a low limit (200):** profit goes from 206,535 to 216,316. A small difference.
- **With a high limit (1,000):** from 144,635 to 216,316. The model is then worth a third of the profit.

### Sharp stake accepted per price: the limit on sharps

The most we accept from informed bettors on each exposed price. It is the most direct control an operator has.

| Limit | Hold | Expected profit | Worst 5% | Chance of a losing season |
| --- | --- | --- | --- | --- |
| 0 | 6.07% | 222,010 | 156,519 | 0% |
| 200 | 5.53% | 206,535 | 140,899 | 0% |
| 1,000 | 3.56% | 144,635 | 56,089 | 0.4% |
| 2,000 | 1.49% | 67,261 | -71,669 | 19.9% |
| 5,000 | -2.93% | -164,864 | -487,046 | 81.8% |

Each unit of limit costs about 77 a season, in a straight line. Profit reaches zero at a limit of about 2,900.

### How prices move

Five controls for step 5. The bets on each match arrive in rounds and the operator can change prices between rounds. All numbers use the v1 model with the other settings at their starting values.

**Pricing rule.** Three options:

- **Fixed (never move).** Prices never change. This is exactly the step 3 book.
- **Move on unexpected money.** The operator shifts its probability towards money it did not expect, in proportion to how far it trusts the customer.
- **Market maker (LMSR).** The price is a formula of the money already bet, as in prediction markets. The margin is charged as a fee and "Where the margin goes" has no effect.

| Sharp limit | Fixed | Unexpected money | LMSR |
| --- | --- | --- | --- |
| 0 | 222,010 | 212,064 | 174,913 |
| 200 | 206,535 | 251,376 | 182,032 |
| 1,000 | 144,635 | 232,890 | 201,470 |
| 5,000 | -164,864 | 16,514 | 210,554 |

The table shows expected season profit. Fixed prices lose in a straight line with the limit; moving prices do not. At a limit of 1,000 the share of prices still exposed at the close is 35% with Fixed, 8% with Unexpected money and 44% with LMSR. The LMSR had no losing season at any limit, but its prices do not approach Bet365's, because casual money moves it just as much.

**Betting rounds per match (starts at 10).** How many pieces the sharp limit is split into. A price can react only between two bets, so the size of one bet matters as much as the total. With the Unexpected money rule and a limit of 5,000:

| Rounds | Largest bet | Expected profit | Exposed at close |
| --- | --- | --- | --- |
| 1 | 5,000 | -164,864 | 32% |
| 5 | 1,000 | -92,463 | 27% |
| 10 | 500 | 16,514 | 15% |
| 20 | 250 | 160,298 | 2% |
| 50 | 100 | 213,052 | 2% |

At the default limit of 200 the effect is much smaller: beyond 5 rounds the profit moves by about 2%. Raise the limit first if you want to see it clearly.

**Speed of price moves (starts at 1).** Applies only to the Unexpected money rule. At 0 it is the fixed book. At a limit of 1,000 profit is 144,635 at 0, 236,099 at 0.5 (the best), 232,890 at 1, 209,903 at 2 and 130,277 at 4. Too slow and sharps get enough bets in; too fast and prices are pushed away from the right place.

**Weight on casual money (starts at 0.1).** How well the operator can tell sharps from casual bettors. At 0 it knows exactly who is sharp and at 1 it cannot tell anyone apart. At 1 the best profit found is 194,367 instead of 236,099, or 18% less. With weight 1 and speed 2 profit falls to 31,014, and with speed 4 it becomes a loss (-21,881).

**Liquidity b (starts at 2,000).** The only setting of the LMSR. A small b means prices jump on little money; a large b means the book behaves like a fixed one. The loss on one match has a ceiling that grows with b.

| b | Profit at limit 200 | Profit at limit 5,000 | Turnover at limit 200 | Maximum loss per match (average) |
| --- | --- | --- | --- | --- |
| 250 | 167,331 | 219,607 | 2.46 million | 423 |
| 2,000 | 182,032 | 210,554 | 3.16 million | 3,383 |
| 20,000 | 199,608 | 129,852 | 3.45 million | 33,829 |

The profit levels with moving prices are optimistic. The sharps in the simulation are always right, and casual bettors leave a worse price but never chase a better one. Trust the ordering of the rules and the shape of the curves, not the amounts.

### Share of casual money that is informed (assumption)

What part of casual money follows the true probabilities instead of the odds on display.

- **When you raise it:** profit from casual bettors falls, because they too exploit the model's mistakes.
- **Number:** at 0.5 profit falls from 206,535 to 140,350, a third. It is the largest sensitivity in the whole model.

### Price sensitivity of casual customers (assumption)

How fast casual turnover falls when our margin exceeds the market's 5.5%.

- **It matters only when the margin is away from 5.5%.** At 6% almost nothing changes.
- **It sets the optimal margin:** 20% at a sensitivity of 4, 11% at 8, 6% at 16. So the app cannot tell you which margin is right: it can show you the shape of the trade-off.

## Experiments to try

Ten combinations, each showing one of the conclusions: the first six for steps 1 to 3 and the last four for steps 4 and 5. First choose the "Step 1 (v1)" model and the "Fixed (never move)" rule. Start from the starting values every time and change only what is listed. Where a number is given, it comes from the notebooks. Where only a direction is given, it has not been run and you will see the number in the app.

| # | What you change | Where to look | What to expect | What it shows |
| --- | --- | --- | --- | --- |
| 1 | Sharp limit to 0 | Profit | Hold 6.07%, profit 222,010, no losing season | Casual money pays the margin, even with a weak model |
| 2 | Sharp limit to 2,000, then 5,000 | Profit | At 2,000 one season in five is a losing one. At 5,000 expected profit is -164,864 | A few informed bettors with large stakes are enough to turn the book into a loss |
| 3 | Sharp limit to 1,000, then model quality 0 → 0.5 → 1 | Profit | Profit 144,635 → 203,772 → 216,316 | A better model is worth money only when you take large bets |
| 4 | Sharp limit to 1,000, then margin 6% → 12% → 20% | Prices and Profit | Exposure and the loss to sharps fall, but so does turnover | The margin is insurance for a weak model, paid for in customers |
| 5 | Prices tab, match West Ham v Leeds (the default), then model quality to 1 | Match table | At 0 we offer West Ham at about 2.55 while Bet365 offers 1.85, so the price is exposed. At 1 the "yes" disappears | What a model mistake looks like in one specific match |
| 6 | Informed casual money to 0.5 | Profit | Profit 140,350, hold 3.75% | The largest uncertainty is an assumption, not a measurement |

Experiments 1, 2 and 3 together are the answer to "which variables drive profit". Experiment 6 is the one that shows you know where your data stops.

| # | What you change | Where to look | What to expect | What it shows |
| --- | --- | --- | --- | --- |
| 7 | Sharp limit to 1,000, then Pricing rule: Fixed → Move on unexpected money → Market maker (LMSR) | Profit | Profit 144,635 → 232,890 → 201,470. In the "Price path" chart the home win in West Ham v Leeds falls from 2.55 to 2.01 with the second rule | Moving prices removes most of the loss to sharps |
| 8 | Move on unexpected money, sharp limit to 5,000, then rounds 1 → 10 → 50 | Profit | Profit -164,864 → 16,514 → 213,052 | The size of one bet matters as much as the total limit |
| 9 | Move on unexpected money, sharp limit to 1,000, weight on casual money to 1, then speed 0.5 → 2 → 4 | Profit | Profit 194,367 → 31,014 → -21,881 | If you cannot tell which customer is sharp, moving prices fast loses money |
| 10 | Sharp limit to 1,000 with Fixed, then model Step 1 (v1) → Improved (v2) | Probabilities and Profit | Log loss gets worse, from 1.0304 to 1.0356, but profit rises from 144,635 to 164,501 | Forecast score and profit do not always move together |

Experiments 7, 8 and 9 are the answer to "how do odds change as money arrives". Experiment 10 shows why every improvement is also measured in money.

## Assumptions and limits

The app is a tool for seeing relationships between variables, not for forecasting real profits. Seven things limit it.

- **"Truth" is Bet365.** Results in the simulation are drawn from Bet365's probabilities with the margin removed. Bet365 makes mistakes too, so the sharps' edge (17 to 20%) is an upper bound.
- **Three numbers are assumptions.** The sharp limit relative to casual turnover, the price sensitivity, and how informed casual money is. A real operator would estimate them from its own betting data.
- **Sharps are always right.** Their probabilities are the "truth" of the simulation. A rule that follows them therefore looks better here than it would against real sharps, who also make mistakes.
- **Casual bettors leave a worse price but never chase a better one.** With moving prices this makes the profit levels optimistic. With the "Fixed" rule the opposite holds: the loss to sharps is the worst case.
- **Few test seasons.** The app shows one season and the walk-forward test four. Differences in the third decimal place of a score are noise. For example, with v1 a blend of 0.75 gives a log loss of 1.0171, marginally better than Bet365's (1.0185): do not present that as a finding.
- **Goals are treated as independent.** The goals model counts each team's goals as a separate Poisson distribution. Real scores have slightly more draws and low scores. On over/under the model does not beat the base rates.
- **What is not modelled.** In-play betting, accumulators, and customers who are neither casual nor sharp.

Two parameters have no slider and change only in the code, in the `BASE` dictionary of `oddslab.py`: `casual_noise` (how unevenly casual money lands, 0.3) and `min_edge` (the smallest edge at which a sharp bets, 2%).

## What to improve

All seven improvements listed in the first version of this guide are done: six in steps 4 and 5, and the seventh, putting the app online, on 14 November 2026.

| # | Improvement | What it showed on the real data |
| --- | --- | --- |
| 1 | A second feature (form, goal difference) | Rejected. On the training seasons plain Elo had a log loss of 0.9743 and the three combinations 0.9753 to 0.9760 |
| 2 | Start of each season in Elo | Promoted teams at 1400 and a 40% move back to the average each summer |
| 3 | Tuning the Elo constants | 81 combinations. Home advantage fell from 60 to 40 and k stayed at 20. Training log loss 0.9804 → 0.9743 |
| 4 | Testing on more seasons | Four test seasons. An average of 78% for both models, with a range of 65 to 84% for v1 and 74 to 87% for v2 |
| 5 | Moving prices | At a limit of 1,000 profit went from 144,635 to 232,890, and prices exposed at the close from 35% to 8% |
| 6 | Other markets | The goals model prices over/under and handicap, but on over/under it does not beat the base rates (0.6962 against 0.6882) |

Two of the six findings are negative (1 and 6). They are results too, and worth presenting as such.

The next steps, in the order they also appear under "What I would do next" in `BUILD_LOG.md`:

1. **Correct the goals model for draws.** The Dixon-Coles correction raises the probability of low scores and draws, which independent Poisson understates.
2. **Make the sharps fallible.** Today they are always right, which favours the rules that follow them. Noise in their probabilities would show how well the step 5 profits hold up.
3. **Give the model information Elo cannot see.** Line-ups, rest days, injuries. Form and goal difference were rejected because Elo already contains them.
4. **Estimate the price sensitivity from real betting data.** It is the assumption that sets the optimal margin, and only an operator with its own data can measure it.

None of the four is needed for the midterm: the app link, the repo and `BUILD_LOG.md` are enough to show the project. Before you hand it in, read the "Done by me" section of `BUILD_LOG.md` once and correct anything that is not exactly true.

## Glossary

| Term in the app | What it means |
| --- | --- |
| Odds | What you get back for every 1 you stake. Odds of 2.00 correspond to a probability of 1/2.00 = 50% |
| Margin, overround | How far above 100% the probabilities implied by a match's odds add up |
| Hold | The operator's profit as a share of turnover. With margin m and correct probabilities it is m/(1+m) |
| Turnover | The total money staked |
| Casual bettor | Bets by following the odds, without looking for value |
| Sharp bettor | Knows better probabilities than we do and bets only where there is an edge |
| Exposed price | Odds an informed bettor can bet with positive expected profit |
| Stake limit | The most we accept on one price |
| Elo rating | One number per team that rises with wins and falls with defeats, by how expected the result was |
| Log loss, Brier | Error measures for probabilities. Lower is better. Log loss punishes confident but wrong forecasts more severely |
| Calibration | Whether the model's probabilities agree with the frequencies observed |
| Price sensitivity | How fast customers leave when our odds become worse than the market's |
| Break-even | The value of a lever at which expected profit becomes zero |
| Walk-forward test | Every season is predicted only from the ones before it, so the result does not depend on one season |
| Over/under 2.5 | A bet on whether the match will have at least 3 goals or at most 2 |
| Asian handicap | A goal line is added to the home team's score before the bet is settled. There is no draw outcome |
| Pricing rule | How the operator changes the odds as money arrives |
| Betting round | One of the pieces in which a match's money arrives. Prices can change only between two rounds |
| LMSR market maker | The mechanism of prediction markets: the price is a formula of the money already bet and the loss per match has a ceiling |
| Liquidity (b) | How much money it takes to move an LMSR price. A small b means prices that jump easily |
| Exposed at close | A final price, after the last round, that a sharp would still bet at a profit |

Data: football-data.co.uk. The numbers in this guide come from runs of the five notebooks on the real data, last checked on 14 November 2026.
