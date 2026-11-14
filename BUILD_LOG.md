# Build log

How this project was made, including what the AI got wrong.

## Who did what

The code was written with Claude (Anthropic) in a chat session over 13 November 2026, one step at a time.
For each step Claude wrote a notebook and tested it on synthetic data; I ran it on the real season files in
Google Colab and sent the outputs back before the next step was started.

- **Written by the AI:** the functions in `oddslab.py`, the notebooks, the tests and `app.py`, including the
  step 4 improvements (Elo season rules, tuning, extra features, walk-forward testing, the goals model)
  and the step 5 moving-price books.
  I chose which improvements to make from a list the AI proposed.
- **Done by me:**
  - I downloaded the six season files from football-data.co.uk and ran every step on them, first in
    Google Colab and later on my own machine. The AI only ever saw synthetic data until I sent the real
    outputs back, and no step was started before the previous one had been run on real data.
  - I decided the scope after step 3. The AI proposed a list of possible improvements; I chose which ones
    to build (the Elo season rules and tuning, the extra features, walk-forward testing, the goals
    markets, and prices that move).
  - I checked the real outputs against each other between steps. That is how mistake 13 below was found:
    the probability for West Ham v Leeds had changed between two steps with no intended change, which
    led to the Elo bug for teams returning after relegation, and to both notebooks being re-run.
  - I used the app as a user would and asked for changes where it was unclear: for example, total
    turnover was not shown anywhere on the profit tab, and I had it added.
  - I did not write the code. What I can do is explain what each step does, why each check exists, and
    what every number in the README means.

## How the work was kept honest

- **Synthetic data first.** A fake league with known probabilities was generated in the same file format as
  the real data. On it the right answer is known in advance (the fake bookmaker should beat any model), so a
  wrong result means a bug, not a finding.
- **Every step has checks tied to a specific mistake**, for example:
  - changing the result of a match must not change that match's own pre-match rating (no look-ahead);
  - a balanced book with a proportional margin must hold exactly m / (1 + m), whoever wins;
  - the simulated average profit must match the exact expected value from the formula.
- **Assumptions are parameters.** Anything not estimated from data (price sensitivity, how informed casual
  money is, the sharp stake limit) is a named input, and the results show how much the answer moves with it.

## What the AI got wrong, and how it was caught

1. **A simulation with no risk in it.** The first profit simulation split casual money exactly in proportion
   to the displayed odds. That makes the payout identical whichever team wins, so 5,000 simulated seasons all
   returned the same profit. It was caught because the "worst 5% season" equalled the average. Casual money is
   now spread unevenly, and the zero-risk case was kept as a test.
2. **A wrong prediction about its own test data.** The step 2 notes claimed that on synthetic data the
   proportional method would recover the bookmaker's probabilities better than the power method. Running it
   showed no meaningful difference. The claim was removed.
3. **A check that asserted something false.** One test assumed a large enough margin protects any model.
   A model that calls a 5% outcome an 80% favourite stays exploitable at any margin. The test failed, and the
   test was wrong, not the code.
4. **A misleading summary line.** The step 3 notebook reported that profit turns negative "at a sharp limit of
   about 5,000". That was only the first point on the grid where profit was below zero. The loss is linear in
   the limit, so the exact figure can be computed: about 2,900. Fixed in the notebook and the app.
5. **An expectation the data did not support.** The AI expected real bookmaker odds to carry extra margin on
   longshots. On these six seasons, removing the margin either way fits the results equally well. The power
   method is kept as a pricing choice, not as a description of what Bet365 does.

6. **Features added without checking they help.** The first version of the improved model used form and goal
   difference unconditionally. On the synthetic league they made the forecasts worse, because Elo already
   summarises past results. The model now keeps an extra feature only if it beats plain Elo on the training
   seasons, and reports all the alternatives.
7. **A check that could not fail.** One test meant to prove that tuning never sees the test season compared a
   table with a copy of itself. It was replaced with a real one: adding a later season must not change any
   feature of an earlier match.
8. **Handicap odds below 1.00.** Applying a margin to a heavy favourite's handicap price produced odds of 0.98,
   a bet that loses money even when it wins. Offered odds now have a floor of 1.01.

9. **A money machine that was not real.** In the first moving-price simulation, casual customers kept betting
   the same outcomes however bad the price became. Any rule that moved prices against them then looked
   hugely profitable: a hold near 10% on a 6% margin. A result that good from a pricing rule alone was the
   warning sign. Casual money now walks away from a worsened price, at the same assumed sensitivity as in
   step 3, and the remaining optimism is stated in the notebook.
10. **Paying for shares at a stale price.** The market maker converted a customer's spend into shares at the
    price before the trade. Buying moves the price as you go, so that overstated what was bought and inflated
    turnover about thirty-fold. The basket is now scaled until it costs exactly what was spent, and a test
    checks that a purchase costs the change in the cost function.
11. **Prices that blew up.** With an unbounded demand response and unbounded price moves, the two fed each
    other until the numbers overflowed. Each round's move is now capped, and the simulation is run with
    numeric warnings treated as errors.
12. **Another over-strict test.** A test claimed that every price sharps bet must end higher than it started.
    When sharps bet two outcomes of one match, rescaling can lower one of them. The test now covers only
    matches where a single outcome was bet.

13. **The same kind of empty test, and this time it hid a real bug.** When the Elo function gained season
    rules, its defaults were meant to leave the step 1 ratings untouched. The test for that compared the
    function with itself, so it passed. In fact a team returning after relegation (Leeds, Burnley and
    Sunderland in 2025/26) was being reset to 1500 instead of keeping its old rating. I found it from the real
    output: the model's probability for West Ham v Leeds had moved from 37% to 32% between step 2 and step 5
    with no intended change. The "step 1 model" figures in the first runs of steps 4 and 5 were therefore
    slightly off (log loss 1.0307 instead of 1.0304) and both notebooks were re-run. The default now keeps
    old ratings, and the test compares against a separately written copy of the step 1 loop on data that
    contains a returning team.

## What I would do next

In the order I would do them:

1. **Correct the goals model for draws (Dixon-Coles).** The model treats the two teams' goals as
   independent, which understates low-scoring draws. It is the smallest change on this list, and the
   over/under scores show the goals model is where the project is weakest: it does worse than always
   quoting the base rate.
2. **Sharps who are sometimes wrong.** In the simulation the sharps' probabilities are the truth, so a
   rule that follows their money can only look good. Giving them noisy probabilities would test whether
   "Move on unexpected money" still beats the fixed book when the money it follows is not always right.
   This is the result I trust least, so it is the one I most want to stress.
3. **Information Elo cannot see.** Line-ups, injuries and rest days. Tuning Elo did not beat the step 1
   model over four seasons (78% of the gap closed for both), which suggests past results alone have
   been used up and the remaining gap to the bookmaker is information, not method.
4. **Estimate the price sensitivity from data.** The profit-maximising margin is 6%, 11% or 20%
   depending on an assumption I cannot check. It needs an operator's own betting records, so it is last:
   it matters most and is the least possible from public data.
