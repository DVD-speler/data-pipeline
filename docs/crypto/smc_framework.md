# Trading Framework: Market Structure, Liquidity, Ranges, FVGs, Order Blocks, Breaker Blocks & Inducement

## Document Purpose

This document is a structured knowledge base derived from seven educational trading lessons. It is intended as context for a downstream language model to reason about trading setups using a specific discretionary framework.

**Important context for the receiving model:**
- This framework is one specific methodology (smart money / liquidity-based price action), not universal trading theory.
- It is a discretionary framework — its concepts require judgment, not mechanical application.
- It is not financial advice and should be treated as a coherent internal system, not as objective market fact.
- When reasoning about charts or setups using this framework, the model should apply these rules together as an integrated system, not in isolation.

---

## Core Philosophy

Markets are moved by liquidity. Large participants ("smart money") cannot enter or exit positions without sufficient counter-orders, so they systematically push price toward areas where liquidity is concentrated. Retail traders, by contrast, get trapped in predictable positions whose stop-losses *become* that liquidity.

Profitable trading within this framework rests on five interdependent skills:

1. **Reading market structure** — identifying valid triggers (breaks of structure) that provide clean invalidation points.
2. **Understanding liquidity** — distinguishing real (engineered) liquidity pools from superficial highs/lows.
3. **Recognizing ranges** — knowing when price is accumulating/distributing versus expanding, and positioning *before* expansion occurs.
4. **Identifying fair value gaps** — recognizing inefficiencies left behind by fast price moves, which act as magnets for price and refine entry points within a setup.
5. **Trading order blocks and breaker blocks** — using zones where smart money accumulated/distributed positions, including the special case where a failed order block flips polarity and becomes a breaker block.

A cross-cutting concept that ties these together is **inducement** — the front-running of a key zone that engineers the liquidity smart money needs to fill orders there. Inducement is a required input for trading order blocks, breaker blocks, and range deviations, and is treated as its own topic in Part 6 because of how widely it applies.

A setup is only high-probability when these align: a valid market structure break, mitigation of valid (engineered) liquidity, alignment with the higher-timeframe range/bias context, a valid inducement when zone-based, and ideally an entry refined by a fair value gap, order block, or breaker block.

---

## How to Use This Document

This document is the conceptual foundation for a downstream trading model that operates on **OHLC (Open, High, Low, Close) time-series data**. The model does not receive chart images. All visual language in this document (e.g., "green candle," "wick," "candle body") refers to measurable properties of OHLC data and must be interpreted accordingly.

The framework is hierarchical and context-dependent. Concepts cannot be applied in isolation — they form a dependency chain in which higher-level context (bias, range structure) gates whether lower-level signals (an MSB, an OB) are tradeable at all. The Decision Hierarchy section below makes the resolution rules explicit.

When the model encounters a candidate setup, it should reason in the order:
1. Bias (Part 2, Part 3) → directional expectation.
2. Range context (Part 3) → is price in accumulation/distribution or expansion?
3. Liquidity context (Part 2) → is there valid engineered liquidity nearby?
4. Trigger (Part 1) → is there a valid MSB?
5. Inducement validation (Part 6) → if the setup involves a zone, is the inducement valid?
6. Entry refinement (Parts 4, 5) → can FVG / OB / Breaker Block produce a better entry?
7. Invalidation → is there a clean stop-loss level?

If any step fails, the setup is not tradeable regardless of how strong the other steps look.

---

## OHLC Translation Guide

This section maps the visual language used throughout the document to operations on OHLC data. The downstream model should treat these as the authoritative definitions.

### Candle Basics

For any candle with open `O`, high `H`, low `L`, close `C`:

- **Green / bullish candle** = `C > O` (close greater than open).
- **Red / bearish candle** = `C < O` (close less than open).
- **Doji** = `C ≈ O` (close approximately equal to open; treat as neither for color-dependent rules).
- **Candle body** = the price range `[min(O,C), max(O,C)]`.
- **Upper wick** = the price range `[max(O,C), H]`.
- **Lower wick** = the price range `[L, min(O,C)]`.
- **"Top of the candle"** = `H` (the high of the candle), unless context specifies "top of the body," which is `max(O,C)`.
- **"Bottom of the candle"** = `L` (the low of the candle), unless context specifies "bottom of the body," which is `min(O,C)`.

### Strength / Momentum Descriptors

- **"Strong" / "impulsive" candle** = large `|C - O|` relative to recent average body size, often with small wicks relative to body. Operationalize as: body size > N× the trailing average body size (N typically 1.5–2.0).
- **"Displacement"** = a sequence of impulsive candles in one direction, often producing an FVG. Operationalize as: consecutive same-direction candles with above-average body sizes.
- **"Slow grind"** = many candles with small bodies relative to recent average, typically in one direction. The opposite of displacement.

### Highs, Lows, and Pivots

- **A pivot high (swing high)** = a local maximum: a candle whose `H` is greater than the `H` of N candles to its left and N candles to its right (typical N = 2–5 depending on timeframe).
- **A pivot low (swing low)** = a local minimum, symmetric definition.
- **"Significant high"** (per Part 1) = a pivot high that is *subsequently* followed by a pivot low that breaks below the *prior* pivot low. The significance is determined retrospectively by the next structural development.
- **"Significant low"** = symmetric: a pivot low followed by a pivot high that breaks above the prior pivot high.
- **"Insignificant high"** = a pivot high that is followed only by a higher pivot low (no lower low formed).
- **"Insignificant low"** = a pivot low followed only by a lower pivot high.

### Break of Structure Detection

For a candle at time `t` to confirm a bullish MSB above a significant high `H_sig`:
- `O_t < H_sig` (candle opened below the level) AND
- `C_t > H_sig` (candle closed above the level)

A wick-through alone (`H_t > H_sig` but `C_t ≤ H_sig`) is **not** an MSB — it is an SFP (Swing Failure Pattern). Symmetric rules apply for bearish MSBs below a significant low.

### Fair Value Gap Detection

For three consecutive candles at `t-2`, `t-1`, `t`:

**Bullish FVG** (high probability) requires:
- All three candles green: `C_{t-2} > O_{t-2}` AND `C_{t-1} > O_{t-1}` AND `C_t > O_t`
- Gap exists: `L_t > H_{t-2}` (low of third candle is above high of first candle)
- The gap itself is the price range `[H_{t-2}, L_t]`.

**Bearish FVG** (high probability) requires:
- All three candles red.
- Gap exists: `H_t < L_{t-2}` (high of third candle is below low of first candle).
- The gap range is `[H_t, L_{t-2}]`.

If a gap exists but the third candle's color doesn't match, it is a **low probability FVG** and should not be used for trade decisions.

### Order Block Detection

A **bullish order block candidate** is the last red candle (`C < O`) before a sequence of strong bullish candles. The OB zone is typically `[L_candidate, H_candidate]` or `[L_candidate, max(O_candidate, C_candidate)]` depending on convention; default to the full candle range `[L, H]`.

A **bearish order block candidate** is the last green candle before a sequence of strong bearish candles, with symmetric zone definition.

A candidate becomes a **high-probability OB** only if all five requirements in Part 5 are satisfied. The model must verify each requirement programmatically:
1. Liquidity sweep on formation → a prior pivot low (for bullish OB) is taken out by the candidate or the candles forming the impulse.
2. FVG on the leg away → an FVG exists within the N candles following the OB (typically N = 1–5).
3. MSB on the leg away → an MSB occurred within the same window.
4. HTF bias alignment → verified against the higher-timeframe context.
5. Valid inducement → see inducement detection below.

### Inducement Detection

For a bullish setup with a demand zone (OB, breaker block, or support) at `[Z_low, Z_high]`:
- An **inducement candidate** is a pivot low `P_ind` such that `Z_high < P_ind < current_price` (price front-ran the zone; pivot is above the zone but below current price).
- The inducement is **valid** if, in the candles between `P_ind` and the current candle, at least one **fake break of structure** occurred — i.e., a candle closed above an insignificant high, trapping bullish breakout traders whose stops would rest below `P_ind`.
- Without a fake BoS in the interim, the inducement is **invalid**.

Symmetric logic for bearish setups (front-run above a supply zone, fake breaks below insignificant lows trap bearish breakdown traders).

### Range Detection

A range requires three pivots minimum:
- Two pivot highs at approximately the same price level (within tolerance, e.g., 0.5–1% of price) + one pivot low between or after them, OR
- Two pivot lows at approximately the same price level + one pivot high between or after them.

Additional requirements:
- The solo pivot (single-touch side) must come from an impulsive move (per "Strong / impulsive candle" definition above).
- The mid-range level `(range_high + range_low) / 2` must show price reaction (rejection wicks, candle reversals) on at least two touches. If price slices through the mid-range without reaction on multiple touches, the range is invalid.

### Liquidity Pool Detection

A **liquidity pool** is a price level where stops are likely concentrated. Candidate levels include:
- Prior pivot highs / lows (especially equal highs / equal lows within tolerance).
- Levels above / below fake breaks of structure.
- Levels above / below inducement pivots.

A pool is **valid** only if recent price action engineered it — i.e., the model can identify a behavioral reason (trapped breakout traders, trapped SFP traders, etc.) for stops to rest there. A pool is **invalid** if it is merely a prior high/low without engineered context.

### Timeframe Notation

- **HTF** = Higher Timeframe (typically 4H, daily, weekly).
- **LTF** = Lower Timeframe (typically 15M, 1H).
- The framework requires MSB confirmation on the highest timeframe where the reversal structure is visible. The model should evaluate the structure at multiple timeframes and select the highest one on which the relevant pivots are present.
- Trade execution timeframe: M15 or higher.

---

## Decision Hierarchy

When concepts appear to conflict, the following resolution rules apply, in order of precedence:

### Rule 1: Bias Overrules Default Trade Selection
- Default trade in a range is the deviation (Part 3), because deviations occur more often than breakouts.
- **But if HTF bias is opposite to the deviation direction, do not trade the deviation.** Wait for a breakout/breakdown in the bias direction instead.
- Bias always wins over statistical default.

### Rule 2: HTF Draw on Liquidity Overrules LTF Structure
- LTF can show a clean break of structure against the trend.
- **But if HTF draw on liquidity is still untaken in the original direction, treat the LTF break as a probable trap.**
- Smart money cannot reverse until they exit existing positions into the HTF draw. The LTF break is likely an inducement for the continuation, not the start of a reversal.

### Rule 3: Inducement Validity Is Non-Negotiable
- A clean zone (OB, breaker block, support/resistance) without valid inducement is **not tradeable**, regardless of how good the zone looks.
- An invalid inducement is equivalent to no inducement.
- This rule supersedes any other zone-quality consideration.

### Rule 4: Engineered Liquidity Overrules Surface-Level Levels
- A prior high/low that looks like a liquidity pool but has no engineered context is **not a valid pool**.
- Trades targeting invalid pools, or relying on invalid pool sweeps for setup confirmation, are low probability.
- "It's a previous high" is not sufficient justification.

### Rule 5: Candle Close Overrules Wick
- An MSB requires candle open AND close beyond the level.
- A wick beyond a significant level without close is an SFP — it serves a different purpose (taking out stops) but does not confirm structure.
- A trade thesis built on a wick-only break is not valid.

### Rule 6: All Five OB Requirements Are Required
- An order block missing any of the five requirements (Part 5) is not high probability.
- Four out of five is not "almost tradeable." It's not tradeable.
- The same all-or-nothing rule applies to the three breaker block requirements.

### Rule 7: When in Doubt, Do Not Trade
- The framework rewards patience. The cost of skipping a marginal setup is zero; the cost of taking a low-probability trade is real capital loss and psychological erosion.
- If multiple checks pass but one is ambiguous, default to no-trade.

### Rule 8: Clean Invalidation Is Required
- Every trade must have a clean, definable stop-loss level — the price at which the thesis is invalidated.
- If no clean invalidation exists (e.g., no swept low to place a stop beneath), the setup is not tradeable, even if everything else looks good.

---

## Part 1: Market Structure

### The Role of Market Structure Breaks

A market structure break (MSB / BoS) is the **only valid trigger** to enter a trade in this framework. Without an MSB there is no clean invalidation point for a stop-loss, and therefore no tradeable setup.

### Significant vs. Insignificant Highs/Lows

This distinction is foundational:

- **Significant high** — a high that leads to a *lower low* afterward.
- **Significant low** — a low that leads to a *higher high* afterward.
- **Insignificant high** — only leads to a higher low (no real shift).
- **Insignificant low** — only leads to a lower high.

Only breaks of *significant* highs/lows count as valid market structure breaks. Breaks of insignificant highs/lows are fake breaks designed to trap retail.

### The Four Categories of Market Structure Breaks

#### 1. Fake Break of Structure
Breaking through an *insignificant* high/low. Engineered to trap retail traders into low-probability breakout positions. **Never traded.** Recognize it as a sign that liquidity is being built for the opposite move.

#### 2. Valid Break of Structure
A candle **opens and closes** above a significant high (or below a significant low), but neither pivot in the reversal structure mitigated a meaningful liquidity pool. Lowest probability of the three valid types.

- Generally **not traded** as a standalone trigger.
- Tradeable only when it aligns with a strong, clear bias *and* the pivot aligns with time of price (significant candle low/high on session, 4H, or daily timeframe).

#### 3. Significant Break of Structure
A valid break where the **second pivot** of the reversal structure mitigates a valid (engineered) liquidity pool. Higher probability than a valid break because smart money had the opportunity to fill orders before the reversal.

- Preferred entry trigger.
- Strength of the sweep matters: a deep mitigation is stronger than a small wick into the pool.

#### 4. High Probability Break of Structure
Highest quality variant. Builds on the significant break with additional confluences (referenced in the source material but not fully expanded in this document).

### Candle Confirmation Rules

A market structure break is only confirmed when a candle **opens AND closes** above the significant high (or below the significant low). A wick through is not confirmation — that is a Swing Failure Pattern (SFP), which serves a different purpose (taking out stops above/below the level).

### Timeframe Rules for MSB Confirmation

Use the **highest timeframe on which the reversal structure is visible**. More data per candle = higher reliability of the trigger.

- Structure visible on 4H + 2H + 1H → wait for **4H candle** open and close.
- Structure visible only on 2H + 1H → wait for **2H candle** open and close.
- Structure visible only on 1H → wait for **1H candle** open and close.

### Execution Timeframes

Trades are executed on **M15 or higher**. Lower timeframes are used only to refine entries within a higher-timeframe setup.

### The SFP Variant (Fake MSB via Wick)

A common manipulation pattern:
1. Price wicks above a significant high without closing above it.
2. Stops above the high are taken.
3. Price reverses, traps the breakout longs, then often traps shorts on the way back up.
4. Result: bidirectional liquidity harvest.

This is structurally a *fake break* even though the high was significant, because the candle did not close above it.

---

## Part 2: Liquidity

### Why Liquidity Matters

Large participants cannot fill or close positions without sufficient counter-orders. If a smart money entity wants to buy 100,000 contracts but only 70,000 contracts of sell-side liquidity exist at the current level, they cannot fill — they get slippage, partial fills, or worse pricing.

This forces them to **push price to areas where liquidity is concentrated**, either:
- To **fill new orders** (entering positions).
- To **close existing orders** (exiting positions into opposing liquidity).

This is why bias alignment with higher-timeframe liquidity is essential: smart money cannot reverse price until they have had the opportunity to exit existing positions into a sufficiently deep pool.

### The Retail Mistake

Most retail traders identify liquidity by simply marking previous highs, lows, and equal highs/lows. If this worked, anyone could be profitable. It doesn't, because **not every high/low is a valid liquidity pool**.

### Valid vs. Invalid Liquidity Pools

A liquidity pool is **valid** only when prior price action has *engineered* it — meaning the price behavior trapped traders into predictable positions whose stops now rest at that level.

**Invalid liquidity pool:** A previous high/low without preceding price action that lured traders into positions. The level exists on the chart but no meaningful stops/orders rest there.

**Valid liquidity pool:** A high/low created by price action that demonstrably trapped participants (e.g., a fake break, an SFP, a strong rejection from a key level).

### Engineered Liquidity: The Mechanism

Engineered liquidity is built when price action systematically lures traders into predictable positions. Common patterns:

- **Fake break of structure** → retail enters breakout longs with stops below the broken level → engineered sell-side liquidity beneath that level.
- **Sharp rejection sweeping internal liquidity** → traps both directions, sets up a range.
- **Aggressive impulse moves** that don't pull back → late longs/shorts enter with poorly placed stops.

The key question to ask of any high or low: *"What did price action do that would have caused traders to take positions here?"* If the answer is "nothing specific," it's not a valid liquidity pool.

### Valid vs. Invalid Liquidity Sweeps

- **Valid sweep**: Price mitigates a valid (engineered) liquidity pool. The subsequent break of structure is high-probability.
- **Invalid sweep**: Price takes out a previous high/low where no real liquidity was engineered. The subsequent break of structure is low-probability.

This is what separates significant MSBs (Part 1, category 3) from merely valid MSBs (Part 1, category 2).

### Sweep Quality

How price mitigates the pool matters:
- **Small wick** into the pool → may not be deep enough for smart money to fill significant orders.
- **Deep push** into the pool → genuine mitigation, higher probability for reversal.

### Draw on Liquidity (Higher-Timeframe Anchor)

The **draw on liquidity** is the next significant higher-timeframe liquidity pool that price is gravitating toward. It serves as the directional anchor for bias.

Critical principle: **Smart money will not reverse price until the higher-timeframe draw is taken**, because they need that pool to exit existing positions. Lower-timeframe breaks of structure against the HTF draw should be treated with skepticism.

### Application: Reversal Setup (High Probability)

1. Downtrend in progress.
2. Price action engineers liquidity above a lower high (e.g., via a fake break to the downside that traps shorts, or a sharp rejection that traps longs).
3. Price sweeps that engineered liquidity (**valid sweep**).
4. Candle opens and closes below a significant low on the appropriate timeframe (**MSB confirmed**).
5. Higher low forms.
6. Entry on the higher low; stop below the swept low; target higher levels.

### Application: Continuation Setup (High Probability)

1. Uptrend in progress; HTF draw on liquidity still above.
2. Pullback occurs with fake breaks to the upside that trap longs (stops below the pullback pivot).
3. Price sweeps those stops (**valid sweep** below the engineered pool).
4. MSB to the upside confirms continuation.
5. Entry; stop below the sweep low; target the HTF draw.

### Application: When NOT to Take a Counter-Trend Trade

Even when lower-timeframe structure breaks against the trend, *do not* short an uptrend (or long a downtrend) if:

1. The HTF draw on liquidity is still untaken in the trend direction.
2. The sweep that preceded the LTF break was an *invalid* sweep (no engineered liquidity).
3. Smart money has not yet had opportunity to exit existing positions.

In these cases, the LTF break is likely a trap; price tends to reclaim the broken level and continue with the original trend.

---

## Part 3: Ranges

### Why Ranges Matter

Markets range approximately **90–95% of the time** and only trend ~5–10% of the time. Identifying ranges and positioning *within* them before the next expansion is the core skill.

### How Ranges Form

Ranges typically form after a strong impulse move (up or down) that includes a sharp rejection sweeping internal liquidity. This rejection:

1. Wrecks traders who shorted the pullback (stops above the rejection high get taken).
2. Wrecks traders who FOMO into longs on the push (stops below the rejection low get taken).

Both directions are now stopped out. Retail traders, eager to recover losses, enter positions hastily without proper confirmation. Smart money exploits this by chopping the price sideways — accumulating or redistributing while retail repeatedly gets stopped out.

### Requirements for a Valid High-Probability Range

A range is **valid** only if all three conditions are met:

#### 1. Three Pivots Minimum
Either:
- Two pivots at the top + one pivot at the bottom, or
- Two pivots at the bottom + one pivot at the top.

The paired pivots must touch the same price (or very nearly the same price).

#### 2. The Solo Pivot Must Be Strong
The lone pivot (where there's only one touch) must come from a **harsh rejection** — typically a sharp move sweeping sell-side or buy-side liquidity. Weak solo pivots invalidate the range.

#### 3. Mid-Range Respect
After drawing the range based on the pivots, the **mid-range level (50%)** must be respected as support/resistance. If price chops through the mid-range without reaction, the range is not valid regardless of how clean the pivots look.

Mid-range respect may be visible only on a lower timeframe than the one used to draw the range — drop down timeframes to check.

### Tool Usage

The Fibonacci retracement tool is used to draw ranges (with the mid-range level at 0.5). Rectangles can also be used for a wider range definition.

### Trading the Range: Two Approaches

Once a valid range is established, there are two strategic choices:

#### Approach 1: Trade Deviations (Default)
Because price stays in the range ~95% of the time, the default is to fade extremes:
- Price pushes below the range low → trade a deviation back toward the range high.
- Price pushes above the range high → trade a deviation back toward the range low.

#### Approach 2: Trade Breakouts/Breakdowns
When directional bias is strong, wait for a confirmed breakout/breakdown model in the direction of bias rather than fading.

### The Critical Rule: Bias Overrides Default

**Probability of deviation > probability of breakout — but bias overrides probability.**

If bias is bearish and price pushes below the range low, do **not** trade the deviation back up just because deviations happen more often. Wait for a confirmed breakdown model instead.

Bias is set by higher-timeframe context: HTF draw on liquidity, HTF structure, HTF wicks needing to be filled, etc. (Part 2 concepts.)

### The Art of Range Trading

The goal is to **position before expansion**, not chase it. The expansion phase is when most retail enters — by then the meat of the move has already happened.

Ideal sequence:
1. Identify a valid range early.
2. Establish bias from HTF context.
3. Position into a deviation in the direction of bias (e.g., long the deviation below the range low when bias is bullish).
4. On confirmed breakout in the direction of bias, **add a second position with the same invalidation**, targeting the expansion.

This pyramids into the move *before* the crowd, with controlled risk.

### Candle Analogy

Apply the same logic to individual candles:
- If anticipating a bullish candle: enter near the candle open (the wick-down area), not after expansion has occurred.
- If anticipating a bearish candle: enter near the high before the body forms.

By the time the candle body is forming, you're late.

---

## Part 4: Fair Value Gaps

### What a Fair Value Gap Is

A **Fair Value Gap (FVG)** is an imbalance left behind when price moves too fast in one direction. Because the move was aggressive, the level was skipped — there was no opportunity for the full normal supply/demand interaction at that price. Price treats these gaps as magnets and tends to return to fill them (partially or fully) before continuing the move or reversing.

The name is literal: it is a gap in the *fair value* of price. The price level inside the gap was never properly traded, so it does not yet represent a settled fair price.

### The Lemon Analogy

If lemons are priced at 5 cents and suddenly shoot to 20 cents due to a supply shock, the demand that existed at 5 cents will not exist at 20 cents — the move was too fast for buyers to participate at fair value. For demand to re-enter, price typically drops back to fill at least part of that gap (e.g., 15 cents), where prior buyers feel they're getting reasonable value again. Only then does the move continue.

The same principle applies to any asset: BTC, gold, stocks. Fast moves leave inefficiencies; price returns to address them.

### Identifying a Fair Value Gap

A fair value gap is defined by **three consecutive candles**:

#### Bullish Fair Value Gap
1. First candle: **green**.
2. Second candle: **green**, with strong upward movement.
3. Third candle: **green** (required for high probability).
4. The **top of the first candle** and the **bottom of the third candle** must NOT touch in price — there must be a clean gap between them.

The gap itself is the price range between the top of candle 1 and the bottom of candle 3.

#### Bearish Fair Value Gap
1. First candle: **red**.
2. Second candle: **red**, with strong downward movement.
3. Third candle: **red** (required for high probability).
4. The **bottom of the first candle** and the **top of the third candle** must NOT touch in price.

The gap is the price range between the bottom of candle 1 and the top of candle 3.

### High Probability vs. Low Probability FVGs

The color of the third candle determines probability:

- **High probability FVG**: All three candles match the direction (3 greens for bullish, 3 reds for bearish).
- **Low probability FVG**: A gap technically exists, but the third candle is the opposite color (e.g., green-red-green still leaves a gap, but the structure is weaker).

**Only high probability FVGs are traded in this framework.** Low probability FVGs are noted but not acted upon.

### Why FVGs Matter: Price Is Attracted to Them

Price consistently gravitates back to high probability fair value gaps before continuing the broader move. In a strong uptrend, you will see price repeatedly:
1. Create a bullish FVG on an impulsive leg up.
2. Push higher briefly.
3. Pull back to tap (partially or fully) the FVG.
4. Resume the upward move.

This happens because demand cannot fully participate at the post-impulse price. The pullback into the FVG gives that demand a chance to enter, which then fuels the next leg.

### Practical Use: Refining Entries

FVGs are most powerful when used to **refine entry points within an already-valid setup**. The setup itself comes from market structure + liquidity + bias (Parts 1, 2, 3). The FVG tells you *where within that setup* to enter for the best risk/reward.

**Example workflow:**
1. Identify a bullish setup: liquidity sweep at a low + bullish MSB + bullish bias.
2. Market-executing at the break gives a stop below the sweep low and a take-profit at the prior high — but the RR is only 1.25.
3. The leg that caused the MSB also left a bullish FVG behind.
4. Set a limit entry at the FVG instead of market-executing.
5. Same stop (below the sweep low), same target — but RR is now 2.39.

The trade gets filled because price is attracted to the FVG. The setup's invalidation hasn't changed, but the entry is now at a much better price.

### FVGs and Setup Quality

An FVG on its own is not a setup. It is a **refinement tool**:
- FVG without a valid MSB + bias + liquidity context → not tradeable.
- FVG within a valid setup → improved entry, better RR, often higher fill probability.

The order of operations is always: bias first → liquidity check → MSB trigger → use FVG to refine entry.

### Relationship to Other Concepts

Fair value gaps interact closely with **order blocks** and **breaker blocks** (covered in Part 5). FVGs often sit *inside* or *adjacent to* these zones, and their confluence with such zones increases probability further.

---

## Part 5: Order Blocks and Breaker Blocks

### What an Order Block Is

An **order block** is the last opposing candle before a strong impulsive move:

- **Bullish order block**: The last red candle before a strong leg up.
- **Bearish order block**: The last green candle before a strong leg down.

The candle marks the zone where smart money accumulated or distributed positions before the expansion.

### Why Order Blocks Matter (Institutional Logic)

Large participants cannot fill enormous positions instantly — it takes time. If smart money begins accumulating in a zone but price expands away before they're fully filled, they have an incomplete position.

The lemon analogy: imagine wanting to buy 1 million lemons at 10 cents each. You manage to fill 500,000 at that price, but then the market moves to 12 cents. You still want the other 500,000 at 10 cents — so you wait for price to return to that level to complete your order.

Smart money does the same thing. If they only filled part of their position before price expanded, they have strong incentive to push price back to that zone to fill the rest before allowing the real expansion to occur. **That return-to-the-zone is what makes order blocks tradeable.**

### The Lemon-vs-Liquidity Distinction (vs. FVG)

The lemon analogy appears in both Part 4 (FVGs) and here — but it explains two different things:

- **For FVGs**: Price moved too fast, so demand at the new price isn't the same. Price returns to fill the inefficiency.
- **For order blocks**: Smart money didn't have *time* to fill their full position. Price returns so they can complete it.

Both concepts describe price returning to a level, but the underlying mechanism is different: FVG = inefficient pricing, OB = incomplete order fills.

### Valid vs. High-Probability Order Blocks

A *valid* order block is simply the last opposing candle before an impulsive move. Most traders stop there — and this is why most traders draw order blocks wrong.

A **high-probability order block must tick all five requirements**. Without all five, it's a valid OB but not a high-probability one, and it should not be traded.

### The Five Requirements for a High-Probability Order Block

#### 1. The OB Must Sweep an Important Liquidity Pool
While the order block is being formed, price must mitigate a significant liquidity pool (typically a higher-timeframe pool). This is the precondition: without liquidity present, smart money cannot fill orders. No swept liquidity = no real institutional accumulation/distribution = not a high-probability OB.

#### 2. The OB Must Produce a Strong FVG on the Impulsive Leg
After the OB is formed and price expands away from it, that expansion must leave behind a **high-probability fair value gap** (per Part 4: three matching-color candles, top of candle 1 and bottom of candle 3 not touching).

The FVG is the signature of genuine buyer/seller participation — a bullish breakaway FVG indicates real buying strength stepped in at the OB. No FVG = weak impulse = weak OB.

#### 3. The Impulse Must Cause a Break of Market Structure
The expansion away from the OB must produce a valid MSB in the direction of the expansion (per Part 1: candle open + close beyond a significant high/low).

The MSB confirms strength — it shows the side that defended the OB has genuinely taken control of structure. An impulse without an MSB is just a wick, not a shift.

#### 4. The OB Must Align with Higher-Timeframe Bias
This is non-negotiable. A bullish OB against a bearish HTF bias is not tradeable, no matter how clean the other four requirements look. **Always form bias first** (per Parts 2 and 3), then look for OBs that align with it.

Bias context might come from:
- Range position (e.g., deviation below range low + bullish bias → look for bullish OBs).
- HTF draw on liquidity (price still has untaken liquidity in the bias direction).
- HTF structure trend.

#### 5. The OB Must Have Valid Inducement
For a bullish OB: valid inducement above the OB (a low engineering liquidity above).
For a bearish OB: valid inducement below the OB (a high engineering liquidity below).

The inducement is what makes the retest tradeable. Without inducement, there's no engineered liquidity to sweep before the OB tap — and without that sweep, the retest is low-probability.

**Note on timeframe**: inducement may not be visible on the same timeframe as the OB. If the OB is on the 4H, the inducement might be visible only on the 1H. Drop down timeframes to confirm.

### Execution Model

When all five requirements are met:
1. Place a **limit order** at the order block.
2. Invalidation (stop-loss) goes beyond the OB.
3. Target: the next logical liquidity pool in the bias direction (e.g., range high for a bullish OB inside a range).

The example from the source: a bullish OB on Bitcoin that ticked all five requirements produced an RR of 6.92 on a single trade — entry at the OB retest, target at the range high liquidity pool.

### Common Mistakes Traders Make with Order Blocks

- Marking *any* last-opposing candle as a high-probability OB without checking the five requirements.
- Trading an OB that doesn't align with HTF bias.
- Ignoring inducement — entering on the OB retest without an engineered liquidity sweep first.
- Confusing valid OBs with high-probability OBs; the distinction is what separates profitable use of the concept from random trade selection.

### Breaker Blocks: The Failed Order Block Flip

A **breaker block** is what an order block becomes when it **fails** — when price slices through it instead of respecting it.

The mechanism:
1. An order block forms (e.g., a bullish OB at a low).
2. Traders accumulate positions at the OB retest, expecting it to hold.
3. Price breaks through the OB instead of bouncing.
4. All those traders are now underwater on their positions.
5. If price later returns to that same zone, those traders take the opportunity to **exit at break-even (or a small loss)**.
6. Their exit orders add to the prevailing pressure, causing the previous OB to now act as the *opposite* (resistance instead of support, or vice versa).

The previous demand zone has effectively flipped polarity — it's now a supply zone. This flipped zone is the **breaker block**.

### Why Breaker Blocks Work

The probability behind a breaker block comes from a specific behavioral mechanism: trapped participants seeking escape at break-even. When a large group of traders is underwater and price returns to their entry, the path of least resistance for them is to exit. Their collective exit orders then reinforce the move in the opposite direction.

This is structurally similar to how engineered liquidity works in Part 2: failed setups create predictable behavior that smart money can exploit.

### Bullish Breaker Block (Failed Bearish OB)

1. A strong bearish order block forms (last green candle before a leg down).
2. Price returns to it but **breaks through** instead of being rejected — multiple candles open and close above it.
3. The shorts who entered at that OB are now underwater.
4. When price retests the previous OB from above, it now acts as **support**.
5. Each touch of the breaker block holds and pushes price higher.

### Bearish Breaker Block (Failed Bullish OB)

1. A strong bullish order block forms (last red candle before a leg up).
2. Price returns to it but slices through instead of bouncing — multiple candles open and close below it.
3. The longs who entered at that OB are now underwater.
4. When price retests the previous OB from below, it now acts as **resistance**.
5. The breaker block produces rejections, pushing price lower.

### Requirements for a Tradeable Breaker Block

A breaker block is only tradeable when **all** of the following are met:

#### 1. The Original Order Block Must Be Strong
The OB that failed must itself have been a high-probability OB by the five requirements above (swept liquidity, FVG on the impulse, MSB on the impulse, aligned with bias at the time, and valid inducement existed). Weak order blocks that fail don't produce reliable breaker blocks — the trapped-trader population isn't large enough to matter.

#### 2. Price Must Break Through with Momentum
The break of the original OB must be impulsive, not slow grind. Look for:
- **Bullish displacement** (large green candles) when breaking a bearish OB.
- **Bearish displacement** (large red candles) when breaking a bullish OB.

Momentum indicates the opposing side has genuinely regained control, which is what traps the original OB participants.

#### 3. Valid Inducement Above/Below the Breaker Block
Before trading the retest of a breaker block, there must be a **valid inducement** on the far side of it — typically a slow-grind move that engineers liquidity above (for a bullish breaker) or below (for a bearish breaker) the zone.

The setup then becomes: price sweeps the inducement liquidity, taps the breaker block, and reverses.

**Without inducement, a breaker block is not tradeable.** This requirement prevents entries on weak retests where no liquidity has been engineered.

### Trade Execution Model

For a bullish breaker block (failed bearish OB):
1. Identify the original strong bearish OB.
2. Confirm price has broken through with bullish displacement.
3. Wait for price to slow-grind above the breaker block, engineering liquidity (inducement).
4. Place a **limit long** at the top of the bullish breaker block.
5. Stop-loss beneath the low of the bullish leg that broke the original OB.
6. Target: prior buy-side liquidity (e.g., equal highs above).

The trade thesis: sweep of inducement liquidity → tap of breaker block → trapped sellers exit → continuation in trend direction.

### Order Block vs. Breaker Block: Quick Comparison

| Aspect | Order Block | Breaker Block |
|---|---|---|
| Origin | Last opposing candle before impulse | Order block that *failed* (got broken through) |
| Mechanism | Smart money defends prior accumulation | Trapped traders exit at break-even |
| Polarity | Same as accumulation direction | **Opposite** of original OB direction |
| Required prerequisite | Strong impulsive move from the candle | A strong OB that was then broken with momentum |
| Required for entry | Valid OB criteria | Valid OB criteria + breakthrough + inducement |

### Integration with Other Concepts

- **Liquidity (Part 2)**: A breaker block works because trapped participants form engineered liquidity inside the zone. The retest *is* a liquidity event.
- **FVGs (Part 4)**: Breaker blocks often have FVGs inside or adjacent to them. Confluence increases probability.
- **MSB (Part 1)**: The break of the original OB often coincides with an MSB in the new direction, confirming the polarity flip.
- **Bias**: A breaker block trade must still align with HTF bias — a bullish breaker against a bearish HTF draw is lower probability.

---

## Part 6: Inducement

### What an Inducement Is

An **inducement** is a *front-run* of a key zone — a price move that taps near but not into a support, resistance, order block, breaker block, or demand/supply zone, reverses briefly, and then later sweeps back through that front-run level *before* finally tapping the actual zone.

Concrete pattern (bullish example):
1. Price approaches a demand zone / bullish OB.
2. Instead of tapping into the zone and bouncing, it front-runs the zone — pulling up *before* reaching it.
3. Price bounces briefly from the front-run, forming a temporary low.
4. Price later returns, sweeps that front-run low (taking out stops resting there), and only *then* taps the actual demand zone.
5. The reversal happens from the real zone, not the front-run.

That swept low (or high, in the bearish case) **is** the inducement.

### Why Inducements Matter

The role of an inducement is to **engineer liquidity inside or adjacent to a key zone**. Without it, smart money has no fuel to fill orders at the zone.

The framework's recurring principle holds: smart money cannot fill orders without liquidity. A support or resistance zone may exist on the chart, but if there is no engineered liquidity pool around it, smart money cannot accumulate sufficient size there, and price will not reverse from it.

This is why:
- A clean support/resistance zone without inducement is still low probability.
- An order block that ticks four of its five requirements but lacks valid inducement is not tradeable (Part 5).
- A breaker block without inducement on the far side is not tradeable (Part 5).

Inducement is not a "nice to have" — it is the mechanism that *qualifies* a zone for institutional participation.

### Valid vs. Invalid Inducements

Identifying that price has front-run a zone is easy. Determining whether that front-run is a **valid** inducement is the actual skill. The distinction parallels the valid/invalid liquidity distinction in Part 2: a level exists on the chart, but the question is whether real engineered liquidity rests there.

#### Invalid Inducement
A front-run produces a low (or high), and price simply bounces and creates a lower high (or higher low) without further structural development. There is no clear price action between the front-run pivot and the eventual sweep that would have trapped retail traders into positions.

Result: stops haven't accumulated at the inducement level. When price sweeps it, there's no real liquidity to mitigate. Smart money cannot fill orders → no reversal at the supposed zone → price slices straight through.

#### Valid Inducement
Between the front-run pivot and the eventual sweep, price behaves in a way that **traps retail traders into positions whose stops rest at the inducement level**. Specifically:

- Price forms one or more **fake breaks of structure** in the direction *opposite* the eventual reversal.
- These fake breaks lure retail into the wrong direction (e.g., into longs above an insignificant high in what's actually a bearish setup).
- Those retail positions place stops below the inducement low (or above the inducement high).
- This creates real, engineered liquidity at the inducement level.

When price then sweeps the inducement, it mitigates that engineered liquidity → smart money fills orders → reversal at the actual zone is high probability.

### How to Test Inducement Validity

For any candidate inducement, ask:

1. **What price action exists between the front-run pivot and the current price?** If there's no meaningful structural development — just a grind without fake breaks or trap patterns — the inducement is likely invalid.
2. **Were retail traders given a reason to enter positions with stops at this level?** If not, no real liquidity is engineered there.
3. **Specifically, were there fake breaks of insignificant highs/lows** (per Part 1) that would have lured breakout/breakdown traders in the wrong direction? These are the most reliable signal that retail stops are now stacked at the inducement.

If the answer to any of these is no, treat the zone as untradeable regardless of how clean the support/resistance looks.

### Example Patterns

**Valid bullish inducement (long setup):**
- A demand zone / bullish OB exists below current price.
- Price front-runs the zone — pulls up before reaching it, forming a low at the front-run.
- Between the front-run and now, price has produced fake bullish breaks of structure (breaking insignificant highs), trapping retail into longs.
- Those trapped longs have stops below the front-run low → engineered sell-side liquidity at the inducement.
- Trade plan: wait for price to sweep the inducement into the actual demand zone; enter long; stop below the demand zone; target the HTF draw on liquidity above.

**Valid bearish inducement (short setup):**
- A supply zone / bearish OB / bearish breaker block exists above current price.
- Price front-runs it — pulls down before reaching it, forming a high at the front-run.
- Between the front-run and now, price has produced fake bearish breaks of insignificant lows, trapping retail into shorts.
- Those trapped shorts have stops above the front-run high → engineered buy-side liquidity at the inducement.
- Trade plan: wait for price to sweep the inducement into the actual supply zone; enter short; stop above the supply zone; target lower liquidity.

**Invalid inducement (skip the trade):**
- A supply zone exists above current price.
- Price forms a high below the zone (potential inducement candidate).
- But between that high and current price, there's just slow grinding upward with no fake breakdowns trapping shorts.
- No stops have accumulated above the candidate inducement.
- When price reaches the supply zone, smart money has no liquidity to work with → zone fails → price continues through it.
- Anyone who shorted the supply zone gets stopped out.

### Integration with Other Concepts

- **Liquidity (Part 2)**: Inducement is one of the primary mechanisms by which engineered liquidity is created. Inducement and engineered liquidity are essentially the same concept seen from two angles — Part 2 explains *what* engineered liquidity is, Part 6 explains *how to find it positioned relative to a tradeable zone*.
- **MSB (Part 1)**: Fake breaks of structure inside the inducement formation are what make it valid. The retail-trapping pattern from Part 1 is the engine that fills the inducement with real liquidity.
- **Order Blocks (Part 5)**: Inducement is requirement #5 of a high-probability OB. Without it, an OB is not tradeable.
- **Breaker Blocks (Part 5)**: Inducement on the far side of a breaker block is the precondition for trading its retest.
- **Ranges (Part 3)**: Range deviations are themselves a form of inducement — front-running the range high/low to engineer liquidity before the actual reversal.

### Why This Concept Is So Often Misapplied

Most traders identify support/resistance, see price front-run it, and assume the front-run *is* the inducement. They enter at the actual zone expecting a reversal. When the zone fails, they're confused.

The error is treating front-running as sufficient. **Front-running is necessary but not sufficient.** What makes a front-run an actual inducement is the engineered liquidity built between the front-run pivot and the eventual sweep — and that liquidity requires specific price action (fake breaks, trapping patterns), not just time passing.

This is the same pattern that appears throughout the framework: a structural condition (a high, a low, a candle close, a front-run) is necessary but only becomes high-probability when liquidity logic also holds.

---

## Part 7: Integrated Setup Checklist

A high-probability setup in this framework requires all of the following to align:

### Bias (HTF Context)
- [ ] Higher-timeframe draw on liquidity identified.
- [ ] Bias direction clear (bullish or bearish) based on HTF structure and unmitigated HTF liquidity.
- [ ] If ambiguous, wait — do not force a trade.

### Range Context
- [ ] Valid range identified (three pivots, strong solo pivot, mid-range respected) — OR clear trending environment.
- [ ] Current price location within the range understood (near high, near low, mid-range).
- [ ] Approach selected: deviation (default) or breakout/breakdown (when bias is strong).

### Liquidity
- [ ] The pivot being traded mitigates a **valid** (engineered) liquidity pool.
- [ ] The sweep is deep enough to constitute genuine mitigation, not a wick.
- [ ] Engineered liquidity is identifiable in prior price action (fake breaks, sharp rejections, etc.).

### Market Structure Trigger
- [ ] Significant high/low identified (one that led to lower low / higher high).
- [ ] Candle has opened AND closed beyond that level.
- [ ] Candle is on the highest timeframe where the reversal structure is visible.
- [ ] Execution timeframe is M15 or higher.

### Entry Refinement (Fair Value Gaps)
- [ ] Did the leg that created the MSB also leave behind a high-probability FVG?
- [ ] If yes, consider a limit entry at the FVG rather than market-executing — same invalidation, better RR.
- [ ] FVG must be high probability (three matching-color candles); do not refine entries on low-probability FVGs.
- [ ] An FVG never substitutes for a valid setup; it only refines entries within one.

### Entry Refinement (Order Blocks / Breaker Blocks)
For an **order block entry**, all five requirements must be met:
- [ ] OB swept an important liquidity pool while being formed.
- [ ] OB produced a high-probability FVG on the impulsive leg away.
- [ ] OB produced a valid MSB on the impulsive leg.
- [ ] OB aligns with HTF bias (this is non-negotiable).
- [ ] Valid inducement exists above (bullish OB) or below (bearish OB).
- [ ] Limit entry at OB; stop beyond OB; target the next logical liquidity pool.

For a **breaker block entry**:
- [ ] The original OB met the five OB requirements at the time it formed.
- [ ] Price broke through it with displacement (impulsive, not slow grind).
- [ ] Valid inducement has formed on the far side of the breaker block.
- [ ] Without inducement, the breaker block is not tradeable.

General:
- [ ] Confluence (OB + FVG + swept liquidity + MSB) is the refinement stack, not a substitute for the core setup.

### Inducement Validation
Whenever an inducement is required (any OB, any breaker block, any range deviation), validate it explicitly:
- [ ] Did price front-run the actual zone? (front-run pivot identified)
- [ ] Between the front-run pivot and current price, has price produced fake breaks of structure that trapped retail in the wrong direction?
- [ ] Are retail stops now likely concentrated at the front-run level (i.e., is engineered liquidity actually present)?
- [ ] If the answer to any of these is no → the inducement is invalid → skip the trade regardless of how clean the zone looks.

### Invalidation
- [ ] Clean invalidation point exists (the swept low/high becomes the stop).
- [ ] Without clean invalidation, the setup is not tradeable — even if everything else looks good.

If any of these fail, the setup is lower probability or untradeable. The framework rewards patience: skipping marginal setups preserves capital and psychological state for the genuinely high-probability ones.

---

## Glossary

**Bias** — Directional expectation derived from higher-timeframe context (draw on liquidity, structure, wicks to fill). Overrides default trade selection.

**Breaker Block** — An order block that has been broken through by price, causing it to flip polarity. A previously-bullish OB becomes resistance; a previously-bearish OB becomes support. Tradeable when the original OB was strong, the breakthrough showed displacement, and valid inducement formed on the far side.

**Breaker Block (Bullish)** — A failed bearish OB that now acts as support; previous sellers are trapped underwater and exit on retest, adding to upward pressure.

**Breaker Block (Bearish)** — A failed bullish OB that now acts as resistance; previous buyers are trapped underwater and exit on retest, adding to downward pressure.

**Break of Structure (BoS / MSB)** — Price closing beyond a significant high or low; the only valid entry trigger in this framework.

**Candle Open and Close Rule** — A break is only confirmed when a candle opens *and* closes beyond the level; wicks do not confirm.

**Deviation** — Price briefly pushing outside a range before returning inside. Default trade type within established ranges.

**Displacement** — A strong, momentum-driven move (often with large candles and FVGs) indicating one side has regained control. Required when judging whether a break of an order block is genuine enough to flip it into a breaker block.

**Draw on Liquidity** — The next significant HTF liquidity pool toward which price is gravitating; the directional anchor for bias.

**Engineered Liquidity** — Liquidity actively created by price action that traps traders into predictable positions with stops at known levels.

**Fair Value Gap (FVG)** — A price imbalance created by a fast move, defined by three consecutive candles where the wick of the first and third do not overlap. Acts as a magnet for price; used to refine entries.

**Fair Value Gap (Bullish)** — Three green candles where the top of candle 1 and the bottom of candle 3 do not touch. The gap is the unfilled price region between them.

**Fair Value Gap (Bearish)** — Three red candles where the bottom of candle 1 and the top of candle 3 do not touch.

**Fair Value Gap (High Probability)** — FVG where all three candles share the direction's color (3 greens or 3 reds). The only FVG type used for trade decisions.

**Fair Value Gap (Low Probability)** — A gap exists but the third candle is the opposite color. Not traded.

**Fake Break of Structure** — Break of an *insignificant* high/low; a trap, never traded as a trigger.

**High Probability Break of Structure** — Highest-quality MSB variant; valid break + significant sweep + additional confluences.

**Inducement** — A front-run of a key zone (support/resistance, OB, breaker block) where price taps near the zone, reverses briefly, then later sweeps back through that front-run before tapping the actual zone. Functions to engineer liquidity that smart money needs to fill orders at the zone.

**Inducement (Valid)** — A front-run where, between the front-run pivot and the eventual sweep, price has produced fake breaks of structure that trapped retail in the opposite direction, accumulating real stops at the inducement level. Required for any OB, breaker block, or range deviation trade.

**Inducement (Invalid)** — A front-run that occurred but was followed by no structural development that would trap retail. No real liquidity has been engineered there, so the eventual sweep mitigates nothing meaningful. Zones backed only by invalid inducements should not be traded.

**Insignificant High** — A high that only leads to a higher low; not a meaningful structural reference.

**Insignificant Low** — A low that only leads to a lower high.

**Invalidation** — The price level that, if reached, proves the trade thesis wrong; defines the stop-loss.

**Invalid Liquidity Pool** — A high/low without engineered liquidity behind it; no real stops/orders rest there.

**Invalid Liquidity Sweep** — Mitigation of an invalid pool; provides no probability boost to the subsequent break.

**Liquidity** — Volume of opposing orders or stop-losses concentrated at a price level.

**Liquidity Pool** — Concentration of orders/stops at a level; target of smart money price pushes.

**Mid-Range Level** — The 50% level of a range; must be respected as S/R for the range to be valid.

**Order Block** — The last opposing candle before a strong impulsive move; marks where smart money accumulated or distributed positions. Acts as a support or resistance zone on retest.

**Order Block (Bullish)** — The last red candle before a strong leg up; acts as demand/support on retest.

**Order Block (Bearish)** — The last green candle before a strong leg down; acts as supply/resistance on retest.

**Order Block (Valid)** — Any last-opposing-candle that meets the basic structural definition. Not sufficient on its own to trade.

**Order Block (High Probability)** — A valid OB that ticks all five requirements: (1) swept an important liquidity pool while forming, (2) produced an FVG on the impulsive leg, (3) produced an MSB on the impulsive leg, (4) aligns with HTF bias, (5) has valid inducement on the relevant side. Only high-probability OBs are tradeable.

**Pivot** — A high or low touch point used to define a range.

**Range** — A sideways price environment where price oscillates between defined high and low boundaries.

**Significant Break of Structure** — Valid break where the second pivot mitigates a valid liquidity pool.

**Significant High** — A high that leads to a lower low; a meaningful structural reference.

**Significant Low** — A low that leads to a higher high.

**Slippage** — Difference between intended and actual fill price; occurs when liquidity at a level is insufficient.

**Smart Money** — Large, profitable market participants whose order size forces them to seek out liquidity pools to fill and exit positions.

**Solo Pivot** — The single-touch pivot of a range; must be a strong, sharp rejection to qualify.

**Swing Failure Pattern (SFP)** — Price wicks beyond a significant level but does not close beyond it; takes stops without confirming structure.

**Sweep** — Price mitigating a liquidity pool by taking out the stops/orders resting there.

**Time of Price / Time Alignment** — Whether a pivot coincides with a meaningful candle's high or low (session, 4H, daily); adds probability to MSB triggers.

**Valid Break of Structure** — Candle open + close beyond a significant level, but without significant liquidity mitigation on the pivots. Lowest of the three valid MSB tiers.

**Valid Liquidity Pool** — A level where prior price action engineered real concentrated liquidity.

**Valid Liquidity Sweep** — Mitigation of a valid pool; provides genuine probability boost to subsequent structure breaks.

---

## Notes for the Receiving Model

1. **Concepts are interdependent.** Do not apply any Part in isolation. A valid MSB without engineered liquidity is structurally clean but probabilistically weak. Liquidity context without MSB lacks an entry trigger. Range context without bias produces directionless setups. The Decision Hierarchy section makes the resolution rules explicit when concepts appear to compete.

2. **Defaults can be overridden.** Deviations are the default within ranges, but bias overrides. Similarly, valid MSBs are tradeable in principle but require additional alignment to be high-probability.

3. **When in doubt, do not trade.** This framework explicitly favors patience: fewer, higher-quality setups over high-frequency execution. The cost of skipping a marginal setup is zero.

4. **This is a discretionary framework being operationalized for a quantitative model.** Concepts like "strong pivot," "deep sweep," and "engineered liquidity" originated from chart-reading judgment. The OHLC Translation Guide gives the model concrete numerical definitions to work with, but these definitions are approximations of the underlying intuition — the model should treat them as starting points that may need calibration during backtesting.

5. **Discretionary thresholds need empirical tuning.** Several rules in the OHLC Translation Guide involve thresholds (e.g., "body size > N× trailing average," "equal highs within 0.5–1% tolerance," "pivot N candles to either side"). The exact values should be tuned against historical data per instrument and timeframe; the values given are reasonable starting points, not fixed parameters.

6. **Out-of-scope topics** referenced but not fully developed in these seven lessons: specific deviation/breakout models (multiple distinct models exist within range trading; only the general principle is covered here), full enumeration of the "high probability MSB" confluences (the fourth MSB category is named but its full criteria are not detailed), and the valid higher-low / lower-high candle rules referenced in lesson 1. If asked about these, acknowledge they are part of the broader framework but not detailed here.

7. **Iteration loop.** This document is the starting kennisbasis. As the downstream model is backtested, recurring error patterns should be mapped back to concepts in this document. If errors cluster around a concept that is underspecified here (e.g., "the model keeps misidentifying inducements"), that is a signal to add a deeper lesson on that topic rather than tune around the symptom.

