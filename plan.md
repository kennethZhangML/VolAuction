### **1. Data Collection and Preprocessing**  
_(Completed 2025-01-21)_

**Goal**: Collect granular auction and pre-market data to establish inputs for modeling volatility and identifying mispricings.

**What to Build**:
- **Auction Data Ingestion Pipeline**:
  - Pull SPX option auction prices and indicative quotes from CBOE.
  - Focus on ATM and near-ATM strikes for 0DTE options.
  - Capture full order book snapshots during auction periods if available.

- **ES/SPX Futures Overnight Feed**:
  - Use CME’s ES futures data to compute overnight return distributions.
  - Calculate return volatility from previous close to auction print (e.g., 4 PM → 9:30 AM).

- **Data Sync & Storage**:
  - Normalize timestamps across datasets (e.g., options, ES futures, SPX cash index).
  - Store raw + processed data in a database (e.g., PostgreSQL or Parquet files) for reproducibility.

**Metrics**:
- Overnight return volatility.
- Auction mid-prices and implied volatilities.
- Order book imbalance at the open.

---

### **2. Modeling the Realized Volatility and Theoretical Pricing**  
_(Completed 2025-02-20)_

**Goal**: Build robust models for open-to-close volatility forecasts and compute theoretical prices from these vol forecasts.

**What to Build**:
- **Realized Volatility Estimators**:
  - Compute overnight vol using:
    - Standard deviation of ES minute bars.
    - Parkinson (High-Low), Garman-Klass, and Yang-Zhang estimators (compare and ensemble).
    - Overnight jump indicators using Merton jump-diffusion model.
    - Rolling volatility-of-volatility metrics from VIX/VVIX or front VIX futures spreads.

- **Volatility Model**:
  - Fit GBM or jump-diffusion SDEs to overnight SPX return distributions:
    $$ dS_t = \mu S_t dt + \sigma S_t dW_t + J_t dN_t $$
    - $J_t$ jump sizes; $dN_t$ Poisson jump process.
    - Estimate jump intensity $\lambda$ and size distribution empirically.

- **Theoretical Price Generator**:
  - Use BS-PDE with Crank-Nicholson or finite difference to solve for prices using $\hat{\sigma}_{real}$.
  - Alternatively use Black-Scholes formula if model is pure diffusion.
  - Compute fair value ATM price and implied vol for a range of strikes.

---

### **3. Alpha Signal Generation**  
_(In-Progress)_

**Goal**: Develop alpha signals based on pricing dislocations and mispriced risk premia.

**What to Build**:
- **Vol Dislocation Signal**:
  - Invert auction price to get $\sigma_{auction}$.
  - Compare to $\hat{\sigma}_{real}$ from Step 2.
  - Generate a signal score:
    $$ \text{Vol Edge} = \sigma_{auction} - \hat{\sigma}_{real} $$
    - Position: Short vol if edge > threshold; long vol if edge < -threshold.

- **Directional Bias Overlay**:
  - Use overnight drift estimate from SDE model + futures imbalance.
  - Add directional skew to straddle (e.g., shift to strangle or call-weighted if bullish bias).

- **Skew Dislocation**:
  - Compute IV skew at auction using 25-delta call vs. put IVs.
  - Model expected skew from overnight distribution asymmetry.
  - Signal: Trade relative premium or tail risk options.

- **Term Structure Slope Signal**:
  - Construct IV term structure using 0DTE and 1DTE options.
  - Use slope or extrapolated intercept as fair 0DTE IV.
  - Trade vol based on discount/premium to this curve.

- **Overnight Vol-of-Vol Signal**:
  - Track VIX/VVIX or VIX futures curve slope (e.g., F1-F2).
  - Model historical correlation between vol-of-vol and SPX 0DTE IV.
  - Signal: If VVIX is up but SPX IV is flat, buy gamma.

---

### **4. Execution Layer**  
_(SPX-IDX Coded Executors)_

**Goal**: Deploy trades and manage risk dynamically based on signal strength and auction execution quality.

**What to Build**:
- **Order Placement System**:
  - Implement logic to place synthetic straddles/strangles at or just after open.
  - Limit orders based on bid-ask spread and signal strength.
  - Fallback to aggressive execution if auction price deviates too far from model.

- **Delta-Hedging Module**:
  - Implement gamma-aware delta hedging logic.
  - Use ES or SPX futures to hedge dynamically every N seconds/minutes post-open.
  - Allow for signal-based re-hedging (more frequent if high gamma).

- **Trade Lifecycle & Risk Manager**:
  - Define exit criteria:
    - Time-based (e.g., exit at 11 AM),
    - IV reversion (e.g., if IV returns to model level),
    - Loss/gain thresholds.
  - Track theta decay and gamma risk throughout holding window.

---

### **Optional Enhancements**
- **Machine Learning Overlay**:
  - Train a regression or neural network model on historical auction mispricings.
  - Features: Overnight vol, VVIX, ES skew, term structure, macro release flags.
  - Predict mispricing probability or confidence score for each trade.

- **Backtesting Framework**:
  - Build full historical simulation from auction price → delta hedging → EOD PnL.
  - Incorporate transaction costs and slippage from real market spreads.

- **Live Dashboard**:
  - Build real-time visual display of:
    - Fair vol vs. auction vol
    - Signal scores for each strategy
    - Greeks and live PnL breakdown

---

