# Problem Statement — A Pareto-Based Multi-Objective Framework for Price–Discount Optimization in Multi-Product Retail Systems

## 1. Overview

A retailer has 9 products (P0..P8). For each product the decision variables are **Price** and **Discount** (discount ∈ [0,1], expressed as a fraction). The retailer wants to find a set of Pareto-optimal price/discount assignments that simultaneously:

* **Maximize Expected Sales** (revenue after discount, or number of units sold times discounted price),
* **Minimize Revenue Loss** (loss caused by discounting relative to full price revenue),
* **Maximize Engagement** (a proxy metric that captures customer interest driven by discount and sales),

These three objectives are conflicting: discounts can increase engagement and units sold but increase revenue loss. NSGA-II (a multi-objective evolutionary algorithm) is used to find trade-off front solutions.

---

## 2. Inputs (given)

For each product i we are given:

* `Product ID` (P0..P8)
* `Price` (P_i) — current or base price
* `Discount` (D_i) — initial discount (0..1)
* `Elasticity` (E_i) — provided as the *number of buyers* (units demanded) in the initial stage.

> **IMPORTANT:** In many economics texts, elasticity is price sensitivity. In this project we treat the provided `Elasticity` as **base demand (units)** for the initial price/discount (you may rename it to `BaseDemand` to avoid confusion).

---

## 3. Decision variables

For each product i (i = 0..8):

* **Price (p_i)** — continuous variable, lower and upper bounds should be specified (e.g. [0.5 × P_i_initial, 1.5 × P_i_initial] or business-driven min/max).
* **Discount (d_i)** — continuous variable in `[0.0, 1.0]` (or smaller upper bound like `0.5` for realistic promotions).

All decision variables form one solution vector of length `2 × N_products`.

---

## 4. Objective functions and formulas

We define per-product quantities and then aggregate them across all products.

### Notation

* \( p_i \): price for product i chosen by optimizer (₹)  
* \( d_i \): discount fraction for product i (0..1)  
* \( e_i^0 \): initial (given) elasticity / base demand (units) at the initial price and discount  
* \( E_i(p_i, d_i) \): realized demand (units) for chosen price and discount — **this can be modeled** (two approaches below)

---

### 4.1 Simple approach — treat Elasticity as fixed units (baseline)

If you treat the given `Elasticity` as fixed number of buyers (do not change with price), then:

* **Total Sales (revenue after discount)**  
  \[
  TS_i = (p_i - p_i \times d_i) \times E_i
  \]

* **Revenue Loss (difference between full-price and discounted revenue)**  
  \[
  RL_i = (p_i \times E_i) - TS_i = (p_i \times d_i) \times E_i
  \]

* **Engagement**  
  \[
  ENG_i = d_i \times E_i
  \]
  Larger discounts and more buyers increase engagement.

* **Profit (if unit cost c_i is known)**  
  \[
  PROFIT_i = (p_i (1 - d_i) - c_i) \times E_i
  \]
  If cost `c_i` is unknown, assume `c_i = 0` and profit equals `TS_i`.

* **Aggregate objectives**
  \[
  \begin{aligned}
  Total\ Sales &= \sum_i TS_i \\
  Total\ Revenue\ Loss &= \sum_i RL_i \\
  Total\ Engagement &= \sum_i ENG_i \\
  Total\ Profit &= \sum_i PROFIT_i
  \end{aligned}
  \]

**Optimization directions:**

* Maximize `Total Sales`
* Minimize `Total Revenue Loss`
* Maximize `Total Engagement`

Because NSGA-II minimizes objectives by default, convert maximizations by negating them (e.g. minimize `-TotalSales`, minimize `TotalRevenueLoss`, minimize `-TotalEngagement`).

---

### 4.2 Realistic approach — model demand as function of price and discount

If you want the demand to change with price and discount (recommended), use a demand function.

**Option A — Constant elasticity model (power law)**  
\[
E_i(p_i, d_i) = e_i^0 \times \left(\frac{p_i}{p_i^0}\right)^{-\epsilon_i} \times (1 + \gamma_i \times d_i)
\]

* \( p_i^0 \): base price where `e_i^0` was measured  
* \( \epsilon_i \): price elasticity exponent (>0, typically 0.5–3)  
* \( \gamma_i \): demand lift from discount (e.g. 0.2–1.0)

**Option B — Linear demand with discount lift**  
\[
E_i(p_i, d_i) = \max\left(0, e_i^0 + a_i (p_i - p_i^0) + b_i d_i\right)
\]
* \( a_i < 0 \): higher price → lower demand  
* \( b_i > 0 \): higher discount → higher demand  

---

## 5. NSGA-II algorithm configuration (recommended)

| Parameter | Recommended Value |
|------------|------------------|
| Population size | 100–200 |
| Generations | 200 |
| Selection | Binary tournament using rank + crowding distance |
| Crossover | SBX (p_c = 0.9, η_c = 15–20) |
| Mutation | Polynomial (p_m = 1/(2N), η_m = 20) |
| Encoding | Real-valued vector (prices and discounts) |
| Constraints | Clip prices and discounts to bounds |

---

## 6. Implementation steps (how to evaluate one individual)

1. Read solution vector: prices `p_i` and discounts `d_i` for all products.  
2. Compute `E_i(p_i,d_i)` using chosen demand model.  
3. Compute `TS_i`, `RL_i`, `ENG_i`, `PROFIT_i` using formulas from §4.  
4. Aggregate totals for all products.  
5. Return objective tuple to NSGA-II (convert maximizations to minimizations).  

---

## 7. Output format (`output.md`)

The output file should contain two sections:

- **Initial Products**
- **Final Optimized Products**

Each section has a table and the `Total Overall Profit` line.

**Table columns:**  
`Product ID | Price | Discount | Elasticity | Total Sales | Revenue Loss | Engagement`

Values should be neatly rounded (e.g. 2 decimals for prices, whole numbers for units).

---

## 8. Example markdown output

```markdown
# Initial Products

| Product ID | Price | Discount | Elasticity | Total Sales | Revenue Loss | Engagement |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 100.00 | 0.10 | 200 | 18,000.00 | 2,000.00 | 20.00 |
| ... | ... | ... | ... | ... | ... | ... |

**Total Overall Profit:** ₹ 36,000.00

# Final Optimized Products

| Product ID | Price | Discount | Elasticity | Total Sales | Revenue Loss | Engagement |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 95.00 | 0.05 | 230 | 20,850.00 | 1,150.00 | 11.50 |
| ... | ... | ... | ... | ... | ... | ... |

**Total Overall Profit:** ₹ 40,120.00
