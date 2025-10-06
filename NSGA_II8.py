import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Pymoo Library Imports ---
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.termination import get_termination
from pymoo.optimize import minimize

# --- -1. Data Setup (Identical to previous script) ---
csv_data = """Product ID,Price,Discount,Base Demand,Cost
P0,100.00,0.10,200,60.00
P1,150.00,0.05,150,90.00
P2,200.00,0.15,100,120.00
P3,75.00,0.20,300,45.00
P4,300.00,0.10,80,180.00
P5,50.00,0.05,500,30.00
P6,500.00,0.25,50,300.00
P7,120.00,0.12,250,72.00
P8,85.00,0.08,400,51.00
"""
try:
    with open("products.csv", "w") as f:
        f.write(csv_data)
    print("✅ 'products.csv' file created successfully for the simulation.")
except IOError as e:
    print(f"Error creating file: {e}")
    exit()

# --- 0. Problem Definition & Data Loading ---
try:
    df = pd.read_csv('products.csv')
except FileNotFoundError:
    print("Error: 'products.csv' not found. Please ensure the data file is in the same directory.")
    exit()

NUM_PRODUCTS = len(df)
NUM_VARIABLES = NUM_PRODUCTS * 2

# --- 1. Pymoo Problem Definition ---
class PricingProblem(ElementwiseProblem):
    def __init__(self):
        # Define bounds for variables: [prices..., discounts...]
        price_bounds = [(p * 0.5, p * 1.5) for p in df['Price']]
        discount_bounds = [(0.0, 0.5)] * NUM_PRODUCTS
        bounds = np.array(price_bounds + discount_bounds)

        super().__init__(
            n_var=NUM_VARIABLES,
            n_obj=3,
            n_ieq_constr=0, # No inequality constraints
            xl=bounds[:, 0], # Lower bounds
            xu=bounds[:, 1]  # Upper bounds
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """
        This is the core evaluation function called by pymoo for each solution 'x'.
        """
        prices = x[:NUM_PRODUCTS]
        discounts = x[NUM_PRODUCTS:]
        
        base_demands = df['Base Demand'].values
        
        # --- Calculate Metrics (same formulas as before) ---
        total_sales = (prices * (1 - discounts)) * base_demands
        revenue_loss = (prices * discounts) * base_demands
        engagement = discounts * base_demands

        # --- Aggregate Objectives ---
        agg_total_sales = np.sum(total_sales)
        agg_revenue_loss = np.sum(revenue_loss)
        agg_engagement = np.sum(engagement)

        # --- Set Objective Values for Pymoo ---
        # Objective 1: Maximize Total Sales (minimize -TotalSales)
        # Objective 2: Minimize Revenue Loss
        # Objective 3: Maximize Engagement (minimize -Engagement)
        out["F"] = [-agg_total_sales, agg_revenue_loss, -agg_engagement]

# --- 2. Helper Functions for Reporting (Almost identical) ---
def get_full_metrics_for_report(solution):
    """Calculates all metrics including profit for a given solution vector."""
    prices = np.array(solution[:NUM_PRODUCTS])
    discounts = np.array(solution[NUM_PRODUCTS:])
    base_demands = df['Base Demand'].values
    costs = df['Cost'].values
    total_sales = (prices * (1 - discounts)) * base_demands
    revenue_loss = (prices * discounts) * base_demands
    engagement = discounts * base_demands
    profit = ((prices * (1 - discounts)) - costs) * base_demands
    return total_sales, revenue_loss, engagement, profit

def create_report_df(solution_vector):
    """Creates a formatted DataFrame and calculates total profit."""
    ts, rl, eng, profit = get_full_metrics_for_report(solution_vector)
    report = pd.DataFrame({
        'Product ID': df['Product ID'], 'Price': solution_vector[:NUM_PRODUCTS],
        'Discount': solution_vector[NUM_PRODUCTS:], 'Elasticity': df['Base Demand'],
        'Total Sales': ts, 'Revenue Loss': rl, 'Engagement': eng
    })
    total_profit = np.sum(profit)
    return report, total_profit

# --- 3. Results Display and Comparison Report Generation ---
def display_and_log_results(result):
    print("\n📈 Analyzing results and generating comparison report...")

    # Extract solutions (X) and objective values (F) from the result object
    solutions = result.X
    objectives = result.F

    # --- 1. Select a "Low Loss" Solution from the Pareto Front ---
    sales, rev_loss, engagement = -objectives[:, 0], objectives[:, 1], -objectives[:, 2]
    norm_sales = (sales - sales.min()) / (sales.max() - sales.min() + 1e-6)
    norm_loss = (rev_loss - rev_loss.min()) / (rev_loss.max() - rev_loss.min() + 1e-6)
    norm_eng = (engagement - engagement.min()) / (engagement.max() - engagement.min() + 1e-6)
    
    loss_weight = 2.0
    distances_to_utopia = np.sqrt((norm_sales - 1)**2 + (loss_weight * (norm_loss - 0))**2 + (norm_eng - 1)**2)
    
    chosen_idx = np.argmin(distances_to_utopia)
    chosen_solution = solutions[chosen_idx]

    # --- 2. Prepare DataFrames for the Markdown Report ---
    initial_solution = np.concatenate([df['Price'].values, df['Discount'].values])
    initial_report_df, initial_total_profit = create_report_df(initial_solution)
    final_report_df, final_total_profit = create_report_df(chosen_solution)
    
    # --- 3. Write to Comparison Markdown File ---
    output_filename = "output_comparison.md"
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("# Comparison of Optimization Results\n\n")
        f.write("This file compares the initial state with the final state optimized using the **Pymoo library**.\n\n")
        
        f.write("## Initial Products\n\n")
        f.write(initial_report_df.to_markdown(index=False, floatfmt=",.2f"))
        f.write(f"\n\n**Total Overall Profit:** ₹ {initial_total_profit:,.2f}\n\n")
        
        f.write("---\n\n")
        
        f.write("## Final Optimized Products (Using Pymoo Library)\n\n")
        f.write(final_report_df.to_markdown(index=False, floatfmt=",.2f"))
        f.write(f"\n\n**Total Overall Profit:** ₹ {final_total_profit:,.2f}\n")

    print(f"✅ Successfully generated comparison report: '{output_filename}'")
    
    # --- 4. Plot the Pareto Front ---
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(sales, rev_loss, engagement, c=rev_loss, cmap='viridis', s=60, label='Pareto Optimal Solutions')
    ax.scatter(sales[chosen_idx], rev_loss[chosen_idx], engagement[chosen_idx], 
               c='red', s=200, marker='*', label='Chosen Low-Loss Solution', depthshade=False)
    ax.set_xlabel('Maximize: Total Sales (₹)')
    ax.set_ylabel('Minimize: Total Revenue Loss (₹)')
    ax.set_zlabel('Maximize: Total Engagement')
    ax.set_title('Pareto Front of Optimal Pricing Strategies (Pymoo)')
    ax.legend()
    fig.colorbar(sc, label='Total Revenue Loss (₹)')
    plt.show()

# --- 4. Main Execution ---
if __name__ == "__main__":
    # 1. Instantiate the problem
    problem = PricingProblem()

    # 2. Configure the NSGA-II algorithm
    algorithm = NSGA2(
        pop_size=200,
        n_offsprings=200, # Typically same as pop_size
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=20),
        mutation=PM(eta=20),
        eliminate_duplicates=True
    )

    # 3. Define the termination condition
    termination = get_termination("n_gen", 200)

    # 4. Run the optimization
    print("\n🚀 Starting NSGA-II Optimization with Pymoo...")
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=1, # For reproducible results
        save_history=True,
        verbose=True # This will print the generation progress
    )
    print("\n✅ Optimization Finished!")

    # 5. Display results and generate the comparison log file
    display_and_log_results(res)