import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import io

# --- -1. Data Setup ---
# For reproducibility, this script creates the 'products.csv' file.
# In a real scenario, you would just have this file in your project directory.
# We've renamed 'Elasticity' to 'Base Demand' for clarity as requested.
# A 'Cost' column is added, assuming cost is 60% of the initial price.
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

# Load the product data from the CSV file
try:
    df = pd.read_csv('products.csv')
except FileNotFoundError:
    print("Error: 'products.csv' not found. Please ensure the data file is in the same directory.")
    exit()

NUM_PRODUCTS = len(df)
# Chromosome structure: [p0, p1,..., pN-1, d0, d1,..., dN-1]
NUM_VARIABLES = NUM_PRODUCTS * 2


# --- 1. Objective Functions & Formulas ---
def get_all_metrics(solution):
    """
    Calculates all per-product metrics for a given solution vector.
    A solution vector contains [prices..., discounts...].
    """
    prices = np.array(solution[:NUM_PRODUCTS])
    discounts = np.array(solution[NUM_PRODUCTS:])
    
    base_demands = df['Base Demand'].values
    costs = df['Cost'].values

    # Formulas from the problem description (Simple Approach)
    total_sales = (prices * (1 - discounts)) * base_demands
    revenue_loss = (prices * discounts) * base_demands
    engagement = discounts * base_demands
    profit = ((prices * (1 - discounts)) - costs) * base_demands
    
    return total_sales, revenue_loss, engagement, profit

def calculate_objectives(solution):
    """
    Calculates the three aggregate objective values for NSGA-II.
    """
    total_sales, revenue_loss, engagement, _ = get_all_metrics(solution)
    
    # Aggregate objectives
    agg_total_sales = np.sum(total_sales)
    agg_revenue_loss = np.sum(revenue_loss)
    agg_engagement = np.sum(engagement)
    
    # NSGA-II minimizes. To maximize Sales/Engagement, we minimize their negative.
    # To minimize Revenue Loss, we minimize its positive value.
    return np.array([-agg_total_sales, agg_revenue_loss, -agg_engagement])


# --- 2. NSGA-II Core Functions ---

def non_dominated_sort(pop_obj_values):
    """Performs non-dominated sorting on the population."""
    pop_size = pop_obj_values.shape[0]
    dominating_counts = np.zeros(pop_size, dtype=int)
    dominated_sets = [[] for _ in range(pop_size)]
    fronts = [[]]

    for p_idx in range(pop_size):
        for q_idx in range(p_idx + 1, pop_size):
            p_sols = pop_obj_values[p_idx]
            q_sols = pop_obj_values[q_idx]
            
            if np.all(p_sols <= q_sols) and np.any(p_sols < q_sols): # p dominates q
                dominated_sets[p_idx].append(q_idx)
                dominating_counts[q_idx] += 1
            elif np.all(q_sols <= p_sols) and np.any(q_sols < p_sols): # q dominates p
                dominated_sets[q_idx].append(p_idx)
                dominating_counts[p_idx] += 1
                
    for i in range(pop_size):
        if dominating_counts[i] == 0:
            fronts[0].append(i)
            
    front_idx = 0
    while front_idx < len(fronts):
        next_front = []
        for p_idx in fronts[front_idx]:
            for q_idx in dominated_sets[p_idx]:
                dominating_counts[q_idx] -= 1
                if dominating_counts[q_idx] == 0:
                    next_front.append(q_idx)
        if next_front:
            fronts.append(next_front)
        front_idx += 1
        
    return fronts

def calculate_crowding_distance(front_indices, pop_obj_values):
    """Calculates the crowding distance for each solution in a front."""
    front_size = len(front_indices)
    if front_size == 0:
        return np.array([])
        
    num_objectives = pop_obj_values.shape[1]
    distances = np.zeros(front_size)
    front_values = pop_obj_values[front_indices]
    
    for obj_idx in range(num_objectives):
        sorted_indices = np.argsort(front_values[:, obj_idx])
        distances[sorted_indices[0]] = np.inf
        distances[sorted_indices[-1]] = np.inf
        
        obj_min = front_values[sorted_indices[0], obj_idx]
        obj_max = front_values[sorted_indices[-1], obj_idx]
        
        if obj_max == obj_min:
            continue
            
        for i in range(1, front_size - 1):
            distances[sorted_indices[i]] += (front_values[sorted_indices[i+1], obj_idx] - 
                                             front_values[sorted_indices[i-1], obj_idx]) / (obj_max - obj_min)
                                             
    return distances


# --- 3. Genetic Operators ---

def tournament_selection(population, ranks, distances):
    """Selects a parent using binary tournament selection."""
    idx1, idx2 = random.sample(range(len(population)), 2)
    if ranks[idx1] < ranks[idx2]: return population[idx1]
    elif ranks[idx2] < ranks[idx1]: return population[idx2]
    else:
        return population[idx1] if distances[idx1] > distances[idx2] else population[idx2]

def simulated_binary_crossover(p1, p2, eta_c=20):
    """Performs simulated binary crossover (SBX)."""
    c1, c2 = p1.copy(), p2.copy()
    for i in range(NUM_VARIABLES):
        if random.random() < 0.9: # Crossover probability
            u = random.random()
            if u <= 0.5:
                beta = (2 * u)**(1 / (eta_c + 1))
            else:
                beta = (1 / (2 * (1 - u)))**(1 / (eta_c + 1))
            c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
            c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
    return c1, c2

def polynomial_mutation(individual, bounds, eta_m=20):
    """Performs polynomial mutation."""
    mutated_ind = individual.copy()
    prob_mut = 1.0 / NUM_VARIABLES
    for i in range(NUM_VARIABLES):
        if random.random() < prob_mut:
            delta1 = (mutated_ind[i] - bounds[i][0]) / (bounds[i][1] - bounds[i][0])
            delta2 = (bounds[i][1] - mutated_ind[i]) / (bounds[i][1] - bounds[i][0])
            u = random.random()
            if u <= 0.5:
                val = 2*u + (1-2*u) * (max(0, 1-delta1)**(eta_m+1))
                delta_q = val**(1/(eta_m+1)) - 1
            else:
                val = 2*(1-u) + 2*u * (max(0, 1-delta2)**(eta_m+1))
                delta_q = 1 - val**(1/(eta_m+1))
            mutated_ind[i] += delta_q * (bounds[i][1] - bounds[i][0])
    return mutated_ind

def enforce_bounds(solution, bounds):
    """Clips solution values to stay within defined bounds."""
    return np.array([np.clip(solution[i], bounds[i][0], bounds[i][1]) for i in range(len(solution))])


# --- 4. Main NSGA-II Algorithm ---
def run_nsga2():
    print("\n🚀 Starting NSGA-II Optimization...")
    # --- Hyperparameters ---
    POP_SIZE = 200
    NUM_GENERATIONS = 200
    
    # --- Define bounds for decision variables ---
    # Price: Allow +/- 50% from original price
    # Discount: Allow between 0% and 50%
    price_bounds = [(p * 0.5, p * 1.5) for p in df['Price']]
    discount_bounds = [(0.0, 0.5)] * NUM_PRODUCTS
    bounds = price_bounds + discount_bounds
    
    # --- Initialization ---
    print(f"1. Initializing Population of size {POP_SIZE}...")
    population = np.zeros((POP_SIZE, NUM_VARIABLES))
    for i in range(POP_SIZE):
        prices = [random.uniform(b[0], b[1]) for b in price_bounds]
        discounts = [random.uniform(b[0], b[1]) for b in discount_bounds]
        population[i, :] = prices + discounts
        
    # --- Main Evolution Loop ---
    for gen in range(NUM_GENERATIONS):
        print(f"\r--- Running Generation {gen + 1}/{NUM_GENERATIONS} ---", end="")
        
        # --- Evaluation ---
        pop_obj_values = np.array([calculate_objectives(ind) for ind in population])
        
        # --- Create Offspring ---
        offspring_population = np.zeros((POP_SIZE, NUM_VARIABLES))
        
        # Re-calculate ranks and distances for selection
        fronts = non_dominated_sort(pop_obj_values)
        ranks = np.zeros(POP_SIZE, dtype=int)
        distances = np.zeros(POP_SIZE)
        for i, front in enumerate(fronts):
            ranks[front] = i
            front_distances = calculate_crowding_distance(front, pop_obj_values)
            for j, idx in enumerate(front):
                distances[idx] = front_distances[j]

        # Generate offspring
        for i in range(0, POP_SIZE, 2):
            p1 = tournament_selection(population, ranks, distances)
            p2 = tournament_selection(population, ranks, distances)
            c1, c2 = simulated_binary_crossover(p1, p2)
            c1 = polynomial_mutation(c1, bounds)
            c2 = polynomial_mutation(c2, bounds)
            offspring_population[i] = enforce_bounds(c1, bounds)
            if i + 1 < POP_SIZE:
                offspring_population[i+1] = enforce_bounds(c2, bounds)

        # --- Combine and Select ---
        combined_population = np.vstack((population, offspring_population))
        combined_obj_values = np.array([calculate_objectives(ind) for ind in combined_population])
        combined_fronts = non_dominated_sort(combined_obj_values)

        new_population = np.zeros((POP_SIZE, NUM_VARIABLES))
        pop_fill_count = 0
        for front in combined_fronts:
            if pop_fill_count + len(front) <= POP_SIZE:
                new_population[pop_fill_count:pop_fill_count+len(front)] = combined_population[front]
                pop_fill_count += len(front)
            else:
                distances = calculate_crowding_distance(front, combined_obj_values)
                sorted_front_indices = [x for _, x in sorted(zip(distances, front), reverse=True)]
                remaining_space = POP_SIZE - pop_fill_count
                new_population[pop_fill_count:] = combined_population[sorted_front_indices[:remaining_space]]
                break
        population = new_population

    print("\n\n✅ Optimization Finished!")
    
    # --- Extract Final Pareto Front ---
    final_obj_values = np.array([calculate_objectives(ind) for ind in population])
    final_fronts = non_dominated_sort(final_obj_values)
    pareto_front_indices = final_fronts[0]
    
    return population[pareto_front_indices], final_obj_values[pareto_front_indices]


# --- 5. Results Display and Reporting ---

def create_report_df(solution_df):
    """Helper to create a DataFrame in the required report format."""
    ts, rl, eng, profit = get_all_metrics(solution_df)
    
    report = pd.DataFrame({
        'Product ID': df['Product ID'],
        'Price': solution_df[:NUM_PRODUCTS],
        'Discount': solution_df[NUM_PRODUCTS:],
        'Elasticity': df['Base Demand'], # Label as Elasticity per instruction
        'Total Sales': ts,
        'Revenue Loss': rl,
        'Engagement': eng
    })
    total_profit = np.sum(profit)
    return report, total_profit
    

def display_results(solutions, objectives):
    """
    Analyzes the Pareto front, generates the output.md report,
    and plots the Pareto front.
    """
    print("📈 Analyzing results and generating report...")
    
    # --- 1. Select a "Balanced" Solution from the Pareto Front ---
    sales = -objectives[:, 0]
    rev_loss = objectives[:, 1]
    engagement = -objectives[:, 2]

    norm_sales = (sales - sales.min()) / (sales.max() - sales.min() + 1e-6)
    norm_loss = (rev_loss - rev_loss.min()) / (rev_loss.max() - rev_loss.min() + 1e-6)
    norm_eng = (engagement - engagement.min()) / (engagement.max() - engagement.min() + 1e-6)
    
    distances_to_utopia = np.sqrt((norm_sales - 1)**2 + (norm_loss - 0)**2 + (norm_eng - 1)**2)
    balanced_idx = np.argmin(distances_to_utopia)
    balanced_solution = solutions[balanced_idx]

    # --- 2. Prepare DataFrames for the Markdown Report ---
    initial_solution = np.concatenate([df['Price'].values, df['Discount'].values])
    initial_report_df, initial_total_profit = create_report_df(initial_solution)
    
    final_report_df, final_total_profit = create_report_df(balanced_solution)
    
    # --- 3. Write to Markdown File ---
    output_filename = "output.md"
    # ***** FIX IS HERE: Added encoding='utf-8' *****
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("# Initial Products\n\n")
        f.write(initial_report_df.to_markdown(index=False, floatfmt=",.2f"))
        f.write(f"\n\n**Total Overall Profit:** ₹ {initial_total_profit:,.2f}\n\n")
        
        f.write("---\n\n")
        
        f.write("# Final Optimized Products\n\n")
        f.write(final_report_df.to_markdown(index=False, floatfmt=",.2f"))
        f.write(f"\n\n**Total Overall Profit:** ₹ {final_total_profit:,.2f}\n")

    print(f"✅ Successfully generated report: '{output_filename}'")
    
    # --- 4. Plot the Pareto Front ---
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    sc = ax.scatter(sales, rev_loss, engagement, c=rev_loss, cmap='viridis', s=60, label='Pareto Optimal Solutions')
    
    ax.scatter(sales[balanced_idx], rev_loss[balanced_idx], engagement[balanced_idx], 
               c='red', s=200, marker='*', label='Chosen Balanced Solution', depthshade=False)
    
    ax.set_xlabel('Maximize: Total Sales (₹)')
    ax.set_ylabel('Minimize: Total Revenue Loss (₹)')
    ax.set_zlabel('Maximize: Total Engagement')
    ax.set_title('Pareto Front of Optimal Pricing Strategies')
    ax.legend()
    
    fig.colorbar(sc, label='Total Revenue Loss (₹)')
    plt.show()


# --- 6. Main Execution ---
if __name__ == "__main__":
    pareto_solutions, pareto_objectives = run_nsga2()
    display_results(pareto_solutions, pareto_objectives)