import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
import matplotlib.pyplot as plt

# --- 1. Define the Custom Problem ---
class PhoneProblem(ElementwiseProblem):

    def __init__(self):
        super().__init__(n_var=3,
                         n_obj=5,
                         n_constr=0,
                         xl=np.array([3000, 2.0, 12]),
                         xu=np.array([8000, 4.5, 108]))

    def _evaluate(self, x, out, *args, **kwargs):
        cost = (x[0] * 0.01) + (x[1] ** 2 * 5) + (x[2] * 0.5)
        thickness = 2 + (x[0] / 1000)
        
        # --- NEW, MORE REALISTIC BATTERY FORMULA ---
        # This formula ensures battery life remains positive and scales better.
        battery_life = (x[0] / 250) - (x[1] * 2) 
        
        camera_quality = x[2]
        cpu_speed = x[1]
        
        out["F"] = [cost, thickness, -battery_life, -camera_quality, -cpu_speed]

# --- 2. Initialize the Algorithm ---
problem = PhoneProblem()
algorithm = NSGA2(
    pop_size=100,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)

# --- 3. Run the Optimization ---
res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               verbose=True)

# --- 4. Process and Visualize the MOGA Result (Pareto Front) ---
pareto_front = res.F
pareto_front[:, 2:] *= -1 # Convert maximization objectives back

print("\n--- MOGA Result: Pareto-Optimal Solutions ---")
print("Cost ($) | Thickness (mm) | Battery (hr) | Camera (MP) | CPU (GHz)")
for i in range(min(10, len(pareto_front))):
    print(f"{pareto_front[i, 0]:.2f}     | {pareto_front[i, 1]:.2f}           | {pareto_front[i, 2]:.2f}         | {pareto_front[i, 3]:.2f}        | {pareto_front[i, 4]:.2f}")

# --- 5. Find the Best "Compromise" Solution ---
# Normalize the objectives. We want to find the point closest to the "ideal"
# The ideal point for us is [min, min, max, max, max]
# In normalized space, this is [0, 0, 1, 1, 1] for our objectives
norm_pareto = (pareto_front - pareto_front.min(axis=0)) / (pareto_front.max(axis=0) - pareto_front.min(axis=0))

# The ideal point in normalized space
# For MIN objectives, the ideal is 0. For MAX objectives, the ideal is 1.
ideal_point = np.array([0, 0, 1, 1, 1])

# Calculate Euclidean distance from the ideal point for each solution
distances = np.linalg.norm(norm_pareto - ideal_point, axis=1)

# Find the index of the solution with the minimum distance
best_index = np.argmin(distances)
best_compromise_solution = pareto_front[best_index]

print("\n--- Best Compromise Solution ---")
print("This solution is the most balanced trade-off among all objectives.")
print(f"Cost ($):         {best_compromise_solution[0]:.2f}")
print(f"Thickness (mm):   {best_compromise_solution[1]:.2f}")
print(f"Battery Life (hr):{best_compromise_solution[2]:.2f}")
print(f"Camera (MP):      {best_compromise_solution[3]:.2f}")
print(f"CPU Speed (GHz):  {best_compromise_solution[4]:.2f}")

# --- 6. Plotting ---
plt.figure(figsize=(8, 6))
plt.scatter(pareto_front[:, 0], pareto_front[:, 2], s=40, facecolors='none', edgecolors='blue')
# Highlight the best compromise solution on the plot
plt.scatter(best_compromise_solution[0], best_compromise_solution[2], s=100, facecolors='none', edgecolors='green', linewidth=2, label='Best Compromise')
plt.title("Trade-Off: Cost vs. Battery Life")
plt.xlabel("Production Cost ($)")
plt.ylabel("Battery Life (hours)")
plt.legend()
plt.grid(True)
plt.savefig("moga_cost_vs_battery_compromise.png")
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(pareto_front[:, 1], pareto_front[:, 4], s=40, facecolors='none', edgecolors='red')
# Highlight the best compromise solution on the plot
plt.scatter(best_compromise_solution[1], best_compromise_solution[4], s=100, facecolors='none', edgecolors='green', linewidth=2, label='Best Compromise')
plt.title("Trade-Off: Thickness vs. CPU Speed")
plt.xlabel("Thickness (mm)")
plt.ylabel("CPU Speed (GHz)")
plt.legend()
plt.grid(True)
plt.savefig("moga_thickness_vs_cpu_compromise.png")
plt.show()