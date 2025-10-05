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
        battery_life = (x[0] / 250) - (x[1] * 2)
        camera_quality = x[2]
        cpu_speed = x[1]
        out["F"] = [cost, thickness, -battery_life, -camera_quality, -cpu_speed]

# --- 2. Initialize the Algorithm ---
problem = PhoneProblem()
algorithm = NSGA2(
    pop_size=100,
    n_offsprings=10,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)

# --- 3. Run the Optimization (with save_history=True) ---
res = minimize(problem,
               algorithm,
               ('n_gen', 200),
               seed=1,
               save_history=True,
               verbose=True)

# --- 4. Process and Display the Detailed Results Table ---
# Get the final population from the history
final_pop = res.history[-1].pop

# Filter for the Pareto-optimal solutions (Rank 0)
pareto_solutions = final_pop[final_pop.get("rank") == 0]

# Get their objective values, ranks, and crowding distances
pareto_f = pareto_solutions.get("F")
pareto_f[:, 2:] *= -1 # Convert maximization objectives back
pareto_rank = pareto_solutions.get("rank")
pareto_crowding = pareto_solutions.get("crowding")

print("\n--- Detailed Pareto Front Solutions ---")
print(f"{'Cost ($)':<10} | {'Thick (mm)':<12} | {'Battery (hr)':<14} | {'Camera (MP)':<13} | {'CPU (GHz)':<11} | {'Rank':<6} | {'Crowding Dist':<15}")
print("-" * 110)

for i in range(len(pareto_solutions)):
    f = pareto_f[i]
    r = pareto_rank[i]
    c = pareto_crowding[i]
    print(f"{f[0]:<10.2f} | {f[1]:<12.2f} | {f[2]:<14.2f} | {f[3]:<13.2f} | {f[4]:<11.2f} | {r:<6} | {c:<15.4f}")


# --- 5. Find the Best "Compromise" Solution ---
# We use the objective values `pareto_f` calculated above
norm_pareto = (pareto_f - pareto_f.min(axis=0)) / (pareto_f.max(axis=0) - pareto_f.min(axis=0))
ideal_point = np.array([0, 0, 1, 1, 1])
distances = np.linalg.norm(norm_pareto - ideal_point, axis=1)
best_index = np.argmin(distances)
best_compromise_solution = pareto_f[best_index]

print("\n--- Best Compromise Solution ---")
print("This solution is the most balanced trade-off among all objectives.")
print(f"Cost ($):         {best_compromise_solution[0]:.2f}")
print(f"Thickness (mm):   {best_compromise_solution[1]:.2f}")
print(f"Battery Life (hr):{best_compromise_solution[2]:.2f}")
print(f"Camera (MP):      {best_compromise_solution[3]:.2f}")
print(f"CPU Speed (GHz):  {best_compromise_solution[4]:.2f}")


# --- 6. Plotting ---
# The plotting code remains the same as before...
plt.figure(figsize=(8, 6))
plt.scatter(pareto_f[:, 0], pareto_f[:, 2], s=40, facecolors='none', edgecolors='blue')
plt.scatter(best_compromise_solution[0], best_compromise_solution[2], s=100, facecolors='none', edgecolors='green', linewidth=2, label='Best Compromise')
plt.title("Trade-Off: Cost vs. Battery Life")
plt.xlabel("Production Cost ($)")
plt.ylabel("Battery Life (hours)")
plt.legend()
plt.grid(True)
plt.savefig("nsga2_cost_vs_battery_compromise.png")
plt.show()