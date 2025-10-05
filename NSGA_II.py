import numpy as np
import os
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.callback import Callback
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize
import matplotlib.pyplot as plt

# --- A. Define the Custom Problem ---
class PhoneProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=3,
                         n_obj=5,
                         n_constr=0,
                         xl=np.array([3000, 2.0, 12]),
                         xu=np.array([8000, 4.5, 108]))

    def _evaluate(self, x, out, *args, **kwargs):
        # Decision Variables
        battery_size = x[0]
        cpu_speed_ghz = x[1]
        camera_mp = x[2]

        # Objective 1: Minimize Cost (in Rupees, assuming 1 USD = 80 INR)
        cost_usd = (battery_size * 0.01) + (cpu_speed_ghz ** 2 * 5) + (camera_mp * 0.5)
        cost_inr = cost_usd * 80

        # Objective 2: Minimize Thickness
        thickness = 2 + (battery_size / 1000)

        # Objectives to MAXIMIZE are returned as negative
        battery_life = (battery_size / 250) - (cpu_speed_ghz * 2)
        camera_quality = camera_mp
        cpu_speed_obj = cpu_speed_ghz
        
        out["F"] = [cost_inr, thickness, -battery_life, -camera_quality, -cpu_speed_obj]

# --- B. Create a Custom Callback to Log Each Generation ---
class GenerationLogCallback(Callback):
    def __init__(self):
        super().__init__()
        self.log_dir = "generation_logs"
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def notify(self, algorithm):
        gen = algorithm.n_gen
        pop_size = algorithm.pop_size
        
        # Open the markdown file for the current generation
        with open(os.path.join(self.log_dir, f"generation-{gen}.md"), "w") as f:
            f.write(f"# Generation {gen} Analysis\n\n")

            # --- Step 1: Offspring Population Q(t) ---
            offspring = algorithm.off
            f_offspring = offspring.get("F")
            f_offspring[:, 2:] *= -1
            f.write("## Step 1: Created Offspring Population Q(t)\n")
            f.write(f"Created {len(offspring)} new offspring through crossover and mutation.\n\n")
            f.write(f"| Offspring ID | Cost (INR) | Thick (mm) | Battery (hr) | Camera (MP) | CPU (GHz) |\n")
            f.write(f"|---|---|---|---|---|---|\n")
            for i in range(len(f_offspring)):
                fo = f_offspring[i]
                f.write(f"| Q{i+1} | {fo[0]:.2f} | {fo[1]:.2f} | {fo[2]:.2f} | {fo[3]:.2f} | {fo[4]:.2f} |\n")
            
            # --- Steps 2, 3, 4: Combined Population, Ranking, and Crowding Distance ---
            # The algorithm's 'pop' object at this stage is the combined and sorted population
            combined_pop = algorithm.pop
            f_combined = combined_pop.get("F")
            f_combined[:, 2:] *= -1
            ranks = combined_pop.get("rank")
            crowding = combined_pop.get("crowding")
            
            f.write("\n## Steps 2-4: Combine, Rank, and Calculate Diversity\n")
            f.write(f"Combined parents and offspring into a super-population of {len(combined_pop)}. This group is then sorted by rank and crowding distance.\n\n")
            f.write(f"| Phone ID | Cost (INR) | Thick (mm) | Battery (hr) | Camera (MP) | CPU (GHz) | Rank | Crowding Dist |\n")
            f.write(f"|---|---|---|---|---|---|---|---|\n")
            for i in range(len(combined_pop)):
                fc = f_combined[i]
                f.write(f"| R{i+1} | {fc[0]:.2f} | {fc[1]:.2f} | {fc[2]:.2f} | {fc[3]:.2f} | {fc[4]:.2f} | {ranks[i]} | {crowding[i]:.4f} |\n")

            # --- Step 5: Select Next Generation's Parents P(t+1) ---
            # The top `pop_size` individuals from the sorted list above become the next parents
            f.write("\n## Step 5: Select Next Generation's Parents P(t+1)\n")
            f.write(f"The top {pop_size} individuals from the table above are selected to form the parent population for the next generation.\n")

# --- C. Initialize and Run the Algorithm ---
problem = PhoneProblem()
log_callback = GenerationLogCallback()

algorithm = NSGA2(
    pop_size=10, # Using a small population for a clear example
    n_offsprings=10,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(eta=20),
    eliminate_duplicates=True
)

# Stopping Condition: Fixed number of generations
res = minimize(problem,
               algorithm,
               ('n_gen', 5), # Using a small number of generations for a quick demo
               seed=1,
               callback=log_callback,
               verbose=False)

# --- D. Write Final Pareto Front and Best Trade-off to File ---
pareto_f = res.F
pareto_f[:, 2:] *= -1

with open("pareto_front.md", "w") as f:
    f.write("# Final Pareto Front Solutions\n\n")
    f.write(f"| Phone ID | Cost (INR) | Thick (mm) | Battery (hr) | Camera (MP) | CPU (GHz) |\n")
    f.write(f"|---|---|---|---|---|---|\n")
    for i in range(len(pareto_f)):
        pf = pareto_f[i]
        f.write(f"| P{i+1} | {pf[0]:.2f} | {pf[1]:.2f} | {pf[2]:.2f} | {pf[3]:.2f} | {pf[4]:.2f} |\n")

    # Find and write the best compromise solution
    norm_pareto = (pareto_f - pareto_f.min(axis=0)) / (pareto_f.max(axis=0) - pareto_f.min(axis=0))
    ideal_point = np.array([0, 0, 1, 1, 1])
    distances = np.linalg.norm(norm_pareto - ideal_point, axis=1)
    best_index = np.argmin(distances)
    best_compromise_solution = pareto_f[best_index]

    f.write("\n# Best Compromise Solution\n\n")
    f.write("This solution is the most balanced trade-off among all objectives.\n\n")
    f.write(f"* **Phone ID**: P{best_index+1}\n")
    f.write(f"* **Cost (INR)**: {best_compromise_solution[0]:.2f}\n")
    f.write(f"* **Thickness (mm)**: {best_compromise_solution[1]:.2f}\n")
    f.write(f"* **Battery Life (hr)**: {best_compromise_solution[2]:.2f}\n")
    f.write(f"* **Camera (MP)**: {best_compromise_solution[3]:.2f}\n")
    f.write(f"* **CPU Speed (GHz)**: {best_compromise_solution[4]:.2f}\n")

print("Optimization complete. Check the 'generation_logs' folder and 'pareto_front.md' file.")

# --- E. Generate Relevant Graphs ---
plt.figure(figsize=(8, 6))
plt.scatter(pareto_f[:, 0], pareto_f[:, 2], s=40, facecolors='none', edgecolors='blue')
plt.scatter(best_compromise_solution[0], best_compromise_solution[2], s=100, facecolors='none', edgecolors='green', linewidth=2, label=f'Best Compromise (P{best_index+1})')
plt.title("Trade-Off: Cost vs. Battery Life")
plt.xlabel("Cost (INR)")
plt.ylabel("Battery Life (hours)")
plt.legend()
plt.grid(True)
plt.savefig("final_tradeoff_plot.png")
plt.show()