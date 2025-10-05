# Generation 5 Analysis

## Step 1: Created Offspring Population Q(t)
Created 10 new offspring through crossover and mutation.

| Offspring ID | Cost (INR) | Thick (mm) | Battery (hr) | Camera (MP) | CPU (GHz) |
|---|---|---|---|---|---|
| Q1 | 10307.11 | 7.06 | 12.64 | 12.03 | 3.80 |
| Q2 | 8795.55 | 5.20 | 7.93 | 97.20 | 2.42 |
| Q3 | 5728.80 | 5.89 | 11.19 | 17.17 | 2.19 |
| Q4 | 12522.36 | 6.49 | 9.04 | 24.86 | 4.45 |
| Q5 | 13690.61 | 9.97 | 25.98 | 96.30 | 2.94 |
| Q6 | 7031.70 | 5.82 | 9.61 | 18.72 | 2.84 |
| Q7 | 16880.33 | 9.62 | 21.85 | 82.46 | 4.33 |
| Q8 | 10457.99 | 7.09 | 12.78 | 15.45 | 3.80 |
| Q9 | 5178.50 | 5.00 | 7.31 | 14.33 | 2.35 |
| Q10 | 9366.41 | 7.69 | 17.91 | 61.04 | 2.43 |

## Steps 2-4: Combine, Rank, and Calculate Diversity
Combined parents and offspring into a super-population of 10. This group is then sorted by rank and crowding distance.

| Phone ID | Cost (INR) | Thick (mm) | Battery (hr) | Camera (MP) | CPU (GHz) | Rank | Crowding Dist |
|---|---|---|---|---|---|---|---|
| R1 | 12576.15 | 9.97 | 27.01 | 96.30 | 2.42 | 0 | inf |
| R2 | 5728.80 | 5.89 | 11.19 | 17.17 | 2.19 | 0 | inf |
| R3 | 8795.55 | 5.20 | 7.93 | 97.20 | 2.42 | 0 | inf |
| R4 | 5178.50 | 5.00 | 7.31 | 14.33 | 2.35 | 0 | inf |
| R5 | 10326.99 | 7.09 | 12.74 | 12.01 | 3.80 | 0 | inf |
| R6 | 13690.61 | 9.97 | 25.98 | 96.30 | 2.94 | 0 | inf |
| R7 | 10934.80 | 5.00 | 3.08 | 14.33 | 4.46 | 0 | inf |
| R8 | 16880.33 | 9.62 | 21.85 | 82.46 | 4.33 | 0 | inf |
| R9 | 9366.41 | 7.69 | 17.91 | 61.04 | 2.43 | 0 | 0.2673 |
| R10 | 11382.60 | 7.69 | 16.68 | 77.78 | 3.05 | 0 | 0.1735 |

## Step 5: Select Next Generation's Parents P(t+1)
The top 10 individuals from the table above are selected to form the parent population for the next generation.
