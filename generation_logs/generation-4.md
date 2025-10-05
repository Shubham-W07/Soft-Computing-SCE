# Generation 4 Analysis

## Step 1: Created Offspring Population Q(t)
Created 10 new offspring through crossover and mutation.

| Offspring ID | Cost (INR) | Thick (mm) | Battery (hr) | Camera (MP) | CPU (GHz) |
|---|---|---|---|---|---|
| Q1 | 10227.75 | 9.88 | 25.83 | 17.53 | 2.84 |
| Q2 | 11624.56 | 9.91 | 25.81 | 47.39 | 2.92 |
| Q3 | 12576.15 | 9.97 | 27.01 | 96.30 | 2.42 |
| Q4 | 6844.56 | 6.51 | 13.31 | 24.86 | 2.37 |
| Q5 | 15281.17 | 9.91 | 24.09 | 81.36 | 3.77 |
| Q6 | 7363.84 | 5.93 | 10.00 | 23.45 | 2.86 |
| Q7 | 6462.64 | 5.12 | 6.82 | 18.45 | 2.84 |
| Q8 | 8929.22 | 5.41 | 8.78 | 96.30 | 2.42 |
| Q9 | 10565.26 | 7.69 | 16.80 | 61.04 | 2.99 |
| Q10 | 16065.35 | 9.88 | 23.47 | 82.46 | 4.02 |

## Steps 2-4: Combine, Rank, and Calculate Diversity
Combined parents and offspring into a super-population of 10. This group is then sorted by rank and crowding distance.

| Phone ID | Cost (INR) | Thick (mm) | Battery (hr) | Camera (MP) | CPU (GHz) | Rank | Crowding Dist |
|---|---|---|---|---|---|---|---|
| R1 | 8759.71 | 5.20 | 7.93 | 96.30 | 2.42 | 0 | inf |
| R2 | 12576.15 | 9.97 | 27.01 | 96.30 | 2.42 | 0 | inf |
| R3 | 6844.56 | 6.51 | 13.31 | 24.86 | 2.37 | 0 | inf |
| R4 | 10326.99 | 7.09 | 12.74 | 12.01 | 3.80 | 0 | inf |
| R5 | 10934.80 | 5.00 | 3.08 | 14.33 | 4.46 | 0 | inf |
| R6 | 16065.35 | 9.88 | 23.47 | 82.46 | 4.02 | 0 | inf |
| R7 | 6462.64 | 5.12 | 6.82 | 18.45 | 2.84 | 0 | inf |
| R8 | 10565.26 | 7.69 | 16.80 | 61.04 | 2.99 | 0 | 0.1876 |
| R9 | 11382.60 | 7.69 | 16.68 | 77.78 | 3.05 | 0 | 0.1852 |
| R10 | 15281.17 | 9.91 | 24.09 | 81.36 | 3.77 | 0 | 0.1698 |

## Step 5: Select Next Generation's Parents P(t+1)
The top 10 individuals from the table above are selected to form the parent population for the next generation.
