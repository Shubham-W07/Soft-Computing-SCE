# Generation 2 Analysis

## Step 1: Created Offspring Population Q(t)
Created 10 new offspring through crossover and mutation.

| Offspring ID | Cost (INR) | Thick (mm) | Battery (hr) | Camera (MP) | CPU (GHz) |
|---|---|---|---|---|---|
| Q1 | 10436.09 | 5.35 | 4.94 | 14.33 | 4.24 |
| Q2 | 6734.70 | 6.51 | 13.26 | 20.86 | 2.39 |
| Q3 | 10290.54 | 9.91 | 25.96 | 18.45 | 2.84 |
| Q4 | 12638.21 | 9.84 | 25.80 | 81.65 | 2.78 |
| Q5 | 10654.34 | 7.09 | 12.74 | 20.19 | 3.80 |
| Q6 | 14156.37 | 9.62 | 21.98 | 20.46 | 4.25 |
| Q7 | 11348.26 | 8.35 | 19.38 | 65.63 | 3.02 |
| Q8 | 15580.62 | 9.38 | 21.49 | 80.25 | 4.02 |
| Q9 | 8213.07 | 5.20 | 7.93 | 82.64 | 2.42 |
| Q10 | 10303.55 | 5.82 | 7.10 | 13.02 | 4.10 |

## Steps 2-4: Combine, Rank, and Calculate Diversity
Combined parents and offspring into a super-population of 10. This group is then sorted by rank and crowding distance.

| Phone ID | Cost (INR) | Thick (mm) | Battery (hr) | Camera (MP) | CPU (GHz) | Rank | Crowding Dist |
|---|---|---|---|---|---|---|---|
| R1 | 15580.62 | 9.38 | 21.49 | 80.25 | 4.02 | 0 | inf |
| R2 | 6684.78 | 6.51 | 13.31 | 20.86 | 2.37 | 0 | inf |
| R3 | 14156.37 | 9.62 | 21.98 | 20.46 | 4.25 | 0 | inf |
| R4 | 10326.99 | 7.09 | 12.74 | 12.01 | 3.80 | 0 | inf |
| R5 | 8759.71 | 5.20 | 7.93 | 96.30 | 2.42 | 0 | inf |
| R6 | 10436.09 | 5.35 | 4.94 | 14.33 | 4.24 | 0 | inf |
| R7 | 10290.54 | 9.91 | 25.96 | 18.45 | 2.84 | 0 | inf |
| R8 | 11382.60 | 7.69 | 16.68 | 77.78 | 3.05 | 0 | 0.2231 |
| R9 | 8429.40 | 5.93 | 10.00 | 50.09 | 2.86 | 0 | 0.1736 |
| R10 | 9006.92 | 5.70 | 9.82 | 88.87 | 2.50 | 0 | 0.1445 |

## Step 5: Select Next Generation's Parents P(t+1)
The top 10 individuals from the table above are selected to form the parent population for the next generation.
