# Generation 1 Analysis

## Step 1: Created Offspring Population Q(t)
Created 10 new offspring through crossover and mutation.

| Offspring ID | Cost (INR) | Thick (mm) | Battery (hr) | Camera (MP) | CPU (GHz) |
|---|---|---|---|---|---|
| Q1 | 10326.99 | 7.09 | 12.74 | 12.01 | 3.80 |
| Q2 | 6684.78 | 6.51 | 13.31 | 20.86 | 2.37 |
| Q3 | 8429.40 | 5.93 | 10.00 | 50.09 | 2.86 |
| Q4 | 11382.60 | 7.69 | 16.68 | 77.78 | 3.05 |
| Q5 | 10843.17 | 6.02 | 7.70 | 14.63 | 4.20 |
| Q6 | 11411.82 | 8.35 | 19.32 | 65.63 | 3.04 |
| Q7 | 9006.92 | 5.70 | 9.82 | 88.87 | 2.50 |
| Q8 | 12510.85 | 9.84 | 25.80 | 78.46 | 2.78 |
| Q9 | 13891.36 | 9.38 | 21.05 | 20.16 | 4.24 |
| Q10 | 8759.71 | 5.20 | 7.93 | 96.30 | 2.42 |

## Steps 2-4: Combine, Rank, and Calculate Diversity
Combined parents and offspring into a super-population of 10. This group is then sorted by rank and crowding distance.

| Phone ID | Cost (INR) | Thick (mm) | Battery (hr) | Camera (MP) | CPU (GHz) | Rank | Crowding Dist |
|---|---|---|---|---|---|---|---|
| R1 | 10326.99 | 7.09 | 12.74 | 12.01 | 3.80 | 0 | inf |
| R2 | 6684.78 | 6.51 | 13.31 | 20.86 | 2.37 | 0 | inf |
| R3 | 8429.40 | 5.93 | 10.00 | 50.09 | 2.86 | 0 | 0.2377 |
| R4 | 11382.60 | 7.69 | 16.68 | 77.78 | 3.05 | 0 | 0.2482 |
| R5 | 10843.17 | 6.02 | 7.70 | 14.63 | 4.20 | 0 | inf |
| R6 | 11411.82 | 8.35 | 19.32 | 65.63 | 3.04 | 0 | 0.2377 |
| R7 | 9006.92 | 5.70 | 9.82 | 88.87 | 2.50 | 0 | 0.1787 |
| R8 | 12510.85 | 9.84 | 25.80 | 78.46 | 2.78 | 0 | inf |
| R9 | 13891.36 | 9.38 | 21.05 | 20.16 | 4.24 | 0 | inf |
| R10 | 8759.71 | 5.20 | 7.93 | 96.30 | 2.42 | 0 | inf |

## Step 5: Select Next Generation's Parents P(t+1)
The top 10 individuals from the table above are selected to form the parent population for the next generation.
