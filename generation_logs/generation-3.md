# Generation 3 Analysis

## Step 1: Created Offspring Population Q(t)
Created 10 new offspring through crossover and mutation.

| Offspring ID | Cost (INR) | Thick (mm) | Battery (hr) | Camera (MP) | CPU (GHz) |
|---|---|---|---|---|---|
| Q1 | 10675.25 | 9.91 | 25.81 | 23.66 | 2.92 |
| Q2 | 10934.80 | 5.00 | 3.08 | 14.33 | 4.46 |
| Q3 | 13741.21 | 9.48 | 21.45 | 14.33 | 4.24 |
| Q4 | 8433.11 | 5.87 | 9.78 | 52.89 | 2.84 |
| Q5 | 15976.99 | 9.88 | 23.47 | 80.25 | 4.02 |
| Q6 | 15669.25 | 9.38 | 21.43 | 80.25 | 4.05 |
| Q7 | 11203.38 | 5.50 | 5.27 | 20.46 | 4.36 |
| Q8 | 6721.27 | 6.51 | 13.27 | 20.86 | 2.39 |
| Q9 | 13413.13 | 9.85 | 23.37 | 16.66 | 4.02 |
| Q10 | 11997.68 | 9.03 | 22.44 | 78.71 | 2.84 |

## Steps 2-4: Combine, Rank, and Calculate Diversity
Combined parents and offspring into a super-population of 10. This group is then sorted by rank and crowding distance.

| Phone ID | Cost (INR) | Thick (mm) | Battery (hr) | Camera (MP) | CPU (GHz) | Rank | Crowding Dist |
|---|---|---|---|---|---|---|---|
| R1 | 15976.99 | 9.88 | 23.47 | 80.25 | 4.02 | 0 | inf |
| R2 | 8759.71 | 5.20 | 7.93 | 96.30 | 2.42 | 0 | inf |
| R3 | 10675.25 | 9.91 | 25.81 | 23.66 | 2.92 | 0 | inf |
| R4 | 10934.80 | 5.00 | 3.08 | 14.33 | 4.46 | 0 | inf |
| R5 | 10326.99 | 7.09 | 12.74 | 12.01 | 3.80 | 0 | inf |
| R6 | 6684.78 | 6.51 | 13.31 | 20.86 | 2.37 | 0 | inf |
| R7 | 10290.54 | 9.91 | 25.96 | 18.45 | 2.84 | 0 | inf |
| R8 | 11382.60 | 7.69 | 16.68 | 77.78 | 3.05 | 0 | 0.3131 |
| R9 | 8429.40 | 5.93 | 10.00 | 50.09 | 2.86 | 0 | 0.1654 |
| R10 | 11997.68 | 9.03 | 22.44 | 78.71 | 2.84 | 0 | 0.1633 |

## Step 5: Select Next Generation's Parents P(t+1)
The top 10 individuals from the table above are selected to form the parent population for the next generation.
