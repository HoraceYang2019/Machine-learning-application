# https://towardsdatascience.com/linear-programming-optimization-foundations-2f12770f66ca
import pulp

# Create the LP problem
lp_problem = pulp.LpProblem("Apples and Candy Bars", pulp.LpMaximize)

# Define the decision variables
apples = pulp.LpVariable('apples', lowBound=0)
candy_bars = pulp.LpVariable('candy_bars', lowBound=0)

# Add the objective function - level of enjoyment from specific diet
lp_problem += 1 * apples + 3 * candy_bars

# Add calorie constraint
lp_problem += 95 * apples + 215 * candy_bars <= 850
# Add sugar constraint
lp_problem += 15 * apples + 20 * candy_bars <= 100

# Solve the problem
lp_problem.solve()

# Print the results
print(f"Status: {pulp.LpStatus[lp_problem.status]}")
print(f"apples = {pulp.value(apples)}")
print(f"candy bars = {pulp.value(candy_bars)}")
print(f"Maximum enjoyment = {pulp.value(lp_problem.objective)}")