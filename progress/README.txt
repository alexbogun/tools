Small script to display the progress of a long computation

Usage:

import progress as pr

pr.start("Long computation", end=""
n = 1000
for i in range(n):
    pr.tick(i=i, n=n, end="")
    #some code
pr.finish()
