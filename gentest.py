
def g(x):
    if x < 5:
        return 0
    if x >= 5:
        step = 1
        while True:
            yield step
            step += 1

for i in range(10):
    print(next(g(i)))
