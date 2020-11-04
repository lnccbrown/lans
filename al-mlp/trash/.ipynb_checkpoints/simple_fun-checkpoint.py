

def f__(x = 10):
    for i in range(10):
        calculate_stupid(i)
        x += x
    return x

def calculate_stupid(i):
    for i in range(100):
        i += i
    return 

f__(x = 20)