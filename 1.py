print("hello world")
def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    else:
        series = [0, 1]
        for i in range(2, n):
            series.append(series[i-1] + series[i-2])
        return series

# Example usage
n = 10
print(f"Fibonacci series up to {n} terms: {fibonacci(n)}")