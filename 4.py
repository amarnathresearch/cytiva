try:
    a = 0
    b = 0
    result = a / b
    print("The result is:", result)
except Exception as e:
    print("An unexpected error occurred:", str(e))

finally:
    print("Execution completed.")

