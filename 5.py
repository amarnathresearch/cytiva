a = 20
def modify_value(x):
    try:
        global a
        a = x + a
        return a
    except Exception as e:
        return str(f"Catch exception {e}")
    
def change(x):
    
    a = x + 2
    return a

print("Modified value:", modify_value(15))
print("Global a value:", a)

print("Change function output:", change(10))