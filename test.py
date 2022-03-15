import numpy as np
print(5333//10000)
arr = np.arange(0,10)
brr= np.arange(20,30)
print(type(arr))
print(brr)
print(np.concatenate([arr,brr]))
list = [1,2,3]
lenList = len(list)
print(lenList)
flag = 0
import numpy as np

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
index = [2, 3, 6]

new_a = np.delete(a, 10)

print(new_a)  # Prints `[1, 2, 5, 6, 8, 9]`