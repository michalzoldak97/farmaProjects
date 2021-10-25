import sys

num = list(map(int, sys.stdin.readline().split()))
some_nums = [1, 9, 4, 7, 6, 2]

print(num)
print(some_nums)

some_nums.sort()
num.sort()

print(num)
print(some_nums)

for i in num[:-1]:
    print(i)
    print("\n")

print("\n")
