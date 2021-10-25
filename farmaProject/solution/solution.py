import fileinput

nums = []

for f in fileinput.input():
    nums.append(int(f))

nums.sort()

for n in nums:
    print(n)
