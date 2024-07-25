import numpy as np

# tabe baraye barresi halt haye consistent va inconsistent

def count_zeros(number):
    #  return str(number).count('0.')
    t = 0
    for u in number:
        if u == float(0):
            t += 1
    return t


# voroodi gereftan

n = int(input())

matrix = []
for i in range(n):
    row = list(map(float, input().split()))
    matrix.append(row)

b = list(map(float, input().split()))

# sakht matrix augmented

augmented_matrix = np.hstack((matrix, np.array(b).reshape(-1, 1)))
print(augmented_matrix)
print()

# forward-phase

i = 0
while i < n:
    if augmented_matrix[i][i] != 0:
        factor = augmented_matrix[i][i]
        for k in range(i, n + 1):
            # 1 kardane zarib moteghayer ha dar har khat
            augmented_matrix[i][k] = augmented_matrix[i][k] / factor
        print(augmented_matrix)
        print()
        for j in range(i + 1, n):
            factor2 = augmented_matrix[j][i]
            for r in range(i, n + 1):
                # 0 kardane zir zarib motaghaier
                augmented_matrix[j][r] -= augmented_matrix[i][r] * factor2
            print(augmented_matrix)
            print()
        i += 1
    else:
        if i < n - 1:
            temp = i + 1
            counter = 0
            while augmented_matrix[temp][i] == 0 and counter != n - 1 - i:
                counter += 1
                if temp < n - 1:
                    temp += 1
            if counter != n - 1 - i:
                # swap row
                augmented_matrix[[i, temp]] = augmented_matrix[[temp, i]]
                print(augmented_matrix)
                print()
            else:
                i += 1
                print(augmented_matrix)
                print()
        else:
            i += 1
            print(augmented_matrix)
            print()

# backward-phase

for ii in range(n - 2, -1, -1):
    m = n - 1
    for jj in range(n - 1, ii, -1):
        fac = augmented_matrix[ii][jj]
        augmented_matrix[ii][n] -= fac * augmented_matrix[jj][n]
        # agar deraye ghotr asli marboote 1 nabood balayee ra 0 nakonad
        if augmented_matrix[m][m] == 1:
            augmented_matrix[ii][jj] = 0
        m -= 1
    print(augmented_matrix)
    print()

inconsistent = 0
consistent = 0
for rr in range(n):
    if count_zeros(augmented_matrix[rr]) == n and augmented_matrix[rr][n] != 0:
        inconsistent = 1
    if count_zeros(augmented_matrix[rr]) == n + 1:
        consistent = 1

if inconsistent == 1:
    print(" dastgah javab nadarad ")

if inconsistent == 0 and consistent == 1:
    print(" dastgah binahayat javab darad ")

if inconsistent == 0 and consistent == 0:
    print(" dastgah tak javab be shekl zir darad ")
    print(augmented_matrix[:, n])
