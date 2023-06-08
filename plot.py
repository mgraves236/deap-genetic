from matplotlib import pyplot as plt

if __name__ == "__main__":

    plt.figure(1)
    x_arr = []
    y_arr = []
    x2_arr = []
    y2_arr = []
    with open('n=100opt.txt') as data:
        for line in data:
            row = line.split()
            if row:
                y_arr.append(float(row[1]))
                x_arr.append(float(row[0]))
    with open('n=5033.txt') as data:
        for line in data:
            row = line.split()
            if row:
                y2_arr.append(float(row[1]))
                x2_arr.append(float(row[0]))
    plt.title("Result vs. Iterations n = 100")
    plt.xlabel("iterations")
    plt.ylabel("result")
    y_arr = y_arr[2:]
    x_arr = x_arr[2:]
    y2_arr = y_arr[2:]
    x2_arr = x_arr[2:]
    maximum = max(y_arr)
    y_arr[:] = [x / maximum for x in y_arr]
    maximum = max(y2_arr)
    y2_arr[:] = [x / maximum for x in y2_arr]
    plt.plot(x_arr, y_arr, color='purple', label ="opt")
    plt.plot(x2_arr, y2_arr, color='blue', label ="c1 = c2 = c3 = 0.33")
    plt.legend()
    plt.show()

