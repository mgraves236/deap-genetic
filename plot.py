from matplotlib import pyplot as plt
import numpy as np

if __name__ == "__main__":

    plt.figure(1)
    x_arr = []
    y_arr = []
    x2_arr = []
    y2_arr = []
    with open('n=20opt.txt') as data:
        for line in data:
            row = line.split()
            if row:
                y_arr.append(float(row[1]))
                x_arr.append(float(row[0]))
    with open('n=2033.txt') as data:
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
    # plt.show()

    plt.figure(2)
    # creating the dataset
    # set width of bar
    barWidth = 0.2
    # fig = plt.subplots(figsize=(12, 8))

    # set height of bar
    k1 = [2548,	2517,	2517,	2660]
    k2 = [67594,	65377,	66631,	73417]
    k3 = [1023,	992,	992,	1135]
    k4 = [1023,	992,	992,	1135]

    # Set position of bar on X axis
    br1 = np.arange(len(k1))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    # Make the plot
    plt.bar(br1, k1, color='purple', width=barWidth,
            edgecolor='grey', label='makespan')
    plt.bar(br2, k2, color='blue', width=barWidth,
            edgecolor='grey', label='total flow time')
    plt.bar(br3, k3, color='orange', width=barWidth,
            edgecolor='grey', label='max tardiness')
    plt.bar(br4, k4, color='red', width=barWidth,
            edgecolor='grey', label='max lateness')

    # Adding Xticks
    plt.ylabel('Result', fontsize=15)
    plt.xticks([r + barWidth for r in range(len(k1))],
               ['rozw1', 'rozw2', 'rozw3', 'rozw4'])
    plt.legend()
    plt.show()