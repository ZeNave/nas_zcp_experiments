import torch
import sys
sys.path.append("/home/josenave/Desktop/nas_experiments")
import os
import time
import argparse
import random
import numpy as np
from my_code.class_dataset import MyDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='EPE-NAS')
parser.add_argument('--data_loc', default='./datasets/cifar', type=str, help='dataset folder')
parser.add_argument('--api_loc', default='./datasets/NAS-Bench-201-v1_0-e61699.pth',
                    type=str, help='path to API')
parser.add_argument('--save_loc', default='results_train', type=str, help='folder to save results')
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--evaluate_size', default=256, type=int)
parser.add_argument('--GPU', default='0', type=str)
parser.add_argument('--seed', default=1, type=int)
parser.add_argument('--trainval', action='store_true')
parser.add_argument('--dataset', default='cifar10', type=str)
parser.add_argument('--n_samples', default=100, type=int)
parser.add_argument('--n_runs', default=1, type=int)
parser.add_argument('--n_hidden_layers', default=3, type=int)
parser.add_argument('--size_of_hidden_layers', default=512, type=int)
parser.add_argument('--training_sets', default=28, type=int)
parser.add_argument('--number_of_datasets', default=16, type=int)
parser.add_argument('--convolution', default=0, type=int)
parser.add_argument('--pca', default=0, type=int)
parser.add_argument('--tuple', default=0, type=int)


args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(args.seed)
np.random.seed(args.seed)

medians = np.array([])
means = np.array([])
q1s = np.array([])
q3s = np.array([])
stds = np.array([])
vars = np.array([])
top_1_values = np.array([])
bottom_1_values = np.array([])
x_total = np.array([])

for i in range(args.training_sets):
    print("iiiii",i)
    dataset = torch.load(f'../mine_dataset/batch_{args.batch_size}/train_{i}_{args.batch_size}.pt')
    x_semi_total = np.array([])
    y_total = np.array([])
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    for batch, (X, y) in enumerate(dataloader):
        # print(f"X.shape: {X.shape}")
        # print(f"y.shape: {y.shape}")
        # print(f"X.shape: {X.shape[0]}")
        # print(f"y.shape: {y.shape[0]}")
        # print(f"X[0]: {X[0]}")
        # print(f"y[0]: {y[0]}")

        x_semi_total = np.append(x_semi_total, X)
        y_total = np.append(y_total, y)

        # Compute prediction and loss
        X = X.to(device).float()
        y = y.to(device).float()
        print(f"batch: {batch}")
        print(f"X.shape: {X.shape}")
        print(f"y.shape: {y.shape}")
        # pred = model(X)
        # pred = pred.to(device)
        # pred = pred.squeeze()
        # loss = loss_fn(pred, y)
        # loss = loss.to(device)
    #print(f"y[0]: {y[0]}")
    print(f"x_semi_total.shape: {x_semi_total.shape}")
    print(f"x_semi_total[:10]: {x_semi_total[:10]}\n")
    print(f"y_total.shape: {y_total.shape}")

    train_y = np.expand_dims(y_total, 1)


    sorted_x_total = np.argsort(x_semi_total)

    top_1_value = x_semi_total[sorted_x_total[-1]]
    bottom_1_value = x_semi_total[sorted_x_total[0]]

    top_10_values = x_semi_total[sorted_x_total[-10:]]
    bottom_10_values = x_semi_total[sorted_x_total[:10]]
    print(f"top 10 values: {top_10_values}\n")
    print(f"bottom 10 values: {bottom_10_values}\n")

    mean = np.mean(x_semi_total) 
    # print("\nMean: ", mean) 
    
    std = np.std(x_semi_total) 
    # print("\nstd: ", std) 
    
    var = np.var(x_semi_total) 
    # print("\nvariance: ", var) 

    median = np.median(x_semi_total)
    # print("\nmedian: ", median) 

    # threshold = 3
    # outliers = []
    # for x in x_semi_total:
    #     z_score = (x - mean) / std
    #     if abs(z_score) > threshold:
    #         outliers.append(x)

    # print("\nfor threshold = 3; z_score = (x - mean) / std; abs(z_score) > threshold; len(Outliers)  : ", len(outliers))
    # print("\nOutliers  : ", outliers)

    q1 = np.percentile(x_semi_total, 25)
    q3 = np.percentile(x_semi_total, 75)
    iqr = q3 - q1
    threshold = 1.5 * iqr
    # outliers = np.where((x_semi_total < q1 - threshold) | (x_semi_total > q3 + threshold))

    # print("\nfor < q1 - threshold; > q3 + threshold; iqr = q3 - q1; threshold = 1.5 * iqr; len(Outliers[0])  : ", len(outliers[0]))
    # print("\nOutliers  : ", outliers)

    # print(f"Q1: {q1}")
    # print(f"Q3: {q3}")

    # print(f"q1 - threshold: {q1 - threshold}")
    # print(f"q3 + threshold: {q3 + threshold}")

    medians = np.append(medians, median)
    means = np.append(means, mean)
    stds = np.append(stds, std)
    vars = np.append(vars, var)
    q1s = np.append(q1s, q1)
    q3s = np.append(q3s, q3)
    top_1_values = np.append(top_1_values, top_1_value)
    bottom_1_values = np.append(bottom_1_values, bottom_1_value)

    print("\nMeans: ", means) 
    print("\nstds: ", stds) 
    print("\nvariances: ", vars) 
    print("\nmedians: ", medians) 

    print(f"Q1s: {q1s}")
    print(f"Q3s: {q3s}")

    print(f"top_1_values: {top_1_values}")
    print(f"bottom_1_values: {bottom_1_values}")

    # x_total = np.append(x_total, x_semi_total)

    print(f"q1 - threshold: {q1 - threshold}; q3 + threshold: {q3 + threshold}")

    # plt.hist(x_semi_total, range=[q1 - threshold, q3 + threshold], bins=10)
    # plt.show() 

    # plt.hist(x_semi_total)
    # plt.show() 

    x_total_sub = x_semi_total.copy()

    for j in range(len(x_semi_total)):    
        if abs(x_semi_total[j] - median) > 3 * std:
            if abs(np.mean(x_semi_total[j - 2:j + 2])) > 3 * std:
                # print(f"x_semi_total[j]: {x_semi_total[j]}")
                # print(f" abs(x_semi_total[j] - median): { abs(x_semi_total[j] - median)}; 3 * std: {3 * std}")
                # print(f"here abs(np.mean(x_semi_total[j - 2:j + 2])): {abs(np.mean(x_semi_total[j - 2:j + 2]))}")
                x_total_sub[j] = median
            else:
                x_total_sub[j] =  abs(np.mean(x_semi_total[j - 2:j + 2]))
    
    sorted_x_total = np.argsort(x_total_sub)

    top_1_value = x_total_sub[sorted_x_total[-1]]
    bottom_1_value = x_total_sub[sorted_x_total[0]]

    top_10_values = x_total_sub[sorted_x_total[-10:]]
    bottom_10_values = x_total_sub[sorted_x_total[:10]]
    print(f"x_total_sub top 10 values: {top_10_values}\n")
    print(f"x_total_sub bottom 10 values: {bottom_10_values}\n") 

    plt.hist(x_total_sub)
    plt.show()
    
    x_total_sub = x_total_sub.reshape(250, 256, 3072)

    traindata = MyDataset(x_total_sub, train_y)
    torch.save(traindata, f'../mine_dataset/outliers_free/train_{i}_{args.batch_size}.pt')

    print(f'../mine_dataset/outliers_free/train_{i}_{args.batch_size}.pt')



x_total_sub = x_total.copy()

median_of_medians = np.median(medians)
print("\nmedian of medians: ", median_of_medians)

median_of_stds = np.median(stds)
print("\nmedian of stds: ", median_of_stds)

median_of_q1s = np.median(q1s)
print("\nmedian of q1s: ", median_of_q1s)

median_of_q3s = np.median(q3s)
print("\nmedian of q3s: ", median_of_q3s)

for i in range(len(x_total)):
    if abs(x_total[i] - median_of_medians) > 3 * median_of_stds:
        x_total_sub[i] = np.mean([x_total[i - 1], x_total[i + 1]])

iqr = median_of_q3s - median_of_q1s
threshold = 1.5 * iqr

plt.hist(x_total_sub, range=[median_of_q1s - threshold, median_of_q3s + threshold], bins=10)
plt.show() 

plt.hist(x_total_sub)
plt.show() 

print(f"x_total.shape: {x_total_sub.shape}")
print(f"x_total[:10]: {x_total_sub[:10]}\n")

sorted_x_total = np.argsort(x_total_sub)

top_1_value = x_total[sorted_x_total[-1]]
bottom_1_value = x_total[sorted_x_total[0]]

top_10_values = x_total[sorted_x_total[-10:]]
bottom_10_values = x_total[sorted_x_total[:10]]
print(f"top 10 values: {top_10_values}\n")
print(f"bottom 10 values: {bottom_10_values}\n")

mean = np.mean(x_total_sub) 
print("\nMean: ", mean) 

std = np.std(x_total_sub) 
print("\nstd: ", std) 

var = np.var(x_total_sub) 
print("\nvariance: ", var) 

median = np.median(x_total_sub)
print("\nmedian: ", median) 

# threshold = 3
# outliers = []
# for x in x_total_sub:
#     z_score = (x - mean) / std
#     if abs(z_score) > threshold:
#         outliers.append(x)

# print("\nfor threshold = 3; z_score = (x - mean) / std; abs(z_score) > threshold; len(Outliers)  : ", len(outliers))
# print("\nOutliers  : ", outliers)

# q1 = np.percentile(x_total_sub, 25)
# q3 = np.percentile(x_total_sub, 75)
# iqr = q3 - q1
# threshold = 1.5 * iqr
# outliers = np.where((x_total_sub < q1 - threshold) | (x_total_sub > q3 + threshold))

# print("\nfor < q1 - threshold; > q3 + threshold; iqr = q3 - q1; threshold = 1.5 * iqr; len(Outliers[0])  : ", len(outliers[0]))
# print("\nOutliers  : ", outliers)

# print(f"Q1: {q1}")
# print(f"Q3: {q3}")

# print(f"q1 - threshold: {q1 - threshold}")
# print(f"q3 + threshold: {q3 + threshold}")

# medians = np.append(medians, median)
# means = np.append(means, mean)
# stds = np.append(stds, std)
# vars = np.append(vars, var)
# q1s = np.append(q1s, q1)
# q3s = np.append(q3s, q3)
# top_1_values = np.append(top_1_values, top_1_value)
# bottom_1_values = np.append(bottom_1_values, bottom_1_value)