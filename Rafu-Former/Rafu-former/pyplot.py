import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

def plot_seq_feature(pred_, true_, history_,label = "train",error = False,input='',wv=''):
    assert(pred_.shape == true_.shape)

    index = -1
    if pred_.shape[2]>800:
        index = 840
    pred = pred_.detach().clone()[..., index].unsqueeze(2)
    true = true_.detach().clone()[..., index].unsqueeze(2)
    history = history_.detach().clone()[..., index].unsqueeze(2)

    if len(pred.shape) == 3:  #BLD
        if error == False:
            pred = pred[0]
            true = true[0]
            history = history[0]
        else:
            largest_loss = 0
            largest_index = 0
            criterion = nn.MSELoss()
            for i in range(pred.shape[0]):
                loss = criterion(pred[i],true[i])
                if  loss > largest_loss:
                    largest_loss = loss
                    largest_index = i
            pred = pred[largest_index]
            true = true[largest_index]
            history = history[largest_index]
            input_error = input[largest_index]
            # wv_error = wv[largest_index]
            # print('input mean',input_error.mean())
            # print('input std',input_error.std())
            # print('out mean',true.mean())
            # print('out std',true.std())
            # print('wv mean',wv_error.mean())
            # print('wv std',wv_error.std())
            # print('end')

    pred = pred.cpu().numpy()
    true = true.cpu().numpy()
    history = history.cpu().numpy()

    L, D = pred.shape
    L_h,D_h = history.shape
    # if D == 1:
    #     pic_row, pic_col = 1, 1
    # else:
    #     pic_col = 2
    #     pic_row = math.ceil(D/pic_col)
    pic_row, pic_col = D, 1


    fig = plt.figure(figsize=(8*pic_row,8*pic_col))
    for i in range(1):
        ax = plt.subplot(pic_row,pic_col,i+1)
        ax.plot(np.arange(L_h), history[:, i], label = "history")
        ax.plot(np.arange(L_h,L_h+L), pred[:, i], label = "pred")
        ax.plot(np.arange(L_h,L_h+L), true[:, i], label = "true")
        ax.set_title("dimension = {},  ".format(i) + label)
        ax.legend()

    return fig


def plot_basis(bases, weights, seq_len, pred_len):
    """
    修正后的基可视化函数
    Args:
        bases: [seq_len, 3] 原始基分量
        weights: [3] 该变量的权重
        seq_len: 历史序列长度
        pred_len: 预测长度
    """
    assert bases.shape[1] == 3  # 确保是趋势/季节/残差3分量
    
    fig, ax = plt.subplots(3, 1, figsize=(12, 10))
    time_steps = np.arange(seq_len) - (seq_len - pred_len)  # 时间坐标对齐
    components = ['Trend', 'Seasonal', 'Residual']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    for i in range(3):
        # 原始分量
        ax[i].plot(time_steps, bases[:, i], 
                 color=colors[i], linestyle='-', 
                 label=f'Raw {components[i]}')
        
        # 加权分量
        weighted = bases[:, i] * weights[i]
        ax[i].plot(time_steps, weighted,
                 color=colors[i], linestyle='--',
                 label=f'Weighted (w={weights[i]:.5f})')
        
        ax[i].axvline(x=0, color='r', linestyle=':', alpha=0.5)
        ax[i].set_ylabel(components[i])
        ax[i].legend()
        ax[i].grid(True, linestyle=':', alpha=0.5)
    
    ax[-1].set_xlabel('Time Steps (0 = Current Moment)')
    plt.suptitle('Basis Components Decomposition')
    plt.tight_layout()
    return fig


def plot_weighted_components(bases, weights, sample_idx=0, var_idx=0):
    """
    bases: [B, T, V, 3] 原始基分量
    weights: [B, V, 3] 各分量权重
    """
    fig, ax = plt.subplots(3, 1, figsize=(12, 9))
    time_steps = np.arange(bases.shape[1])
    components = ['Trend', 'Seasonal', 'Residual']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    raw_bases = bases[sample_idx, :, var_idx].cpu().numpy()  # [T,3]
    comp_weights = weights[sample_idx, var_idx].cpu().numpy() # [3]
    
    for i in range(3):
        # 原始分量与加权分量对比
        ax[i].plot(time_steps, raw_bases[:, i], 
                 color=colors[i], linestyle='-',
                 label=f'Raw {components[i]}')
        ax[i].plot(time_steps, raw_bases[:, i] * comp_weights[i],
                 color=colors[i], linestyle='--', 
                 label=f'Weighted Component')  # 移除了权重数值显示
        
        ax[i].set_ylabel(f'{components[i]} Value')
        ax[i].legend()
        ax[i].grid(True, alpha=0.3)
    
    ax[-1].set_xlabel('Time Steps')
    plt.suptitle('Basis Components Before/After Weighting')
    plt.tight_layout()
    return fig

def plot_weight_heatmap(weights, sample_idx=0):
    """
    weights: [B, V, 3] 权重矩阵
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    weight_data = weights[sample_idx].cpu().numpy()  # [V,3]
    
    im = ax.imshow(weight_data.T, cmap='viridis', aspect='auto')
    ax.set_xticks(np.arange(weight_data.shape[0]))
    ax.set_yticks(np.arange(3))
    ax.set_yticklabels(['Trend', 'Seasonal', 'Residual'])
    ax.set_xlabel('Variable Index')
    plt.colorbar(im, label='Weight Value')
    plt.title('Basis Weights Across Variables')
    return fig