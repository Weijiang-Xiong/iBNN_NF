import os
import matplotlib.pyplot as plt 

def plot_multiple_results(result_list, anno_list, fig_dir=".", save_name=None, figsize=(15, 8)):
    fig = plt.figure(figsize=figsize)
    for result in result_list:
        loss_list, ll_list, kl_list, acc_list, ece_list = result 
        plt.subplot(2,3,1)
        plt.plot(loss_list)
        plt.title("Negative ELBO")
        plt.legend(anno_list)
        plt.subplot(2,3,2)
        plt.plot(ll_list)
        plt.title("Log Likelihood")
        plt.legend(anno_list)
        plt.subplot(2,3,3)
        plt.plot(kl_list)
        plt.title("KL Divergence")
        plt.legend(anno_list)
        plt.subplot(2,3,4)
        plt.plot(acc_list)
        plt.title("Test Accuracy")
        plt.legend(anno_list)
        plt.subplot(2,3,5)
        plt.plot(ece_list)
        plt.title("ECE on testset")
        plt.legend(anno_list)

    # plt.show()
    fig.tight_layout()
    if os.path.exists(fig_dir) and save_name!=None:
        fig.savefig(fig_dir + "/" + "{}.jpg".format(save_name))
        
def plot_results(results, anno="", fig_dir=".", figsize=(15,8)):
    loss_list, ll_list, kl_list, acc_list, ece_list = results 
    fig = plt.figure(figsize=figsize)
    plt.subplot(2,3,1)
    plt.plot(loss_list)
    plt.title("Negative ELBO")
    plt.subplot(2,3,2)
    plt.plot(ll_list)
    plt.title("Log Likelihood")
    plt.subplot(2,3,3)
    plt.plot(kl_list)
    plt.title("KL Divergence")
    plt.subplot(2,3,4)
    plt.plot(acc_list)
    plt.title("Test Accuracy")
    plt.subplot(2,3,5)
    plt.plot(ece_list)
    plt.title("ECE on testset")
    # plt.show()
    fig.tight_layout()
    fig.savefig(fig_dir + "/" + "{}.jpg".format(anno))