import matplotlib.pyplot as plt
import matplotlib.lines as lines
from matplotlib.ticker import FuncFormatter
import numpy as np

'''
Plot Error rate difference
'''
def plotErrorRateDiff(err_diff, comm, policyName, W, S):
    fig, ax1 = plt.subplots()
    ax1.grid(True)
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: "{:g}".format(x)))
    ax1.plot(range(1,len(err_diff)+1), err_diff, fillstyle='bottom')

    # instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx() 
    ax2.tick_params(axis="y", labelcolor="xkcd:red orange")
    # ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, p: "{:g}".format(x)))
    ax2.plot(range(0,len(comm)), comm, fillstyle='bottom', color="xkcd:red orange")

    plt.xlim(left=0)
    plt.ylim(bottom=0)

    plt.xlabel("Window index")
    ax1.set_ylabel("Error rate difference, |e-e'|", color="b")
    ax2.set_ylabel('Communication rate', color="xkcd:red orange")
    plt.title("Absolute error difference for SUV sensor ["+S+\
        "], w="+str(W)+",\nusing Linear Regression and "+policyName)

    plt.tight_layout()

    plt.savefig('results/dataset_2_lin_reg/'+policyName+'/abs_err_diff_'+S+'_w_'+str(W)+'.png')

    plt.close(fig)
'''
Plot histogram of |e-e'|
'''
def plotHistErr(err_diff, policyName, W, S, SIZE):
    fig, ax = plt.subplots()

    n, bins, patches = ax.hist(err_diff, density=True, color='xkcd:azure',bins=(SIZE-100-W)//3, edgecolor='black')

    median = np.median(err_diff)

    props = dict(boxstyle='round', facecolor='white')
    for i in range(1,len(patches)):
        if patches[i-1].xy[0]<=median and median<patches[i].xy[0]:
            patches[i-1].set_color('xkcd:banana')
            patches[i-1].set_edgecolor('black')
            patches[i-1].set_hatch('/')
            ax.text(patches[i-1].xy[0]+patches[i-1].get_width()/2,n[i-1],"median:\n{:g}".format(median),va='center', color='r', bbox=props)
    if patches[i].xy[0]<=median:
        patches[i].set_color('xkcd:banana')
        patches[i].set_edgecolor('black')
        patches[i].set_hatch('/')
        ax.text(patches[i].xy[0]+patches[i].get_width()/2,n[i-1],"median:\n{:g}".format(median),va='center', color='r', bbox=props)

    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: "{:g}".format(x)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: "{:g}".format(x)))

    plt.xlabel("Error rate difference, |e-e'|")
    plt.ylabel("Frequency")
    plt.title("Absolute error difference for SUV sensor ["+S\
    +"], w="+str(W)+",\nusing Linear Regression and "+policyName+\
    "\nand the corresponding median for the data")

    fig.tight_layout()

    plt.savefig('results/dataset_2_lin_reg/'+policyName+'/hist_data_dist_median_'+S+'_w_'+str(W)+'.png')

    plt.close(fig)

'''
Plot all error rates
'''
def plotErrRate(err_storage, init_err, policyName, W, S):
    fig, ax = plt.subplots()

    plt.plot(range(0,len(err_storage)), err_storage[0:], fillstyle='bottom')

    ax.hlines(init_err,0,len(err_storage)-1,colors='r')
    props = dict(boxstyle='round', facecolor='white')
    ax.text(len(err_storage)-0.5,init_err,"{:g}".format(init_err),va='center', color='r', bbox=props)
    ax.grid(True)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: "{:g}".format(x)))

    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.xlabel("Window index")
    plt.ylabel("Error rate, e")
    plt.title("Error rate increase/decrease compared\n"+\
        "to initial error rate for SUV sensor ["+S+\
        "], w="+str(W)+",\nusing Linear Regression and"+policyName)

    plt.tight_layout()

    plt.savefig('results/dataset_2_lin_reg/'+policyName+'/err_rates_'+S+'_w_'+str(W)+'.png')

    plt.close(fig)

    '''
Plot waiting time
'''
def plotHistForWaitingTime(t, B, policyName, W, S, SIZE):
    fig, ax = plt.subplots()

    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: "{:g}".format(x)))

    plt.xlabel("Waiting time, t")
    plt.ylabel("Frequency")
    plt.title("Waiting time until sending an up-to-date model for SUV sensor ["+S\
        +"], w="+str(W)+",\nusing Linear Regression and "+policyName)
    bins = [i for i in range(max(t))]
    n, bins, patches = ax.hist(t, bins=bins, density=False, color="green", edgecolor='black')

    fig.tight_layout()

    if B == -1:
        plt.savefig('results/dataset_2_lin_reg/'+policyName+'/hist_waitingtime/hist_waiting_'+S+'_w_'+str(W)+'.png')
    else:
        plt.savefig('results/dataset_2_lin_reg/'+policyName+'/hist_waitingtime/hist_waiting_'+S+'_w_'+str(W)+'_B_'+str(B)+'.png')    

    plt.close(fig)


'''
Plot waiting times in a multi-boxplot
'''

def plotBoxPlotsForWaitingTime(t, B, W, S):

    fig, ax  = plt.subplots()
    plt.xlabel("OST penalty, B")

    plt.title("Waiting time until sending an up-to-date model for SUV sensor ["+S\
        +"], w="+str(W)+",\nusing Liear Regression and policyOST")
    plt.xticks([x for x in range(len(B))], B)
    ax.boxplot(t)

    # fig.tight_layout()
    
    plt.savefig('results/dataset_2_lin_reg/policyOST'+\
        '/boxplot_waiting_'+S+'_w_'+str(W)+'.png')
    plt.close(fig)