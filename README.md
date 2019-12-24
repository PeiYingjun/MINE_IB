# MINE_IB
An implementation of information bottleneck with mutual information neural estimation(MINE)

Notice that I didn't make a moving average between network parameters every traning steps, and the test error is worse than that in the paper. Also since the original paper misses lots of technical details(I also failed to find the source code), I really suspect that even with the moving average trick, the performance will still be worse than that the paper have claimed.  Yet at least, results show that a model with MINE as a regularizer performs better than the vanila one. I tested it in the MNIST, the test error of MINE_IB is 1.39% while the error of baseline is 1.43%.  
To run MINE_IB, just run this code in the shell `python model.py --algo='MINE_IB'`  
To run the baseline, excute this command: `python model.py --algo='regular'`  
If you get better results with(or without) this code, feel free to contact me.


