import pandas as pd
import matplotlib.pyplot as plt
def read(path:str ):
    '''path: path to csv history from transformer training'''

    pd.set_option('display.max_columns', None)
    history = pd.read_csv(path)
    
    history = history[history['total_loss_te'].notna()]
    plt.plot(history.lr)
    plt.plot(history.valid.fillna(0)/50)
    plt.show()
    print(history)

read(path='')