import numpy as np

def bootstrap_replicate_1d(data, func, random_state=np.random):
    return func(random_state.choice(data, size=len(data)))

def draw_bs_reps(data, func, size=1, random_state=np.random):
    """Draw bootstrap replicates."""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func, random_state=random_state)

    return bs_replicates



####more generic and it gets pairs (below funcs)####
def bootstrap_replicate(func, random_state=np.random, *datas):
    assert len(datas) > 0
    data_len = len(datas[0])
    assert np.all([len(data) == data_len for data in datas[1:]])
    
    indices = random_state.randint(low=0, high=data_len, size=data_len)
    datas = map(lambda dd : np.array(dd), datas)
    resampled_data = [data[indices] for data in datas]
    #print len(resampled_data)
    #for resample in resampled_data:
    #    print resample.shape
    return func(*resampled_data)

def draw_bootstrap_replicates(func, size=1, random_state=np.random, *datas):
    """Draw bootstrap replicates."""

    # Generate replicates    
    return [
            bootstrap_replicate(func, random_state, *datas) for ii in range(size)
        ]
