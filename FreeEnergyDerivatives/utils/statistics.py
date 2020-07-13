import numpy as np


def block_statistics(observations, maxBlockSize=0):
    
    nObs = len(observations)
    
    minBlockSize = 1
    
    if (maxBlockSize == 0):
        maxBlockSize = np.int(nObs / 4)
    
    numBlocks = maxBlockSize - minBlockSize
    
    blockMean = np.zeros(numBlocks)
    blockStd = np.zeros(numBlocks)
    
    blockCtr = 0
    
    mean = np.mean(observations)
    
    for blockSize in range(minBlockSize, maxBlockSize):
        nBlock = np.int(nObs / blockSize)
        
        obsParcel = np.zeros(nBlock)
        
        for i in range(1, nBlock + 1):
            
            ibeg = (i - 1) * blockSize
            iend = ibeg + blockSize
            obsParcel[i - 1] = np.mean(observations[ibeg:iend])
        
        blockMean[blockCtr] = np.mean(obsParcel)
        blockStd[blockCtr] = (np.mean(obsParcel ** 2) - mean ** 2)
        blockCtr += 1
        
    return blockMean, blockStd

    
if __name__ == "__main__":
    
    mean = 10.0
    stddev = 0.1
    
    N = 10000
    
    observations = np.random.normal(loc=mean, scale=stddev, size=N)
    
    bMeans, bVar = block_statistics(observations, maxBlockSize=10)
    
    print (np.mean(bMeans), np.mean(bVar))
    
