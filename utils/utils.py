import numpy as np
import math

# Bucket Map
# 0-1 => 0
# 1-2 => 1
# 2-3 => 2
# 3-4 => 3
# 4-5 => 4

def extractBuckets(scores):
    return list(map(scores, lambda s: getBuckets(s)))

def getBuckets(score):
    buckets = np.zeros(5)
    bucketId = math.floor(score)
    buckets[bucketId] = 1
    return buckets
