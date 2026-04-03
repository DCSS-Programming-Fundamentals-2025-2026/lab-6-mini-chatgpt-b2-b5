using System;

namespace Lib.Batching.Sampling;

public class BatchWindowSampler
{
    public int[] GetRandomStartIndices(int totalTokens, int batchSize, int blockSize, Random rng)
    {
        int maxStartIndex = totalTokens - blockSize - 1;
        
        if (maxStartIndex < 0)
        {
            throw new InvalidOperationException("Not enough tokens to form a single batch window.");
        }

        int[] indices = new int[batchSize];
        for (int i = 0; i < batchSize; i++)
        {
            indices[i] = rng.Next(0, maxStartIndex + 1);
        }
        return indices;
    }
}