using System;
using Lib.Batching.Configuration;
using Lib.Batching.Sampling;

namespace Lib.Batching;

public class TokenBatchProvider : IBatchProvider
{
    private readonly ITokenStream _stream;
    private readonly BatchWindowSampler _sampler;

    public TokenBatchProvider(ITokenStream stream, BatchWindowSampler sampler)
    {
        _stream = stream;
        _sampler = sampler;
    }

    public Batch GetBatch(BatchConfig config, Random rng)
    {
        var tokens = _stream.GetTokens();
        
        var startIndices = _sampler.GetRandomStartIndices(tokens.Length, config.BatchSize, config.BlockSize, rng);

        int[][] inputs = new int[config.BatchSize][];
        int[] targets = new int[config.BatchSize];

        for (int i = 0; i < config.BatchSize; i++)
        {
            int startIndex = startIndices[i];
            
            inputs[i] = new int[config.BlockSize];
            Array.Copy(tokens, startIndex, inputs[i], 0, config.BlockSize);
            
            targets[i] = tokens[startIndex + config.BlockSize];
        }

        return new Batch(inputs, targets);
    }
}