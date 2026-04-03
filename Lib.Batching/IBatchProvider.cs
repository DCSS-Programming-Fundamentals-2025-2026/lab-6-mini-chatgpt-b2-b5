using System;
using Lib.Batching.Configuration;

namespace Lib.Batching;

public interface IBatchProvider
{
    Batch GetBatch(BatchConfig config, Random rng);
}