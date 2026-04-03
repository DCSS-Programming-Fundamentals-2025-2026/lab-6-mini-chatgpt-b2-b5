using Contracts;
using Lib.Batching;
using Lib.Batching.Configuration;
using Lib.Training.Configuration;
using Lib.Training.Metrics;

namespace Lib.Training;

public interface ITrainingLoop
{
    TrainingMetrics Train(ILanguageModel model, IBatchProvider batchProvider, TrainingConfig config, BatchConfig batchConfig, int[] tokens);
}