using Contracts;
using Lib.Batching;
using Lib.Batching.Configuration;
using Lib.Training.Configuration;
using Lib.Training.Metrics;

namespace Lib.Training;

public class TrainingLoop : ITrainingLoop
{
    public TrainingMetrics Train (ILanguageModel model, IBatchProvider batchProvider, TrainingConfig config, BatchConfig batchConfig, int[] tokens)
    {
        if (model == null)
        {
            throw new ArgumentException("Invalid data");
        }

        TrainingLoopImpl loopImpl = new TrainingLoopImpl();
        if (model.ModelKind == "bigram" || model.ModelKind == "trigram")
        {
            return loopImpl.TrainNGram(model, tokens, config);
        }
        else if (model.ModelKind == "TinyNN")
        {
            return loopImpl.TrainTinyNN(model, batchProvider, config, batchConfig);
        }                        
        else if (model.ModelKind == "Transformer")
        {
            new TrainingMetrics();
        }

        throw new ArgumentException("Invalid data");
    }
}