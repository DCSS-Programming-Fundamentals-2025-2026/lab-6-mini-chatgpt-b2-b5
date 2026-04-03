using Contracts;
using Lib.Batching;
using Lib.Batching.Configuration;
using Lib.Models.TinyNN;
using Lib.Training.Configuration;
using Lib.Training.Metrics;
using Lib.Training.Scheduling;
using System.Text.Json;

namespace Lib.Training;

public class TrainingLoopImpl 
{
    public TrainingMetrics TrainTinyNN(ILanguageModel model, IBatchProvider batchProvider, TrainingConfig config, BatchConfig batchConfig)
    {
        if (model is not TinyNNModel tinyNNModel)
            throw new InvalidCastException("Invalid model");

        TrainingMetrics metrics = new TrainingMetrics();
        Random rng = new Random();

        int totalSteps = 0;
        for (int i = 0; i < config.Epochs; i++)
        {
            float sumLoss = 0f;
            int counter = 0;
            DateTime startTime = DateTime.Now;

            Batch batch = batchProvider.GetBatch(batchConfig, rng);
            int[][] inputs = batch.Inputs;
            int[] targets = batch.Targets;

            for (int j = 0; j < inputs.Length; j++)
            {
                int[] context = inputs[j];
                int target = targets[j];

                float loss = tinyNNModel.TrainStep(context, target, config.LearningRate);
                sumLoss += loss;
                counter++;
            }

            totalSteps += counter;

            if (CheckpointScheduler.ScheduleCheck(i + 1, config.CheckpointInterval, config.Epochs))
            {
                DateTime finishTime = DateTime.Now;
                TimeSpan delta = finishTime - startTime;

                float averageLoss = 0f;
                if (counter != 0)
                {
                    averageLoss = sumLoss / counter;
                }

                metrics.UpdateTinyNN(i + 1, averageLoss, totalSteps, delta);
            }
        }

        return metrics;
    }

    public TrainingMetrics TrainNGram(ILanguageModel model, int[] tokens, TrainingConfig config)
    {
        int n;
        INGramModel nGramModel;

        if (model.ModelKind == "bigram" && model is NGramModel bigramModel)
        {
            nGramModel = bigramModel;
            n = 2;
        }
        else if (model.ModelKind == "trigram" && model is TrigramModel trigramModel)
        {
            nGramModel = trigramModel;
            n = 3;
        }
        else
        {
            throw new InvalidCastException("Invalid model");
        }

        TrainingMetrics metrics = new TrainingMetrics();

        for (int i = 0; i < config.Epochs; i++)
        {
            if (i == 0 || i == config.Epochs - 1)
            {
                float perplexity = 0f;
                int nGramCount = 0;

                DateTime startTime = DateTime.Now;

                nGramModel.Train(tokens);

                DateTime finishTime = DateTime.Now;
                TimeSpan delta = finishTime - startTime;

                PerplexityCalculator calculator = new PerplexityCalculator();

                if (n == 2)
                {
                    perplexity = calculator.ComputePerplexityBigram((NGramModel)nGramModel, tokens);
                }
                else
                {
                    perplexity = calculator.ComputePerplexityTrigram((TrigramModel)nGramModel, tokens);
                }

                nGramCount = tokens.Length - n + 1;

                metrics.UpdateNGram(i + 1, perplexity, nGramCount, delta);

                if (CheckpointScheduler.ScheduleCheck(i + 1, config.CheckpointInterval, config.Epochs))
                {
                    var jsonElement = model.GetPayloadForCheckpoint();
                    string json = JsonSerializer.Serialize(jsonElement, new JsonSerializerOptions { WriteIndented = true });

                    File.WriteAllText("NGramCheckpoints.json", json);
                }
            }
        }

        return metrics;
    }
}