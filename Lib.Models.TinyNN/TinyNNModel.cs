using Contracts;
using Lib.MathCore;
using Lib.Models.TinyNN.Configuration;
using Lib.Models.TinyNN.Layers;
using Lib.Models.TinyNN.State;

namespace Lib.Models.TinyNN;

public class TinyNNModel : ILanguageModel
{
    public string ModelKind { get; set; }
    public int VocabSize { get; set; }
    public TinyNNConfig Config { get; set; }
    public TinyNNWeights Weights { get; set; }
    public EmbeddingLayer Embedding { get; set; }
    public LinearHead Linear { get; set; }

    public TinyNNModel(string modelKind, int vocabSize, TinyNNConfig config, TinyNNWeights weights)
    {
        ModelKind = modelKind;
        VocabSize = vocabSize;
        Config = config;
        Weights = weights;

        Embedding = new EmbeddingLayer(VocabSize, Config, Weights);
        Linear = new LinearHead(VocabSize, Config, Weights);
    }

    public object GetPayloadForCheckpoint()
    {
        return ToPayload();
    }
    
    public TinyNNPayload ToPayload()
    {
        return new TinyNNPayload
        {
            Config = this.Config,
            Weights = this.Weights
        };
    }

    public float[] NextTokenScores(ReadOnlySpan<int> context)
    {
        int[] tokens = context.ToArray();
        if (tokens.Length == 0)
        {
            throw new ArgumentException("Context is empty.");
        }

        float[] hidden = Embedding.EncodeContext(tokens);   

        float[] logits = Linear.Project(hidden);
        for (int i = 0; i < logits.Length; i++)
        {
            if (float.IsNaN(logits[i])) throw new ArgumentException("Logits is NaN.");
            if (float.IsInfinity(logits[i])) throw new ArgumentException("Logits is infinity.");
        }

        return logits;
    }

    public float TrainStep(ReadOnlySpan<int> context, int target, float lr)
    {
        int[] tokens = context.ToArray();
        if (tokens.Length == 0)
        {
            throw new ArgumentException("Context is empty.");
        }
        else if (target < 0 || VocabSize <= target)
        {                                                                    
            throw new ArgumentOutOfRangeException("Uncorrect taget value.");
        }
        else if (lr <= 0)
        {
            throw new ArgumentException("lr don't valid");
        }

        MathOpsImpl mathOpsImpl = new MathOpsImpl();
        float[] hidden = Embedding.EncodeContext(tokens);

        float[] logits = NextTokenScores(context);
        float[] probs = mathOpsImpl.Softmax(logits);

        float probsTarget = probs[target];
        float loss = (float)Math.Log(probsTarget);

        float[] dLogits = CalculateGradient(probs, target);

        float[] dHidden = Linear.Backward(hidden, dLogits, lr);
        Embedding.Backward(context, dHidden, lr);

        return loss * -1;
    }

    public float[] CalculateGradient(float[] probs, int target)
    {
        probs[target]--;
        return probs;
    }
}