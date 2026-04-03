using System;
using Lib.MathCore.Utilities;

namespace Lib.MathCore;

public class MathOpsImpl : IMathOps
{
    public float[] Softmax(ReadOnlySpan<float> logits)
    {
        return SoftmaxCalculator.Softmax(logits);
    }

    public float CrossEntropyLoss(ReadOnlySpan<float> logits, int target)
    {
        return LossCalculator.CrossEntropyLoss(logits, target);
    }

    public int ArgMax(ReadOnlySpan<float> scores)
    {
        return ScoreUtilities.ArgMax(scores);
    }

    public int SampleFromProbs(ReadOnlySpan<float> probs, Random rng)
    {
        return ProbabilitySampler.SampleFromProbs(probs, rng);
    }
}