using System;

namespace Lib.MathCore;

public interface IMathOps
{
    float[] Softmax(ReadOnlySpan<float> logits);

    float CrossEntropyLoss(ReadOnlySpan<float> logits, int target);

    int ArgMax(ReadOnlySpan<float> scores);

    int SampleFromProbs(ReadOnlySpan<float> probs, Random rng);
}