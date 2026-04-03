using System;

namespace Lib.MathCore;

public static class LossCalculator
{
    public static float CrossEntropyLoss(ReadOnlySpan<float> logits, int target)
    {
        if (logits.Length == 0)
            return 0f;

        var probs = SoftmaxCalculator.Softmax(logits);

        float p = probs[target];

        if (p <= 0f)
            p = 1e-9f;

        return -MathF.Log(p);
    }
}