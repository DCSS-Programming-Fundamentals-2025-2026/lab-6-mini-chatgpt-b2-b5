using System;

namespace Lib.MathCore;

public static class SoftmaxCalculator
{
    public static float[] Softmax(ReadOnlySpan<float> logits)
    {
        if (logits.Length == 0)
            return Array.Empty<float>();

        float max = logits[0];
        for (int i = 1; i < logits.Length; i++)
        {
            if (logits[i] > max)
                max = logits[i];
        }

        float[] result = new float[logits.Length];
        float sum = 0f;

        for (int i = 0; i < logits.Length; i++)
        {
            float exp = MathF.Exp(logits[i] - max);
            result[i] = exp;
            sum += exp;
        }

        for (int i = 0; i < result.Length; i++)
        {
            result[i] /= sum;
        }

        return result;
    }
}