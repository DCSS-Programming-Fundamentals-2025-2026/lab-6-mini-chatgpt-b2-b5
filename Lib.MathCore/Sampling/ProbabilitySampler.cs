using System;

namespace Lib.MathCore;

public static class ProbabilitySampler
{
    public static int SampleFromProbs(ReadOnlySpan<float> probs, Random rng)
    {
        if (probs.Length == 0)
            return 0;

        float r = (float)rng.NextDouble();

        float sum = 0f;

        for (int i = 0; i < probs.Length; i++)
        {
            sum += probs[i];

            if (r < sum)
                return i;
        }

        return probs.Length - 1;
    }
}