namespace Lib.MathCore.Utilities;

public static class ScoreUtilities
{
    public static int ArgMax(ReadOnlySpan<float> scores)
    {
        if (scores.IsEmpty)
        {
            throw new ArgumentException("Масив не може бути порожнім", nameof(scores));
        }

        int maxIndex = 0;
        float maxValue = scores[0];

        for (int i = 1; i < scores.Length; i++)
        {
            if (scores[i] > maxValue)
            {
                maxValue = scores[i];
                maxIndex = i;
            }
        }

        return maxIndex;
    }
}