using Lib.Models.TinyTransformer.State;

namespace Lib.Models.TinyTransformer.Layers
{
    public class FeedForwardLayer
    {
        public float[] Compute(float[] x, TinyTransformerWeights weights, int d)
        {
            int dff = 4 * d;
            float[] hidden = new float[dff];

            for (int j = 0; j < dff; j++)
            {
                float sum = weights.Ffn1Bias[j];
                for (int i = 0; i < d; i++)
                {
                    sum += x[i] * weights.Ffn1[i, j];
                }
                hidden[j] = Math.Max(0, sum);
            }

            float[] output = new float[d];
            for (int j = 0; j < d; j++)
            {
                float sum = weights.Ffn2Bias[j];
                for (int i = 0; i < dff; i++)
                {
                    sum += hidden[i] * weights.Ffn2[i, j];
                }
                output[j] = sum;
            }

            return output;
        }
    }
}
