using Lib.MathCore;
using Lib.Models.TinyTransformer.State;

namespace Lib.Models.TinyTransformer.Layers
{
    public class SelfAttentionLayer
    {
        public float[][] Compute(float[][] x, TinyTransformerWeights weights, int d)
        {
            int n = x.Length;
            float[][] Q = Multiply(x, weights.Wq, d);
            float[][] K = Multiply(x, weights.Wk, d);
            float[][] V = Multiply(x, weights.Wv, d);

            float[][] scores = new float[n][];
            float scale = (float)Math.Sqrt(d);

            for (int i = 0; i < n; i++)
            {
                scores[i] = new float[n];
                for (int j = 0; j < n; j++)
                {
                    if (j > i)
                    {
                        scores[i][j] = float.NegativeInfinity;
                        continue;
                    }

                    float dot = 0;
                    for (int k = 0; k < d; k++)
                    {
                        dot += Q[i][k] * K[j][k];
                    }
                    scores[i][j] = dot / scale;
                }
            }

            float[][] output = new float[n][];
            for (int i = 0; i < n; i++)
            {
                float[] attentionWeights = MathOps.Default.Softmax(scores[i]);
                output[i] = new float[d];
                for (int j = 0; j <= i; j++)
                {
                    for (int k = 0; k < d; k++)
                    {
                        output[i][k] += attentionWeights[j] * V[j][k];
                    }
                }
            }

            return Multiply(output, weights.Wo, d);
        }

        private float[][] Multiply(float[][] input, float[,] matrix, int d)
        {
            int n = input.Length;
            float[][] result = new float[n][];
            for (int i = 0; i < n; i++)
            {
                result[i] = new float[d];
                for (int j = 0; j < d; j++)
                {
                    for (int k = 0; k < d; k++)
                    {
                        result[i][j] += input[i][k] * matrix[k, j];
                    }
                }
            }
            return result;
        }
    }
}
