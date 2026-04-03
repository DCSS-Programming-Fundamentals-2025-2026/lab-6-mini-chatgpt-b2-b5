using Lib.Models.TinyTransformer.State;

namespace Lib.Models.TinyTransformer.Layers
{
    public class TransformerBlock
    {
        private readonly SelfAttentionLayer _attention = new();
        private readonly FeedForwardLayer _ffn = new();

        public float[][] Forward(float[][] x, TinyTransformerWeights weights, int d)
        {
            float[][] attnOut = _attention.Compute(x, weights, d);
            float[][] result = new float[x.Length][];

            for (int i = 0; i < x.Length; i++)
            {
                result[i] = _ffn.Compute(attnOut[i], weights, d);
            }

            return result;
        }
    }
}
