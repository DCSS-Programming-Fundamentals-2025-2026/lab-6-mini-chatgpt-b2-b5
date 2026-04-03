using System.Text.Json.Serialization;
using Lib.Models.TinyTransformer.Configuration;
using Lib.Models.TinyTransformer.Layers;
namespace Lib.Models.TinyTransformer.State
{
    public class TinyTransformerWeights
    {
        public float[,] TokenEmbeddings { get; set; } = null!;
        public float[,] Wq { get; set; } = null!;
        public float[,] Wk { get; set; } = null!;
        public float[,] Wv { get; set; } = null!;
        public float[,] Wo { get; set; } = null!;
        public float[,] Ffn1 { get; set; } = null!;
        public float[] Ffn1Bias { get; set; } = null!;
        public float[,] Ffn2 { get; set; } = null!;
        public float[] Ffn2Bias { get; set; } = null!;
        public float[,] OutputW { get; set; } = null!;
        public float[] OutputBias { get; set; } = null!;

        public TinyTransformerWeights()
        {
        }

        [JsonConstructor]
        public TinyTransformerWeights(float[][] tokenEmbeddings, float[][] wq, float[][] wk, float[][] wv, float[][] wo, 
                                    float[][] ffn1, float[] ffn1Bias, float[][] ffn2, float[] ffn2Bias, 
                                    float[][] outputW, float[] outputBias)
        {
            TokenEmbeddings = FromJaggedArray(tokenEmbeddings);
            Wq = FromJaggedArray(wq);
            Wk = FromJaggedArray(wk);
            Wv = FromJaggedArray(wv);
            Wo = FromJaggedArray(wo);
            Ffn1 = FromJaggedArray(ffn1);
            Ffn1Bias = ffn1Bias;
            Ffn2 = FromJaggedArray(ffn2);
            Ffn2Bias = ffn2Bias;
            OutputW = FromJaggedArray(outputW);
            OutputBias = outputBias;
        }
         

        public static TinyTransformerWeights Initialize(int vocabSize, int embeddingSize, Random? random = null)
        {
            if (random == null)
            {
                random = new Random();
            }
            int dff = 4 * embeddingSize;
            float scale = 0.02f;

            TinyTransformerWeights weights = new TinyTransformerWeights
            {
                TokenEmbeddings = InitMatrix(vocabSize, embeddingSize, scale, random),
                Wq = InitMatrix(embeddingSize, embeddingSize, scale, random),
                Wk = InitMatrix(embeddingSize, embeddingSize, scale, random),
                Wv = InitMatrix(embeddingSize, embeddingSize, scale, random),
                Wo = InitMatrix(embeddingSize, embeddingSize, scale, random),
                Ffn1 = InitMatrix(embeddingSize, dff, scale, random),
                Ffn1Bias = new float[dff],
                Ffn2 = InitMatrix(dff, embeddingSize, scale, random),
                Ffn2Bias = new float[embeddingSize],
                OutputW = InitMatrix(embeddingSize, vocabSize, scale, random),
                OutputBias = new float[vocabSize]
            };

            return weights;
        }

        private static float[,] InitMatrix(int rows, int cols, float scale, Random random)
        {
            float[,] matrix = new float[rows, cols];
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    matrix[i, j] = (float)(random.NextDouble() * 2 - 1) * scale;
                }
            }
            return matrix;
        }

        private static float[,] FromJaggedArray(float[][] jagged)
        {
            if (jagged == null || jagged.Length == 0) return new float[0, 0];
            
            int rows = jagged.Length;
            int cols = jagged[0].Length;
            float[,] result = new float[rows, cols];
            
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    result[i, j] = jagged[i][j];
                }
            }
            return result;
        }
    }
}
