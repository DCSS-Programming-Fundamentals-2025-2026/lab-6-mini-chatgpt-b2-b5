using Contracts;
using Lib.MathCore;
using Lib.Models.TinyTransformer.Configuration;
using Lib.Models.TinyTransformer.Layers;
using Lib.Models.TinyTransformer.State;

namespace Lib.Models.TinyTransformer
{
    public class TinyTransformerModel : ILanguageModel
    {
        private readonly TinyTransformerConfig _config;
        private readonly TinyTransformerWeights _weights;
        private readonly TransformerBlock _transformerBlock;

        public string ModelKind => "tinytransformer";
        public int VocabSize => _config.VocabSize;
        public int ContextSize => _config.ContextSize;

        public TinyTransformerModel(TinyTransformerConfig config, TinyTransformerWeights weights)
        {
            _config = config;
            _weights = weights;
            _transformerBlock = new TransformerBlock();
        }

        public float[] NextTokenScores(ReadOnlySpan<int> context)
        {
            if (context.Length == 0)
            {
                return new float[_config.VocabSize];
            }

            var truncatedContext = context.Length > _config.ContextSize
                ? context.Slice(context.Length - _config.ContextSize)
                : context;

            return Forward(truncatedContext, _config.VocabSize, _config.EmbeddingSize);
        }

        private float[] Forward(ReadOnlySpan<int> context, int vocabSize, int embeddingSize)
        {
            float[][] embeddings = Embed(context, embeddingSize);

            float[][] blockOutput = _transformerBlock.Forward(embeddings, _weights, embeddingSize);

            float[] lastHidden = blockOutput[blockOutput.Length - 1];

            return Project(lastHidden, vocabSize);
        }

        private float[][] Embed(ReadOnlySpan<int> context, int embeddingSize)
        {
            float[][] embeddings = new float[context.Length][];
            for (int i = 0; i < context.Length; i++)
            {
                embeddings[i] = new float[embeddingSize];
                int tokenId = context[i];
                for (int j = 0; j < embeddingSize; j++)
                {
                    embeddings[i][j] = _weights.TokenEmbeddings[tokenId, j];
                }
            }
            return embeddings;
        }

        private float[] Project(float[] hidden, int vocabSize)
        {
            float[] logits = new float[vocabSize];
            int embeddingSize = hidden.Length;

            for (int v = 0; v < vocabSize; v++)
            {
                float sum = _weights.OutputBias[v];
                for (int i = 0; i < embeddingSize; i++)
                {
                    sum += hidden[i] * _weights.OutputW[i, v];
                }
                logits[v] = sum;
            }

            return logits;
        }

        public TinyTransformerPayload ToPayload()
        {
            return new TinyTransformerPayload
            {
                Config = this._config,
                TokenEmbeddings = ToJaggedArray(this._weights.TokenEmbeddings),
                Wq = ToJaggedArray(this._weights.Wq),
                Wk = ToJaggedArray(this._weights.Wk),
                Wv = ToJaggedArray(this._weights.Wv),
                Wo = ToJaggedArray(this._weights.Wo),
                Ffn1 = ToJaggedArray(this._weights.Ffn1),
                Ffn1Bias = this._weights.Ffn1Bias,
                Ffn2 = ToJaggedArray(this._weights.Ffn2),
                Ffn2Bias = this._weights.Ffn2Bias,
                OutputW = ToJaggedArray(this._weights.OutputW),
                OutputBias = this._weights.OutputBias
            };
        }

        public object GetPayloadForCheckpoint()
        {
            return new
            {
                config = new
                {
                    vocabSize = _config.VocabSize,
                    embeddingSize = _config.EmbeddingSize,
                    headCount = _config.HeadCount,
                    contextSize = _config.ContextSize
                },
                tokenEmbeddings = ToJaggedArray(_weights.TokenEmbeddings),
                wq = ToJaggedArray(_weights.Wq),
                wk = ToJaggedArray(_weights.Wk),
                wv = ToJaggedArray(_weights.Wv),
                wo = ToJaggedArray(_weights.Wo),
                ffn1 = ToJaggedArray(_weights.Ffn1),
                ffn1Bias = _weights.Ffn1Bias,
                ffn2 = ToJaggedArray(_weights.Ffn2),
                ffn2Bias = _weights.Ffn2Bias,
                outputW = ToJaggedArray(_weights.OutputW),
                outputBias = _weights.OutputBias
            };
        }

        private static float[][] ToJaggedArray(float[,] matrix)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);
            float[][] result = new float[rows][];
            for (int i = 0; i < rows; i++)
            {
                result[i] = new float[cols];
                for (int j = 0; j < cols; j++)
                {
                    result[i][j] = matrix[i, j];
                }
            }
            return result;
        }
    }
}
