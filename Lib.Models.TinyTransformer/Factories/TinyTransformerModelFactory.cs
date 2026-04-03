using Lib.Models.TinyTransformer.Configuration;
using Lib.Models.TinyTransformer.State;
using System.Text.Json;

namespace Lib.Models.TinyTransformer.Factories
{
    public class TinyTransformerModelFactory
    {
        public TinyTransformerModel Create(TinyTransformerConfig config, TinyTransformerWeights weights)
        {
            return new TinyTransformerModel(config, weights);
        }

        public TinyTransformerModel Create(int vocabSize, int? seed = null)
        {
            TinyTransformerConfig config = new TinyTransformerConfig(vocabSize);
            Random random = seed.HasValue ? new Random(seed.Value) : new Random();
            TinyTransformerWeights weights = TinyTransformerWeights.Initialize(vocabSize, config.EmbeddingSize, random);
            return new TinyTransformerModel(config, weights);
        }

        public TinyTransformerModel Create(int vocabSize, int embeddingSize, int headCount, int contextSize, int? seed = null)
        {
            TinyTransformerConfig config = new TinyTransformerConfig(vocabSize, embeddingSize, headCount, contextSize);
            Random random = seed.HasValue ? new Random(seed.Value) : new Random();
            TinyTransformerWeights weights = TinyTransformerWeights.Initialize(vocabSize, embeddingSize, random);
            return new TinyTransformerModel(config, weights);
        }

        public TinyTransformerModel CreateFromPayload(JsonElement payload)
        {
            var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
            var data = JsonSerializer.Deserialize<TinyTransformerPayload>(payload.GetRawText(), options);

            if (data == null || data.Config == null)
            {
                throw new ArgumentException("Payload is invalid or empty");
            }

            var weights = new TinyTransformerWeights(
                data.TokenEmbeddings, data.Wq, data.Wk, data.Wv, data.Wo,
                data.Ffn1, data.Ffn1Bias, data.Ffn2, data.Ffn2Bias,
                data.OutputW, data.OutputBias
            );

            return new TinyTransformerModel(data.Config, weights);
        }
    }
}
