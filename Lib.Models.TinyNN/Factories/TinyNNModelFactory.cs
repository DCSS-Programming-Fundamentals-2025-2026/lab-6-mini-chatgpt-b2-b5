using System.Text.Json;
using Lib.Models.TinyNN.Configuration;
using Lib.Models.TinyNN.State;
namespace Lib.Models.TinyNN.Factories
{
    public class TinyNNModelFactory
    {
        public TinyNNModel CreateNewModel(string modelKind, int vocabSize, int embeddingSize = 32, int contextSize = 8192)
        {
            TinyNNConfig config = new TinyNNConfig(vocabSize, embeddingSize, contextSize);
            TinyNNWeights weights = new TinyNNWeights(vocabSize, embeddingSize);

            return new TinyNNModel(modelKind, vocabSize, config, weights);
        }

        public TinyNNModel CreateFromPayload(JsonElement payload, string modelKind)
        {
            if (modelKind != "tinynn")
            {
                throw new ArgumentException("Incorrect ModelKind!");
            }

            var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
            var data = JsonSerializer.Deserialize<TinyNNPayload>(payload.GetRawText(), options);

            if (data == null || data.Config == null || data.Weights == null)
            {
                throw new ArgumentNullException(nameof(payload), "Payload cannot be empty!");
            }
            
            return new TinyNNModel(modelKind, data.Config.VocabSize, data.Config, data.Weights);
        }
    }
}