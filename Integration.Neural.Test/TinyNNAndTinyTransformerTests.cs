using Contracts;
using Lib.MathCore;
using Lib.Models.TinyNN;
using Lib.Models.TinyNN.Configuration;
using Lib.Models.TinyNN.Factories;
using Lib.Models.TinyNN.State;
using Lib.Models.TinyTransformer;
using Lib.Models.TinyTransformer.Configuration;
using Lib.Models.TinyTransformer.Factories;
using Lib.Models.TinyTransformer.State;
using System.Text.Json;

namespace Integration.Neural.Test
{
    public class TinyNNAndTinyTransformerTests
    {
        private int _vocabSize;
        private int[] _tokens;

        private TinyNNConfig _tinyNNConfig;
        private TinyNNWeights _tinyNNWeights;

        [SetUp]
        public void Setup()
        {
            _vocabSize = 10;
            _tokens = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

            _tinyNNConfig = new TinyNNConfig(_vocabSize);
            _tinyNNWeights = new TinyNNWeights(_tinyNNConfig.VocabSize, _tinyNNConfig.EmbeddingSize);
        }

        [Test]
        public void TinyNNAndTinyTransformer_BothImplementILanguageModel()
        {
            ILanguageModel tinyNN = new TinyNNModel("TinyNN", _vocabSize, _tinyNNConfig, _tinyNNWeights);

            var transformerConfig = new TinyTransformerConfig(_vocabSize, 32, 1, 8);
            var transformerWeights = TinyTransformerWeights.Initialize(_vocabSize, 32, new Random(42));
            ILanguageModel transformer = new TinyTransformerModel(transformerConfig, transformerWeights);

            float[] logitsTinyNN = tinyNN.NextTokenScores(_tokens);
            float[] logitsTransformer = transformer.NextTokenScores(_tokens);

            Assert.That(logitsTinyNN.Length, Is.EqualTo(_vocabSize));
            for (int i = 0; i < logitsTinyNN.Length; i++)
            {
                Assert.That(logitsTinyNN[i], Is.Not.NaN);
            }

            Assert.That(logitsTransformer.Length, Is.EqualTo(_vocabSize));
            for (int i = 0; i < logitsTransformer.Length; i++)
            {
                Assert.That(logitsTransformer[i], Is.Not.NaN);
            }
        }

        [Test]
        public void TinyNN_SoftmaxSumsToOne()
        {
            var factory = new TinyNNModelFactory();
            var model = factory.CreateNewModel("tinynn", _vocabSize);

            float[] logits = model.NextTokenScores(_tokens);
            float[] probabilities = SoftmaxCalculator.Softmax(logits); 

            float sum = 0;
            foreach (var p in probabilities)
            {
                sum += p;
            }

            Assert.That(sum, Is.EqualTo(1.0f).Within(1e-6f), "Сума ймовірностей після Softmax має дорівнювати 1.0"); 
            Assert.That(probabilities, Is.All.GreaterThanOrEqualTo(0f), "Ймовірності не можуть бути від'ємними");
        }

        [Test]
        public void TinyNN_CheckpointRoundTrip_PreservesModel()
        {
            var factory = new TinyNNModelFactory();
            var originalModel = factory.CreateNewModel("tinynn", _vocabSize, _tinyNNConfig.EmbeddingSize);

            float[] originalScores = originalModel.NextTokenScores(_tokens);

            var payload = originalModel.ToPayload();
            string json = JsonSerializer.Serialize(payload);

            var doc = JsonDocument.Parse(json);
            var restoredModel = factory.CreateFromPayload(doc.RootElement, "tinynn");

            float[] restoredScores = restoredModel.NextTokenScores(_tokens);

            Assert.Multiple(() =>
            {
                Assert.That(restoredScores.Length, Is.EqualTo(originalScores.Length), "Довжина логітів має збігатися з VocabSize."); 

                for (int i = 0; i < originalScores.Length; i++)
                {
                    Assert.That(restoredScores[i], Is.EqualTo(originalScores[i]).Within(1e-6f), $"Логіт на позиції {i} розбігається після завантаження!");
                }
            });
        }

        [Test]
        public void TinyTransformer_CheckpointRoundTrip_PreservesModel()
        {
            var factory = new TinyTransformerModelFactory();
            int headCount = 2;
            
            var originalModel = factory.Create(_vocabSize, _tinyNNConfig.EmbeddingSize, headCount, _tinyNNConfig.ContextSize);

            float[] originalScores = originalModel.NextTokenScores(_tokens);

            var payload = originalModel.ToPayload();
            string json = JsonSerializer.Serialize(payload);
            
            using var doc = JsonDocument.Parse(json);
            var restoredModel = factory.CreateFromPayload(doc.RootElement);

            float[] restoredScores = restoredModel.NextTokenScores(_tokens);

            Assert.That(restoredScores.Length, Is.EqualTo(originalScores.Length), "Кількість логітів має бути рівною VocabSize");

            for (int i = 0; i < originalScores.Length; i++)
            {
                Assert.That(restoredScores[i], Is.EqualTo(originalScores[i]).Within(1e-6f), $"Розбіжність у вагах трансформера на позиції {i}");
            }
        }  
    }
}