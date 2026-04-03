using Lib.Models.TinyNN;
using Lib.Models.TinyNN.Configuration;
using Lib.Models.TinyNN.State;

namespace Integration.Neural.Test
{
    public class TinyNNTinyNNTrainStep
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
        public void TinyNN_TrainStep_DecreasesLoss()
        {
            var tinyNN = new TinyNNModel("TinyNN", _vocabSize, _tinyNNConfig, _tinyNNWeights);
            int target = 5;
            float lr = 0.01f;

            float loss1 = tinyNN.TrainStep(_tokens, target, lr);
            float loss2 = tinyNN.TrainStep(_tokens, target, lr);

            bool loss2IsLess = loss2 < loss1; 

            Assert.That(loss2IsLess, Is.True);
        }
    }
}