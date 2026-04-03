using Contracts;
using Lib.Batching;
using Lib.Batching.Configuration;
using Lib.Batching.Sampling;
using Lib.Batching.Tests;
using Lib.Models.TinyNN;
using Lib.Models.TinyNN.Configuration;
using Lib.Models.TinyNN.State;
using Lib.Training;
using Lib.Training.Configuration;
using Lib.Training.Metrics;

namespace Integration.TrainingData.Test
{
    public class TinyNNTrainingTests
    {
        private TinyNNConfig _tinyNNConfig;
        private TinyNNWeights _tinyNNWeights;
        private ArrayTokenStream _arrayTokenStream;
        private TrainingLoop _trainingLoop;

        private int _vocabSize;
        private string _modelKind;

        private int[] _tokens;
                                                        
        [SetUp]
        public void Setup()
        {
            _vocabSize = 4;
            _modelKind = "TinyNN";
            _tokens = [0, 1, 2, 3];

            _trainingLoop = new TrainingLoop();
            _arrayTokenStream = new ArrayTokenStream(_tokens);
            _tinyNNConfig = new TinyNNConfig(_vocabSize);
            _tinyNNWeights = new TinyNNWeights(_tinyNNConfig.VocabSize, _tinyNNConfig.EmbeddingSize);
        }

        [Test]
        public void BatchingAndTraining_CanRunOneEpoch()
        {
            ILanguageModel model = new TinyNNModel(_modelKind, _vocabSize, _tinyNNConfig, _tinyNNWeights);
            IBatchProvider batchProvider = new TokenBatchProvider(_arrayTokenStream, new BatchWindowSampler());

            int epochs = 1;
            float lr = 0.01f;
            int checkpointIntreval = 1;
            TrainingConfig trainingConfig = new TrainingConfig(epochs, lr, checkpointIntreval);

            int batchSize = 2;
            int blockSize = 2;
            BatchConfig batchConfig = new BatchConfig(batchSize, blockSize);

            TrainingMetrics metrics = _trainingLoop.Train(model, batchProvider, trainingConfig, batchConfig, null);

            Assert.That(metrics.AverageLoss, Is.Not.NaN);
            Assert.That(metrics.CurrentEpoch, Is.EqualTo(1));
            Assert.That(metrics.ElapsedTime, Is.Not.Default);
            Assert.That(metrics.TotalSteps, Is.EqualTo(2));
        }

        [Test]
        public void BatchingAndTraining_TinyNN_ReceivesTrainStepCalls()
        {
            ILanguageModel model = new TinyNNModel(_modelKind, _vocabSize, _tinyNNConfig, _tinyNNWeights);
            IBatchProvider batchProvider = new TokenBatchProvider(_arrayTokenStream, new BatchWindowSampler());

            int epochs = 10;
            float lr = 0.01f;
            int checkpointIntreval = 2;
            TrainingConfig trainingConfig = new TrainingConfig(epochs, lr, checkpointIntreval);

            int batchSize = 2;
            int blockSize = 2;
            BatchConfig batchConfig = new BatchConfig(batchSize, blockSize);

            TrainingMetrics metrics = _trainingLoop.Train(model, batchProvider, trainingConfig, batchConfig, null);

            Assert.That(metrics.AverageLoss, Is.Not.NaN);
            Assert.That(metrics.CurrentEpoch, Is.EqualTo(10));
            Assert.That(metrics.ElapsedTime, Is.Not.Default);
            Assert.That(metrics.TotalSteps, Is.EqualTo(20));
        }

        [Test]
        public void BatchingAndTraining_NullModel_ThrowsArgumentException()
        {
            ILanguageModel? model = null;
            IBatchProvider batchProvider = new TokenBatchProvider(_arrayTokenStream, new BatchWindowSampler());

            int epochs = 10;
            float lr = 0.01f;
            int checkpointIntreval = 2;
            TrainingConfig trainingConfig = new TrainingConfig(epochs, lr, checkpointIntreval);

            int batchSize = 2;
            int blockSize = 2;
            BatchConfig batchConfig = new BatchConfig(batchSize, blockSize);

            Assert.Throws<ArgumentException>(() => _trainingLoop.Train(model, batchProvider, trainingConfig, batchConfig, null)); 
        }

        [Test]
        public void TinyNNTraining_ShortTokenStream_Handled()
        {
            var vocabSize = 2;
            var modelKind = "TinyNN";
            int[] tokens = [0, 1];

            var arrayTokenStream = new ArrayTokenStream(tokens);
            var tinyNNConfig = new TinyNNConfig(vocabSize);
            var tinyNNWeights = new TinyNNWeights(tinyNNConfig.VocabSize, tinyNNConfig.EmbeddingSize);

            ILanguageModel model = new TinyNNModel(modelKind, vocabSize, tinyNNConfig, tinyNNWeights);
            IBatchProvider batchProvider = new TokenBatchProvider(arrayTokenStream, new BatchWindowSampler());

            int epochs = 10;
            float lr = 0.01f;
            int checkpointIntreval = 2;
            TrainingConfig trainingConfig = new TrainingConfig(epochs, lr, checkpointIntreval);

            int batchSize = 2;
            int blockSize = 1;
            BatchConfig batchConfig = new BatchConfig(batchSize, blockSize);

            TrainingMetrics metrics = _trainingLoop.Train(model, batchProvider, trainingConfig, batchConfig, null);

            Assert.That(metrics.AverageLoss, Is.Not.NaN);
            Assert.That(metrics.CurrentEpoch, Is.EqualTo(10));
            Assert.That(metrics.ElapsedTime, Is.Not.Default);
            Assert.That(metrics.TotalSteps, Is.EqualTo(20));
        }

        [Test]
        public void TinyNNTraining_ShortTokenStream_HandledException()
        {
            var vocabSize = 1;
            var modelKind = "TinyNN";
            int[] tokens = [0];

            var arrayTokenStream = new ArrayTokenStream(tokens);
            var tinyNNConfig = new TinyNNConfig(vocabSize);
            var tinyNNWeights = new TinyNNWeights(tinyNNConfig.VocabSize, tinyNNConfig.EmbeddingSize);

            ILanguageModel model = new TinyNNModel(modelKind, vocabSize, tinyNNConfig, tinyNNWeights);
            IBatchProvider batchProvider = new TokenBatchProvider(arrayTokenStream, new BatchWindowSampler());

            int epochs = 10;
            float lr = 0.01f;
            int checkpointIntreval = 2;
            TrainingConfig trainingConfig = new TrainingConfig(epochs, lr, checkpointIntreval);

            int batchSize = 2;
            int blockSize = 1;
            BatchConfig batchConfig = new BatchConfig(batchSize, blockSize);

            Assert.Throws<InvalidOperationException>(() => _trainingLoop.Train(model, batchProvider, trainingConfig, batchConfig, null));
        }

        [Test]
        public void BatchingAndTraining_SeededRng_ProducesSameBatches()
        {
            var tokens = new[] { 0, 1, 2, 1, 0, 1, 2, 0, 1, 2 };

            var arrayTokenStream = new ArrayTokenStream(tokens);
            var batchProvider1 = new TokenBatchProvider(arrayTokenStream, new BatchWindowSampler());
            var batchProvider2 = new TokenBatchProvider(arrayTokenStream, new BatchWindowSampler());

            var batchConfig = new BatchConfig(2, 3);

            var rng1 = new Random(42);
            var rng2 = new Random(42);

            var batch1 = batchProvider1.GetBatch(batchConfig, rng1);
            var batch2 = batchProvider2.GetBatch(batchConfig, rng2);

            Assert.That(batch1.Inputs.Length, Is.EqualTo(batch2.Inputs.Length));
            Assert.That(batch1.Targets.Length, Is.EqualTo(batch2.Targets.Length));

            for (int i = 0; i < batch1.Inputs.Length; i++)
            {
                Assert.That(batch1.Inputs[i], Is.EqualTo(batch2.Inputs[i]));
            }

            Assert.That(batch1.Targets, Is.EqualTo(batch2.Targets));
        }

        [Test]
        public void BatchingAndTraining_StepCount_MatchesExpected()
        {
            ILanguageModel model = new TinyNNModel(_modelKind, _vocabSize, _tinyNNConfig, _tinyNNWeights);
            IBatchProvider batchProvider = new TokenBatchProvider(_arrayTokenStream, new BatchWindowSampler());

            int epochs = 100;
            float lr = 0.01f;
            int checkpointIntreval = 2;
            TrainingConfig trainingConfig = new TrainingConfig(epochs, lr, checkpointIntreval);

            int batchSize = 10;
            int blockSize = 2;
            BatchConfig batchConfig = new BatchConfig(batchSize, blockSize);

            TrainingMetrics metrics = _trainingLoop.Train(model, batchProvider, trainingConfig, batchConfig, null);

            Assert.That(metrics.AverageLoss, Is.Not.NaN);
            Assert.That(metrics.CurrentEpoch, Is.EqualTo(100));
            Assert.That(metrics.ElapsedTime, Is.Not.Default);
            Assert.That(metrics.TotalSteps, Is.EqualTo(1000));
        }
    }
}