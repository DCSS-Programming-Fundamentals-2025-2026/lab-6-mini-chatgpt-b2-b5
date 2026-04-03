using Lib.Batching;
using Lib.Batching.Configuration;
using Lib.Batching.Sampling;
using Lib.Batching.Tests;
using NUnit.Framework.Legacy;

namespace Lib.BatchingTests
{
    public class BatchingTests
    {
        [TestFixture]
        public class TokenBatchProviderTests
        {
            [Test]
            public void GetBatch_ReturnsCorrectShape()
            {
                int[] tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
                var stream = new ArrayTokenStream(tokens);
                var sampler = new BatchWindowSampler();
                var provider = new TokenBatchProvider(stream, sampler);

                var config = new BatchConfig(2, 3);
                var batch = provider.GetBatch(config, new Random(42));

                Assert.That(batch.Inputs.Length, Is.EqualTo(2));
                Assert.That(batch.Targets.Length, Is.EqualTo(2));
                Assert.That(batch.Inputs[0].Length, Is.EqualTo(3));
                Assert.That(batch.Inputs[1].Length, Is.EqualTo(3));
            }

            [Test]
            public void GetBatch_Targets_AreImmediatelyAfterContexts()
            {
                int[] tokens = { 10, 20, 30, 40, 50, 60, 70 };
                var stream = new ArrayTokenStream(tokens);
                var sampler = new BatchWindowSampler();
                var provider = new TokenBatchProvider(stream, sampler);

                var config = new BatchConfig(5, 2);
                var batch = provider.GetBatch(config, new Random(1));

                for (int i = 0; i < batch.Inputs.Length; i++)
                {
                    int[] context = batch.Inputs[i];
                    int target = batch.Targets[i];

                    bool matchedWindow = false;

                    for (int start = 0; start <= tokens.Length - config.BlockSize - 1; start++)
                    {
                        bool same = true;

                        for (int j = 0; j < config.BlockSize; j++)
                        {
                            if (tokens[start + j] != context[j])
                            {
                                same = false;
                                break;
                            }
                        }

                        if (same)
                        {
                            Assert.That(target, Is.EqualTo(tokens[start + config.BlockSize]));
                            matchedWindow = true;
                            break;
                        }
                    }

                    Assert.That(matchedWindow, Is.True);
                }
            }

            [Test]
            public void GetBatch_Throws_WhenNotEnoughTokens()
            {
                int[] tokens = { 1, 2 };
                var stream = new ArrayTokenStream(tokens);
                var sampler = new BatchWindowSampler();
                var provider = new TokenBatchProvider(stream, sampler);

                var config = new BatchConfig(1, 2);

                Assert.Throws<InvalidOperationException>(() =>
                    provider.GetBatch(config, new Random(42)));
            }

            [Test]
            public void GetBatch_IsDeterministic_WithSameSeed()
            {
                int[] tokens = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
                var stream = new ArrayTokenStream(tokens);
                var sampler = new BatchWindowSampler();
                var provider = new TokenBatchProvider(stream, sampler);

                var config = new BatchConfig(3, 2);

                var batch1 = provider.GetBatch(config, new Random(123));
                var batch2 = provider.GetBatch(config, new Random(123));

                Assert.That(batch1.Inputs.Length, Is.EqualTo(batch2.Inputs.Length));
                Assert.That(batch1.Targets.Length, Is.EqualTo(batch2.Targets.Length));

                for (int i = 0; i < batch1.Inputs.Length; i++)
                {
                    CollectionAssert.AreEqual(batch1.Inputs[i], batch2.Inputs[i]);
                    Assert.That(batch1.Targets[i], Is.EqualTo(batch2.Targets[i]));
                }
            }
        }
    }
}
