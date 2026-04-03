using Lib.Models.TinyNN.Configuration;
using Lib.Models.TinyNN.State;

namespace Lib.Models.TinyNN.Tests.State;

public class TinyNNWeightsTest
{
    private int _vocabSize = 10;
    private TinyNNConfig _config;
    private TinyNNWeights _weights;

    [SetUp]
    public void SetUp()
    {
        _config = new TinyNNConfig(_vocabSize);
        _weights = new TinyNNWeights(_vocabSize, _config.EmbeddingSize);
    }

    [Test]
    public void InitializeEmbeddings_ValidSizes_CreatesCorrectMatrix()
    {
        float[][] embeddings = _weights.Embeddings;

        Assert.That(embeddings, Is.Not.Null);
        Assert.That(embeddings.Length, Is.EqualTo(_vocabSize));

        for (int i = 0; i < embeddings.Length; i++)
        {
            Assert.That(embeddings[i], Is.Not.Null);
            Assert.That(embeddings[i].Length, Is.EqualTo(_config.EmbeddingSize));

            for (int j = 0; j < embeddings[i].Length; j++)
            {
                Assert.That(embeddings[i][j], Is.InRange(-0.1f, 0.1f));
            }
        }
    }

    [Test]
    public void InitializeOutputWeights_ValidSizes_CreatesCorrectMatrix()
    {
        float[][] weights = _weights.OutputWeights;

        Assert.That(weights, Is.Not.Null);
        Assert.That(weights.Length, Is.EqualTo(_config.EmbeddingSize));

        for (int i = 0; i < weights.Length; i++)
        {
            Assert.That(weights[i], Is.Not.Null);
            Assert.That(weights[i].Length, Is.EqualTo(_vocabSize));

            for (int j = 0; j < weights[i].Length; j++)
            {
                Assert.That(weights[i][j], Is.InRange(-0.1f, 0.1f));
            }
        }
    }

    //This test is not needed now

    //[Test]
    //public void UpdateAllWeights_ValidInputs_UpdatesWeightsCorrectly()
    //{
        //float[][] newEmbeddings = new float[_vocabSize][];
        //for (int i = 0; i < _vocabSize; i++)
        //{
            //newEmbeddings[i] = new float[_config.EmbeddingSize];
        //}

        //float[][] newOutputWeights = new float[_config.EmbeddingSize][];
        //for (int i = 0; i < _config.EmbeddingSize; i++)
        //{
            //newOutputWeights[i] = new float[_vocabSize];
        //}

        //float[] newOutputBias = new float[_vocabSize];

        //_weights.UpdateAllWeights(newEmbeddings, newOutputWeights, newOutputBias);

        //Assert.That(_weights.Embeddings, Is.EqualTo(newEmbeddings));
        //Assert.That(_weights.OutputWeights, Is.EqualTo(newOutputWeights));
        //Assert.That(_weights.OutputBias, Is.EqualTo(newOutputBias));
    //}
}