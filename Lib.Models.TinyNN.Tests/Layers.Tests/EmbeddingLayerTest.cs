using Lib.Models.TinyNN.Configuration;
using Lib.Models.TinyNN.State;
using Lib.Models.TinyNN.Layers;

namespace Lib.Models.TinyNN.Tests.Layers;

public class EmbeddingLayerTest
{
    private int _vocabSize = 10;
    private TinyNNConfig _config;
    private TinyNNWeights _weights;
    private EmbeddingLayer _layer;

    [SetUp]
    public void SetUp()
    {
        _config = new TinyNNConfig(_vocabSize);
        _weights = new TinyNNWeights(_vocabSize, _config.EmbeddingSize);
        _layer = new EmbeddingLayer(_vocabSize, _config, _weights);
    }

    [Test]
    public void EncodeContext_ValidContext_ReturnsVectorWithEmbeddingSizeLength()
    {
        int[] context = new int[] { 0, 3, 5, 2, 7 };
        float[] hidden = _layer.EncodeContext(context);
        Assert.That(hidden.Length, Is.EqualTo(32));
    }

    [Test]
    public void EncodeContext_ContextExceedsLimit_ReturnsVectorWithEmbeddingSizeLength()
    {
        int[] context = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        float[] hidden = _layer.EncodeContext(context);
        Assert.That(hidden.Length, Is.EqualTo(32));
    }

    [Test]
    public void EncodeContext_ContextIsEmpty_ThrowArgumentException()
    {
        int[] context = new int[] { };
        Assert.Throws<ArgumentException>(() => _layer.EncodeContext(context));
    }

    [Test]
    public void GetVectorFromId_ValidId_ReturnsCorrectEmbeddingArray()
    {
        int validId = 1;
        float[] expectedVector = _weights.Embeddings[validId];

        float[] result = _layer.GetVectorFromId(validId);

        Assert.That(result, Is.EqualTo(expectedVector));
    }

    [Test]
    public void GetVectorFromId_IdLargerThanVocabSize_ThrowsArgumentOutOfRangeException()
    {
        int invalidId = 11;
        Assert.Throws<ArgumentOutOfRangeException>(() => _layer.GetVectorFromId(invalidId));
    }

    [Test]
    public void ContextCutter_ContextExceedsLimit_ReturnsOnlyLastTokens()
    {
        int[] context1 = new int[] { 2, 3, 4, 5, 6, 7, 8, 9 };
        int[] context2 = new int[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        int[] cuttedContext1 = _layer.ContextCutter(context1);
        int[] cuttedContext2 = _layer.ContextCutter(context2);

        Assert.That(cuttedContext1, Is.Not.EqualTo(cuttedContext2));
    }

    [Test]
    public void Backward_ValidInputs_UpdatesEmbeddingsCorrectly()
    {
        int[] tokens = { 3 };
        ReadOnlySpan<int> context = tokens; 
        float[] dHidden =
        {
            -0.5f, -0.7f, 1f, 2f, -1.4f, 0.7f, 0.8f, 0.6f, 0.2f, -0.3f, 1.1f, -0.9f, 0.4f, 0.5f, -0.2f, 0.3f,
            -0.6f, 0.9f, -1.2f, 0.1f, 0.8f, -0.4f, 0.6f, -0.1f, 1.3f, -0.8f, 0.2f, 0.7f, -0.5f, 0.4f, -0.3f, 0.9f
        };

        float[] oldEmbeddings = new float[_weights.Embeddings[3].Length]; 
        Array.Copy(_weights.Embeddings[3], oldEmbeddings, _weights.Embeddings[3].Length);        

        float lr = 0.01f;
        _layer.Backward(context, dHidden, lr); 

        for (int i = 0; i < _weights.Embeddings[3].Length; i++) 
        {
            float delta = Math.Abs(dHidden[i] * lr);
            float absDelta = Math.Abs(oldEmbeddings[i] - _weights.Embeddings[3][i]);

            Assert.That(delta, Is.EqualTo(absDelta).Within(0.000001f)); 
        }
    }
}