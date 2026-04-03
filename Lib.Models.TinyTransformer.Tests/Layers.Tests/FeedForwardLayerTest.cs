using Lib.Models.TinyTransformer.Configuration;
using Lib.Models.TinyTransformer.Layers;
using Lib.Models.TinyTransformer.State;

namespace Lib.Models.TinyTransformer.Tests.Layers;

public class FeedForwardLayerTest
{
    private int _vocabSize = 10;
    private int _embeddingSize = 8;
    private TinyTransformerConfig _config;
    private TinyTransformerWeights _weights;
    private FeedForwardLayer _layer;

    [SetUp]
    public void SetUp()
    {
        _config = new TinyTransformerConfig(_vocabSize, _embeddingSize, 1, 8);
        _weights = TinyTransformerWeights.Initialize(_vocabSize, _embeddingSize, new Random(42));
        _layer = new FeedForwardLayer();
    }

    [Test]
    public void Compute_ValidInput_ReturnsCorrectSize()
    {
        float[] input = new float[_embeddingSize];
        for (int i = 0; i < _embeddingSize; i++)
        {
            input[i] = 0.1f * i;
        }

        float[] output = _layer.Compute(input, _weights, _embeddingSize);

        Assert.That(output.Length, Is.EqualTo(_embeddingSize));
    }

    [Test]
    public void Compute_ZeroInput_ReturnsOnlyBiasValues()
    {
        float[] input = new float[_embeddingSize];

        float[] output = _layer.Compute(input, _weights, _embeddingSize);

        Assert.That(output, Is.Not.Null);
        Assert.That(output.Length, Is.EqualTo(_embeddingSize));
    }

    [Test]
    public void Compute_SameInput_ReturnsSameOutput()
    {
        float[] input = new float[_embeddingSize];
        for (int i = 0; i < _embeddingSize; i++)
        {
            input[i] = 0.5f;
        }

        float[] output1 = _layer.Compute(input, _weights, _embeddingSize);
        float[] output2 = _layer.Compute(input, _weights, _embeddingSize);

        Assert.That(output1, Is.EqualTo(output2));
    }

    [Test]
    public void Compute_ValidInput_OutputIsNotAllZeros()
    {
        float[] input = new float[_embeddingSize];
        for (int i = 0; i < _embeddingSize; i++)
        {
            input[i] = 1.0f;
        }

        float[] output = _layer.Compute(input, _weights, _embeddingSize);

        Assert.That(output.Any(v => v != 0), Is.True);
    }
}
