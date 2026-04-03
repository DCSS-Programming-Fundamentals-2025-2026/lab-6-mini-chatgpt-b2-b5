using Lib.Models.TinyTransformer.Configuration;
using Lib.Models.TinyTransformer.Layers;
using Lib.Models.TinyTransformer.State;

namespace Lib.Models.TinyTransformer.Tests.Layers;

public class SelfAttentionLayerTest
{
    private int _vocabSize = 10;
    private int _embeddingSize = 8;
    private TinyTransformerConfig _config;
    private TinyTransformerWeights _weights;
    private SelfAttentionLayer _layer;

    [SetUp]
    public void SetUp()
    {
        _config = new TinyTransformerConfig(_vocabSize, _embeddingSize, 1, 8);
        _weights = TinyTransformerWeights.Initialize(_vocabSize, _embeddingSize, new Random(42));
        _layer = new SelfAttentionLayer();
    }

    [Test]
    public void Compute_SingleToken_ReturnsCorrectShape()
    {
        float[][] input = new float[1][];
        input[0] = new float[_embeddingSize];
        for (int i = 0; i < _embeddingSize; i++)
        {
            input[0][i] = 0.1f * i;
        }

        float[][] output = _layer.Compute(input, _weights, _embeddingSize);

        Assert.That(output.Length, Is.EqualTo(1));
        Assert.That(output[0].Length, Is.EqualTo(_embeddingSize));
    }

    [Test]
    public void Compute_MultipleTokens_ReturnsCorrectShape()
    {
        int seqLen = 4;
        float[][] input = new float[seqLen][];
        for (int i = 0; i < seqLen; i++)
        {
            input[i] = new float[_embeddingSize];
            for (int j = 0; j < _embeddingSize; j++)
            {
                input[i][j] = 0.1f * (i + j);
            }
        }

        float[][] output = _layer.Compute(input, _weights, _embeddingSize);

        Assert.That(output.Length, Is.EqualTo(seqLen));
        for (int i = 0; i < seqLen; i++)
        {
            Assert.That(output[i].Length, Is.EqualTo(_embeddingSize));
        }
    }

    [Test]
    public void Compute_ValidInput_OutputIsNotAllZeros()
    {
        float[][] input = new float[2][];
        for (int i = 0; i < 2; i++)
        {
            input[i] = new float[_embeddingSize];
            for (int j = 0; j < _embeddingSize; j++)
            {
                input[i][j] = 1.0f;
            }
        }

        float[][] output = _layer.Compute(input, _weights, _embeddingSize);

        bool hasNonZero = output.Any(row => row.Any(v => v != 0));
        Assert.That(hasNonZero, Is.True);
    }

    [Test]
    public void Compute_SameInput_ReturnsSameOutput()
    {
        float[][] input = new float[2][];
        for (int i = 0; i < 2; i++)
        {
            input[i] = new float[_embeddingSize];
            for (int j = 0; j < _embeddingSize; j++)
            {
                input[i][j] = 0.5f;
            }
        }

        float[][] output1 = _layer.Compute(input, _weights, _embeddingSize);
        float[][] output2 = _layer.Compute(input, _weights, _embeddingSize);

        for (int i = 0; i < output1.Length; i++)
        {
            Assert.That(output1[i], Is.EqualTo(output2[i]));
        }
    }
}
