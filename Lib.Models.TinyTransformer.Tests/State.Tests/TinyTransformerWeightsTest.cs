using Lib.Models.TinyTransformer.State;

namespace Lib.Models.TinyTransformer.Tests.State;

public class TinyTransformerWeightsTest
{
    private int _vocabSize = 10;
    private int _embeddingSize = 8;

    [Test]
    public void Initialize_ValidSizes_CreatesTokenEmbeddings()
    {
        var weights = TinyTransformerWeights.Initialize(_vocabSize, _embeddingSize);

        Assert.That(weights.TokenEmbeddings, Is.Not.Null);
        Assert.That(weights.TokenEmbeddings.GetLength(0), Is.EqualTo(_vocabSize));
        Assert.That(weights.TokenEmbeddings.GetLength(1), Is.EqualTo(_embeddingSize));
    }

    [Test]
    public void Initialize_ValidSizes_CreatesAttentionWeights()
    {
        var weights = TinyTransformerWeights.Initialize(_vocabSize, _embeddingSize);

        Assert.That(weights.Wq, Is.Not.Null);
        Assert.That(weights.Wk, Is.Not.Null);
        Assert.That(weights.Wv, Is.Not.Null);
        Assert.That(weights.Wo, Is.Not.Null);

        Assert.That(weights.Wq.GetLength(0), Is.EqualTo(_embeddingSize));
        Assert.That(weights.Wq.GetLength(1), Is.EqualTo(_embeddingSize));
    }

    [Test]
    public void Initialize_ValidSizes_CreatesFfnWeights()
    {
        int dff = 4 * _embeddingSize;
        var weights = TinyTransformerWeights.Initialize(_vocabSize, _embeddingSize);

        Assert.That(weights.Ffn1, Is.Not.Null);
        Assert.That(weights.Ffn1.GetLength(0), Is.EqualTo(_embeddingSize));
        Assert.That(weights.Ffn1.GetLength(1), Is.EqualTo(dff));

        Assert.That(weights.Ffn2, Is.Not.Null);
        Assert.That(weights.Ffn2.GetLength(0), Is.EqualTo(dff));
        Assert.That(weights.Ffn2.GetLength(1), Is.EqualTo(_embeddingSize));

        Assert.That(weights.Ffn1Bias.Length, Is.EqualTo(dff));
        Assert.That(weights.Ffn2Bias.Length, Is.EqualTo(_embeddingSize));
    }

    [Test]
    public void Initialize_ValidSizes_CreatesOutputWeights()
    {
        var weights = TinyTransformerWeights.Initialize(_vocabSize, _embeddingSize);

        Assert.That(weights.OutputW, Is.Not.Null);
        Assert.That(weights.OutputW.GetLength(0), Is.EqualTo(_embeddingSize));
        Assert.That(weights.OutputW.GetLength(1), Is.EqualTo(_vocabSize));

        Assert.That(weights.OutputBias, Is.Not.Null);
        Assert.That(weights.OutputBias.Length, Is.EqualTo(_vocabSize));
    }

    [Test]
    public void Initialize_WithSeed_ReturnsDeterministicWeights()
    {
        var weights1 = TinyTransformerWeights.Initialize(_vocabSize, _embeddingSize, new Random(42));
        var weights2 = TinyTransformerWeights.Initialize(_vocabSize, _embeddingSize, new Random(42));

        for (int i = 0; i < _vocabSize; i++)
        {
            for (int j = 0; j < _embeddingSize; j++)
            {
                Assert.That(weights1.TokenEmbeddings[i, j], Is.EqualTo(weights2.TokenEmbeddings[i, j]));
            }
        }
    }

    [Test]
    public void Initialize_WeightsInExpectedRange()
    {
        var weights = TinyTransformerWeights.Initialize(_vocabSize, _embeddingSize, new Random(42));

        for (int i = 0; i < _vocabSize; i++)
        {
            for (int j = 0; j < _embeddingSize; j++)
            {
                Assert.That(weights.TokenEmbeddings[i, j], Is.InRange(-0.02f, 0.02f));
            }
        }
    }

    [Test]
    public void JsonConstructor_ValidArrays_CreatesWeights()
    {
        float[][] tokenEmbed = CreateJagged(_vocabSize, _embeddingSize);
        float[][] wq = CreateJagged(_embeddingSize, _embeddingSize);
        float[][] wk = CreateJagged(_embeddingSize, _embeddingSize);
        float[][] wv = CreateJagged(_embeddingSize, _embeddingSize);
        float[][] wo = CreateJagged(_embeddingSize, _embeddingSize);
        int dff = 4 * _embeddingSize;
        float[][] ffn1 = CreateJagged(_embeddingSize, dff);
        float[] ffn1Bias = new float[dff];
        float[][] ffn2 = CreateJagged(dff, _embeddingSize);
        float[] ffn2Bias = new float[_embeddingSize];
        float[][] outputW = CreateJagged(_embeddingSize, _vocabSize);
        float[] outputBias = new float[_vocabSize];

        var weights = new TinyTransformerWeights(
            tokenEmbed, wq, wk, wv, wo,
            ffn1, ffn1Bias, ffn2, ffn2Bias,
            outputW, outputBias
        );

        Assert.That(weights.TokenEmbeddings.GetLength(0), Is.EqualTo(_vocabSize));
        Assert.That(weights.Wq.GetLength(0), Is.EqualTo(_embeddingSize));
    }

    private static float[][] CreateJagged(int rows, int cols)
    {
        var result = new float[rows][];
        for (int i = 0; i < rows; i++)
        {
            result[i] = new float[cols];
        }
        return result;
    }
}
