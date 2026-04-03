using Contracts;
using Lib.MathCore;
using Lib.Models.TinyTransformer.Configuration;
using Lib.Models.TinyTransformer.State;

namespace Lib.Models.TinyTransformer.Tests.Model;

public class TinyTransformerModelTest
{
    private int _vocabSize = 10;
    private int _embeddingSize = 8;
    private TinyTransformerConfig _config;
    private TinyTransformerWeights _weights;
    private TinyTransformerModel _model;

    [SetUp]
    public void SetUp()
    {
        _config = new TinyTransformerConfig(_vocabSize, _embeddingSize, 1, 8);
        _weights = TinyTransformerWeights.Initialize(_vocabSize, _embeddingSize, new Random(42));
        _model = new TinyTransformerModel(_config, _weights);
    }

    [Test]
    public void ModelKind_ReturnsExpectedValue()
    {
        Assert.That(_model.ModelKind, Is.EqualTo("tinytransformer"));
    }

    [Test]
    public void VocabSize_ReturnsConfigValue()
    {
        Assert.That(_model.VocabSize, Is.EqualTo(_vocabSize));
    }

    [Test]
    public void ContextSize_ReturnsConfigValue()
    {
        Assert.That(_model.ContextSize, Is.EqualTo(8));
    }

    [Test]
    public void NextTokenScores_ValidContext_ReturnsCorrectSize()
    {
        int[] tokens = { 1, 2, 3 };
        ReadOnlySpan<int> context = tokens;

        float[] scores = _model.NextTokenScores(context);

        Assert.That(scores, Is.Not.Null);
        Assert.That(scores.Length, Is.EqualTo(_vocabSize));
    }

    [Test]
    public void NextTokenScores_EmptyContext_ReturnsZeros()
    {
        int[] tokens = { };
        ReadOnlySpan<int> context = tokens;

        float[] scores = _model.NextTokenScores(context);

        Assert.That(scores.Length, Is.EqualTo(_vocabSize));
        Assert.That(scores.All(s => s == 0), Is.True);
    }

    [Test]
    public void NextTokenScores_SingleToken_ReturnsScores()
    {
        int[] tokens = { 0 };
        ReadOnlySpan<int> context = tokens;

        float[] scores = _model.NextTokenScores(context);

        Assert.That(scores.Length, Is.EqualTo(_vocabSize));
    }

    [Test]
    public void NextTokenScores_SameContext_ReturnsSameScores()
    {
        int[] tokens = { 1, 2, 3 };

        float[] scores1 = _model.NextTokenScores(tokens);
        float[] scores2 = _model.NextTokenScores(tokens);

        Assert.That(scores1, Is.EqualTo(scores2));
    }

    [Test]
    public void ToPayload_ReturnsValidPayload()
    {
        var payload = _model.ToPayload();

        Assert.That(payload, Is.Not.Null);
        Assert.That(payload.Config, Is.Not.Null);
        Assert.That(payload.Config.VocabSize, Is.EqualTo(_vocabSize));
        Assert.That(payload.TokenEmbeddings, Is.Not.Null);
        Assert.That(payload.OutputW, Is.Not.Null);
        Assert.That(payload.OutputBias, Is.Not.Null);
    }

    [Test]
    public void GetPayloadForCheckpoint_ReturnsNotNull()
    {
        var payload = _model.GetPayloadForCheckpoint();

        Assert.That(payload, Is.Not.Null);
    }

    [Test]
    public void Model_ImplementsILanguageModel()
    {
        Assert.That(_model, Is.InstanceOf<ILanguageModel>());
    }

    [Test]
    public void NextTokenScores_CanBeUsedWithMathOpsSoftmax()
    {
        int[] tokens = { 1, 2, 3 };

        float[] logits = _model.NextTokenScores(tokens);
        float[] probs = MathOps.Default.Softmax(logits);

        Assert.That(probs.Length, Is.EqualTo(_vocabSize));
        Assert.That(probs.Sum(), Is.EqualTo(1.0f).Within(0.0001f));
    }

    [Test]
    public void NextTokenScores_CanBeUsedWithMathOpsArgMax()
    {
        int[] tokens = { 1, 2, 3 };

        float[] logits = _model.NextTokenScores(tokens);
        int predictedToken = MathOps.Default.ArgMax(logits);

        Assert.That(predictedToken, Is.GreaterThanOrEqualTo(0));
        Assert.That(predictedToken, Is.LessThan(_vocabSize));
    }

    [Test]
    public void NextTokenScores_CanBeUsedWithMathOpsSampleFromProbs()
    {
        int[] tokens = { 1, 2, 3 };

        float[] logits = _model.NextTokenScores(tokens);
        float[] probs = MathOps.Default.Softmax(logits);
        int sampledToken = MathOps.Default.SampleFromProbs(probs, new Random(42));

        Assert.That(sampledToken, Is.GreaterThanOrEqualTo(0));
        Assert.That(sampledToken, Is.LessThan(_vocabSize));
    }
}
