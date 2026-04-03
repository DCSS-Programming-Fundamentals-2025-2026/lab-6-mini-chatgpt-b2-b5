using Lib.Models.TinyNN;
using Lib.Models.TinyNN.Configuration;
using Lib.Models.TinyNN.State;

namespace Lib.Models.TinyNN.Tests.Model;

public class TinyNNModelTest
{
    private string _modelKind = "TinyNN";
    private int _vocabSize = 10;
    private TinyNNConfig _config;
    private TinyNNWeights _weights;
    private TinyNNModel _model;

    [SetUp]
    public void SetUp()
    {
        _config = new TinyNNConfig(_vocabSize);
        _weights = new TinyNNWeights(_vocabSize, _config.EmbeddingSize);
        _model = new TinyNNModel(_modelKind, _vocabSize, _config, _weights);
    }

    [Test]
    public void NextTokenScores_ValidContext_ReturnsScoresWithCorrectSize()
    {
        int[] tokens = { 1, 2, 3 };
        ReadOnlySpan<int> context = tokens;

        float[] scores = _model.NextTokenScores(context);

        Assert.That(scores, Is.Not.Null);
        Assert.That(scores.Length, Is.EqualTo(_config.VocabSize));
    }

    [Test]
    public void NextTokenScores_EmptyContext_ThrowsArgumentException()
    {
        int[] tokens = { };
        ReadOnlySpan<int> context = tokens;
        bool exception = false;

        try
        {
            _model.NextTokenScores(context);
        }
        catch (ArgumentException)
        {
            exception = true;   
        }

        Assert.That(exception, Is.True);
    }

    [Test]
    public void CalculateGradient_ValidInputs_DecreasesTargetProbability()
    {
        float[] probs = { 0.09f, 0.01f, 0,2f, 0,5f, 0,07f, 0,03f, 0,1f };
        int target = 3;
        float resultValue = probs[target] - 1;

        float[] resultProbs = _model.CalculateGradient(probs, target);

        Assert.That(resultValue, Is.EqualTo(resultProbs[target]));
    }

    [Test]
    public void TrainStep_CorrectContext_ReturnsIsNotNan()
    {
        int[] tokens = { 0, 1, 2, 3 };
        ReadOnlySpan<int> context = tokens;

        int target = 1;
        float lr = 0.01f;

        float result = _model.TrainStep(context, target, lr);

        Assert.That(result, Is.Not.NaN);
    }

    [Test]
    public void TrainStep_EmptyContext_ThrowsArgumentException()
    {
        int[] tokens = { };
        ReadOnlySpan<int> context = tokens;

        int target = 0;
        float lr = 0.01f;
        bool exception = false;

        try
        {
            _model.TrainStep(context, target, lr);
        }
        catch (ArgumentException)
        {
            exception = true;
        }

        Assert.That(exception, Is.True);
    }

    [TestCase(-1)]
    [TestCase(10)]
    public void TrainStep_InvalidTarget_ThrowsArgumentOutOfRangeException(int target)
    {
        int[] tokens = { 1, 3, 5, 2 };
        ReadOnlySpan<int> context = tokens;

        float lr = 0.01f;
        bool exception = false;

        try
        {
            _model.TrainStep(context, target, lr);
        }
        catch (ArgumentOutOfRangeException)
        {
            exception = true;
        }

        Assert.That(exception, Is.True);
    }

    [TestCase(0)]
    [TestCase(-1)]
    public void TrainStep_InvalidLearningRate_ThrowsArgumentException(int lr)
    {
        int[] tokens = { 1, 3, 5, 2 };
        ReadOnlySpan<int> context = tokens;

        int target = 1;
        bool exception = false;

        try
        {
            _model.TrainStep(context, target, lr);
        }
        catch (ArgumentException)
        {
            exception = true;
        }

        Assert.That(exception, Is.True);
    }
}