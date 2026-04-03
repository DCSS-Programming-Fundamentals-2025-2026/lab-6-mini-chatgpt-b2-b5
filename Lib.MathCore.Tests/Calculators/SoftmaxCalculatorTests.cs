using NUnit.Framework;
using Lib.MathCore;

namespace Lib.MathCore.Tests;

public class SoftmaxCalculatorTests
{
    [Test]
    public void Softmax_SumIsOne()
    {
        float[] logits = { 2f, 1f, 0f };

        var probs = SoftmaxCalculator.Softmax(logits);

        float sum = 0f;
        foreach (var p in probs)
            sum += p;

        Assert.That(sum, Is.EqualTo(1f).Within(1e-5));
    }

    [Test]
    public void Softmax_HigherLogit_HigherProbability()
    {
        float[] logits = { 3f, 1f };

        var probs = SoftmaxCalculator.Softmax(logits);

        Assert.That(probs[0], Is.GreaterThan(probs[1]));
    }
}