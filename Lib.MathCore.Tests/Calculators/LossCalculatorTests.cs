using NUnit.Framework;
using Lib.MathCore;

namespace Lib.MathCore.Tests;

public class LossCalculatorTests
{
    [Test]
    public void CrossEntropy_LowerWhenCorrectPrediction()
    {
        float[] logitsGood = { 1f, 3f };
        float[] logitsBad = { 3f, 1f };

        float lossGood = LossCalculator.CrossEntropyLoss(logitsGood, 1);
        float lossBad = LossCalculator.CrossEntropyLoss(logitsBad, 1);

        Assert.That(lossGood, Is.LessThan(lossBad));
    }

    [Test]
    public void CrossEntropy_PositiveValue()
    {
        float[] logits = { 1f, 2f };

        float loss = LossCalculator.CrossEntropyLoss(logits, 1);

        Assert.That(loss, Is.GreaterThan(0f));
    }
}