using NUnit.Framework;
using Lib.MathCore;

namespace Lib.MathCore.Tests;

public class ProbabilitySamplerTests
{
    [Test]
    public void SampleFromProbs_ReturnsValidIndex()
    {
        float[] probs = { 0.1f, 0.7f, 0.2f };
        var rng = new Random(42);

        int index = ProbabilitySampler.SampleFromProbs(probs, rng);

        Assert.That(index, Is.InRange(0, probs.Length - 1));
    }

    [Test]
    public void SampleFromProbs_HighProbabilityAlwaysChosen()
    {
        float[] probs = { 0f, 1f, 0f };
        var rng = new Random(42);

        int index = ProbabilitySampler.SampleFromProbs(probs, rng);

        Assert.That(index, Is.EqualTo(1));
    }
}