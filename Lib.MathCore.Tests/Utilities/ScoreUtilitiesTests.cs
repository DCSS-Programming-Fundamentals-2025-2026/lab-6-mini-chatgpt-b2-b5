using System;
using NUnit.Framework;
using Lib.MathCore.Utilities;

namespace Lib.MathCore.Tests.Utilities;

[TestFixture]
public class ScoreUtilitiesTests
{
    [Test]
    public void ArgMax_FindsCorrectIndex_InNormalArray()
    {
        float[] scores = { 0.1f, 0.8f, 0.3f, 0.9f, 0.2f };
        int result = ScoreUtilities.ArgMax(scores);
        Assert.That(result, Is.EqualTo(3)); 
    }

    [Test]
    public void ArgMax_ThrowsException_WhenArrayIsEmpty()
    {
        float[] emptyScores = Array.Empty<float>();

        void DangerousAction()
        {
            ScoreUtilities.ArgMax(emptyScores);
        }

        Assert.Throws<ArgumentException>(DangerousAction);
    }
}