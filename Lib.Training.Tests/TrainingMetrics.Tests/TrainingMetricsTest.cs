using System.Reflection;
using Lib.Training.Metrics;
namespace Lib.Training.Tests;

public class UpdateMetricsTest
{
    [TestCase(17, 4.3f, 23)]
    [TestCase(4, 1.1f, 128)]
    [TestCase(88, 0.12f, 11)]
    public void Update_IfValuesIsValid_MetricsPropertiesEqualToParameters(int currentEpoch, float averageLoss, int totalSteps)
    {
        TrainingMetrics trainingMetrics = new TrainingMetrics();
        trainingMetrics.UpdateTinyNN(currentEpoch, averageLoss, totalSteps, new TimeSpan(10, 12, 3));

        Assert.Multiple(() =>
        {
            Assert.That(trainingMetrics.CurrentEpoch, Is.EqualTo(currentEpoch));
            Assert.That(trainingMetrics.AverageLoss, Is.EqualTo(averageLoss));
            Assert.That(trainingMetrics.TotalSteps, Is.EqualTo(totalSteps));
        });
    }

    [TestCase(-10, 4.3f, 4)]
    [TestCase(3, -2.1f, -80)]
    [TestCase(-102, -0.221f, 67)]
    public void Update_IfValuesIsInvalid_ThrowArgumentException(int currentEpoch, float averageLoss, int totalSteps)
    {
        TrainingMetrics trainingMetrics = new TrainingMetrics();

        Assert.Throws<ArgumentException>(() => trainingMetrics.UpdateTinyNN(currentEpoch, averageLoss, totalSteps, new TimeSpan(0, 11, 34)));
    }
}