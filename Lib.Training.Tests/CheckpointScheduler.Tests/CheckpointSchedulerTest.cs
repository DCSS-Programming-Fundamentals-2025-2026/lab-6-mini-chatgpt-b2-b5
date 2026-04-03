using Lib.Training.Scheduling;
namespace Lib.Training.Tests;

public class CheckpointSchedulerTest
{
    [TestCase(10, 5, 40, true)]
    [TestCase(30, 8, 30, true)]
    [TestCase(12, 4, 90, true)]
    public void ScheduleCheck_IsTimeToCheckpoint_True(int currentEpoch, int checkpointInterval, int totalEpochs, bool expected)
    {
        bool res = CheckpointScheduler.ScheduleCheck(currentEpoch, checkpointInterval, totalEpochs);

        Assert.That(res, Is.EqualTo(expected));                                            
    }

    [TestCase(4, 10, 40, false)]                                   
    [TestCase(1, 8, 30, false)]
    [TestCase(99, 10, 100, false)]
    public void ScheduleCheck_IsNotTimeToCheckpoint_False(int currentEpoch, int checkpointInterval, int totalEpochs, bool expected)
    {
        bool res = CheckpointScheduler.ScheduleCheck(currentEpoch, checkpointInterval, totalEpochs);

        Assert.That(res, Is.EqualTo(expected)); 
    }

    [TestCase(-14, -2, 0, false)]
    [TestCase(13, -5, 30, false)]
    [TestCase(-1000, 10, 100, false)]
    public void ScheduleCheck_IsInvalidArguments_False(int currentEpoch, int checkpointInterval, int totalEpochs, bool expected)
    {
        bool res = CheckpointScheduler.ScheduleCheck(currentEpoch, checkpointInterval, totalEpochs);

        Assert.That(res, Is.EqualTo(expected)); 
    }
}