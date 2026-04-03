namespace Lib.Training.Scheduling
{
    public static class CheckpointScheduler
    {
        public static bool ScheduleCheck(int currentEpoch, int checkpointInterval, int totalEpochs)
        {
            if(currentEpoch < 1 || checkpointInterval < 1 || totalEpochs < 1)
            {
                return false;
            }
            
            else if(currentEpoch % checkpointInterval == 0)
            {
                return true;
            }

            else if(currentEpoch == totalEpochs)
            {
                return true;
            }

            return false;
        }
    }
}