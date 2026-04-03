namespace Lib.Training.Configuration
{
    public class TrainingConfig
    {
        public int Epochs {get; set;}
        public float LearningRate {get; set;}
        public int CheckpointInterval {get; set;}

        public TrainingConfig (int epochs, float learningRate, int checkpointInterval)
        {
            Epochs = epochs;
            LearningRate = learningRate;
            CheckpointInterval = checkpointInterval;
        }
    }
}