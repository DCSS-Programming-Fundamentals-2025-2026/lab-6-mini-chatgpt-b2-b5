namespace Lib.Training.Metrics
{
    public class TrainingMetrics
    {
        public int CurrentEpoch { get; private set; }
        public float? AverageLoss { get; private set; }
        public float? Perplexity { get; private set; }
        public int? TotalSteps { get; private set; }
        public int? NGramCount { get; private set; }
        public TimeSpan ElapsedTime { get; private set; }

        public TrainingMetrics() 
        {
            AverageLoss = 0;
            Perplexity = 0;
            TotalSteps = 0;
            NGramCount = 0;
        }

        public void UpdateTinyNN(int currentEpoch, float averageLoss, int totalSteps, TimeSpan elapsedTime)
        {
            if(currentEpoch < 0 || averageLoss < 0 || totalSteps < 0 || elapsedTime < TimeSpan.Zero)
            {
                throw new ArgumentException("Invalid training metrics values!");
            }

            CurrentEpoch = currentEpoch;
            AverageLoss = averageLoss;
            TotalSteps = totalSteps;
            ElapsedTime += elapsedTime;
            Perplexity = null;
            NGramCount = null;
        }

        public void UpdateNGram(int currentEpoch, float perplexity, int nGramCount, TimeSpan elapsedTime)
        {
            if (currentEpoch < 0 || perplexity < 0 || nGramCount < 0 || elapsedTime < TimeSpan.Zero)
            {
                throw new ArgumentException("Invalid training metrics values!");
            }

            CurrentEpoch = currentEpoch;
            AverageLoss = null;
            TotalSteps = null;
            ElapsedTime += elapsedTime;
            Perplexity = perplexity;
            NGramCount = nGramCount;
        }
    }
}