public class PerplexityCalculatorTests
{
    [Test]
    public void ComputePerplexityBigram_NotEnoughTokens()
    {
        // Arrange
        PerplexityCalculator calculator = new PerplexityCalculator();
        NGramModel model = new NGramModel(3);
        int[] tokens = { 2 };

        // Act
        float result = calculator.ComputePerplexityBigram(model, tokens);

        // Assert
        Assert.That(result, Is.EqualTo(float.PositiveInfinity));
    }

    [Test]
    public void ComputePerplexityTrigram_NotEnoughTokens()
    {
        // Arrange
        PerplexityCalculator calculator = new PerplexityCalculator();
        TrigramModel model = new TrigramModel(3);
        int[] tokens = { };

        // Act
        float result = calculator.ComputePerplexityTrigram(model, tokens);

        // Assert
        Assert.That(result, Is.EqualTo(float.PositiveInfinity));
    }

    [Test]
    public void ComputePerplexityBigram_ZeroProbs()
    {
        // Arrange
        PerplexityCalculator calculator = new PerplexityCalculator();
        NGramModel model = new NGramModel(3);
        int[] tokens = { 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 3 };
        model.Train(tokens);

        tokens = new int[] { 2, 3}; 

        // Act
        float result = calculator.ComputePerplexityBigram(model, tokens);

        // Assert
        Assert.IsTrue(result > 0);
    }

    [Test]
    public void ComputePerplexityTrigram_ZeroProbs()
    {
        // Arrange
        PerplexityCalculator calculator = new PerplexityCalculator();
        TrigramModel model = new TrigramModel(4);
        int[] tokens = { 2, 4, 1, 2, 4, 2, 1, 4, 1, 4, 3};
        model.Train(tokens);

        tokens = new int[] { 4, 3 };

        // Act
        float result = calculator.ComputePerplexityTrigram(model, tokens);

        // Assert
        Assert.IsTrue(result > 0);
    }
}

