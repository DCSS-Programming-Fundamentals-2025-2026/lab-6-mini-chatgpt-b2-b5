using System.Text.Json;

public class NGramPayloadMapperTests
{
    [Test]
    public void FromBigramToJsonAndBack()
    {
        // Arrange
        NGramModel model = new NGramModel(4);
        int[] tokens = { 3, 4, 1, 4, 2, 3, 1, 3, 4, 2 };
        int[] context = { 1, 4 };

        model.Train(tokens);
        float[] beforeSerialization = model.NextTokenScores(context);

        NGramPayloadMapper mapper = new NGramPayloadMapper();
        NGramModel newModel = new NGramModel(1); 

        // Act
        JsonElement json = mapper.FromBigramToJson(model);
        mapper.FromJsonElementToBigram(json, newModel);
        float[] afterSerialization = newModel.NextTokenScores(context);

        // Assert
        Assert.That(beforeSerialization[0], Is.EqualTo(afterSerialization[0]));
        Assert.That(beforeSerialization[1], Is.EqualTo(afterSerialization[1]));
        Assert.That(beforeSerialization[2], Is.EqualTo(afterSerialization[2]));
        Assert.That(beforeSerialization[3], Is.EqualTo(afterSerialization[3]));
    }

    [Test]
    public void FromTrigramToJsonAndBack()
    {
        // Arrange
        TrigramModel model = new TrigramModel(3);
        int[] tokens = { 3, 2, 1, 2, 1, 3, 2, 1, 2, 3, 2, 3, 1, 3, 2, 2 };
        int[] context = { 1, 3, 3, 2 };

        model.Train(tokens);
        float[] beforeSerialization = model.NextTokenScores(context);

        NGramPayloadMapper mapper = new NGramPayloadMapper();
        TrigramModel newModel = new TrigramModel(1);

        // Act
        JsonElement json = mapper.FromTrigramToJson(model);
        mapper.FromJsonElementToTrigram(json, newModel);
        float[] afterSerialization = newModel.NextTokenScores(context);

        // Assert
        Assert.That(beforeSerialization[0], Is.EqualTo(afterSerialization[0]));
        Assert.That(beforeSerialization[1], Is.EqualTo(afterSerialization[1]));
        Assert.That(beforeSerialization[2], Is.EqualTo(afterSerialization[2]));
    }

    [Test]
    public void FromTrigramToJsonAndBack_BigramFallBack()
    {
        // Arrange
        TrigramModel model = new TrigramModel(3);
        int[] tokens = { 3, 2, 1, 2, 1, 3, 2, 1, 2, 3, 2, 3, 1, 3, 2, 2 };
        int[] context = { 1 };

        model.Train(tokens);
        float[] beforeSerialization = model.NextTokenScores(context);

        NGramPayloadMapper mapper = new NGramPayloadMapper();
        TrigramModel newModel = new TrigramModel(1);

        // Act
        JsonElement json = mapper.FromTrigramToJson(model);
        mapper.FromJsonElementToTrigram(json, newModel);
        float[] afterSerialization = newModel.NextTokenScores(context);

        // Assert
        Assert.That(beforeSerialization[0], Is.EqualTo(afterSerialization[0]));
        Assert.That(beforeSerialization[1], Is.EqualTo(afterSerialization[1]));
        Assert.That(beforeSerialization[2], Is.EqualTo(afterSerialization[2]));
    }
}

