using Lib.Models.TinyTransformer.Configuration;
using Lib.Models.TinyTransformer.Factories;
using Lib.Models.TinyTransformer.State;
using System.Text.Json;

namespace Lib.Models.TinyTransformer.Tests.Factory;

public class TinyTransformerModelFactoryTest
{
    private TinyTransformerModelFactory _factory;

    [SetUp]
    public void SetUp()
    {
        _factory = new TinyTransformerModelFactory();
    }

    [Test]
    public void Create_WithConfigAndWeights_ReturnsModel()
    {
        var config = new TinyTransformerConfig(10, 8, 1, 8);
        var weights = TinyTransformerWeights.Initialize(10, 8, new Random(42));

        var model = _factory.Create(config, weights);

        Assert.That(model, Is.Not.Null);
        Assert.That(model.VocabSize, Is.EqualTo(10));
    }

    [Test]
    public void Create_WithVocabSize_ReturnsModel()
    {
        var model = _factory.Create(100);

        Assert.That(model, Is.Not.Null);
        Assert.That(model.VocabSize, Is.EqualTo(100));
    }

    [Test]
    public void Create_WithSeed_ReturnsDeterministicModel()
    {
        var model1 = _factory.Create(50, seed: 123);
        var model2 = _factory.Create(50, seed: 123);

        var scores1 = model1.NextTokenScores(new int[] { 0 });
        var scores2 = model2.NextTokenScores(new int[] { 0 });

        Assert.That(scores1, Is.EqualTo(scores2));
    }

    [TestCase(100, 16, 1, 8)]
    [TestCase(500, 32, 2, 16)]
    [TestCase(1000, 64, 4, 32)]
    public void Create_VariousParams_ReturnsModelWithCorrectConfig(int vocab, int embed, int heads, int ctx)
    {
        var model = _factory.Create(vocab, embed, heads, ctx);

        Assert.That(model.VocabSize, Is.EqualTo(vocab));
        Assert.That(model.ContextSize, Is.EqualTo(ctx));
    }

    [Test]
    public void CreateFromPayload_ValidPayload_ReturnsModel()
    {
        var config = new TinyTransformerConfig(5, 4, 1, 4);
        var weights = TinyTransformerWeights.Initialize(5, 4, new Random(42));
        var originalModel = new TinyTransformerModel(config, weights);

        var payload = originalModel.ToPayload();
        var json = JsonSerializer.Serialize(payload);
        var element = JsonDocument.Parse(json).RootElement;

        var restoredModel = _factory.CreateFromPayload(element);

        Assert.That(restoredModel.VocabSize, Is.EqualTo(5));
        Assert.That(restoredModel.ContextSize, Is.EqualTo(4));
    }

    [Test]
    public void CreateFromPayload_ValidPayload_PreservesScores()
    {
        var config = new TinyTransformerConfig(5, 4, 1, 4);
        var weights = TinyTransformerWeights.Initialize(5, 4, new Random(42));
        var originalModel = new TinyTransformerModel(config, weights);

        var payload = originalModel.ToPayload();
        var json = JsonSerializer.Serialize(payload);
        var element = JsonDocument.Parse(json).RootElement;

        var restoredModel = _factory.CreateFromPayload(element);

        var originalScores = originalModel.NextTokenScores(new int[] { 0, 1 });
        var restoredScores = restoredModel.NextTokenScores(new int[] { 0, 1 });

        for (int i = 0; i < originalScores.Length; i++)
        {
            Assert.That(restoredScores[i], Is.EqualTo(originalScores[i]).Within(1e-5f));
        }
    }

    [Test]
    public void CreateFromPayload_InvalidPayload_ThrowsException()
    {
        var emptyPayload = JsonDocument.Parse("{}").RootElement;

        Assert.Throws<ArgumentException>(() => _factory.CreateFromPayload(emptyPayload));
    }
}
