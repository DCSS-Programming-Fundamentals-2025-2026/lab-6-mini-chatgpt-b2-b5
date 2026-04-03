using Lib.Models.TinyNN.Configuration;
using Lib.Models.TinyNN.State;
using Lib.Models.TinyNN.Layers;

namespace Lib.Models.TinyNN.Tests.Layers;

public class LinearHeadTest
{
    private int _vocabSize = 10;
    private TinyNNConfig _config;
    private TinyNNWeights _weights;
    private EmbeddingLayer _embeddinglayer;
    private LinearHead _linearhead;

    [SetUp]
    public void SetUp()
    {
        _config = new TinyNNConfig(_vocabSize);
        _weights = new TinyNNWeights(_vocabSize, _config.EmbeddingSize);
        _embeddinglayer = new EmbeddingLayer(_vocabSize, _config, _weights); 
        _linearhead = new LinearHead(_vocabSize, _config, _weights);
    }

    [Test]
    public void Project_ValidHiddenVector_ReturnsLogitsWithVocabSizeLength()
    {
        int[] context = new int[] {0, 3, 5, 2, 7};
        float[] hidden = _embeddinglayer.EncodeContext(context);
        float[] logits = _linearhead.Project(hidden);

        Assert.That(logits.Length, Is.EqualTo(_vocabSize));
    }

    [Test]
    public void AddBiasToVector_ZeroInputVector_ReturnsOutputBiasValues()
    {
        float[] zeroVector = new float[_weights.OutputBias.Length];
        float[] expected = _weights.OutputBias;

        float[] result = _linearhead.AddBiasToVector(zeroVector);

        Assert.That(result, Is.EqualTo(expected));
    }

    [Test]
    public void MultiplyHiddenOnWeights_UnitVectorInput_ReturnsCorrespondingWeightsRow()
    {
        float[] hidden = new float[_config.EmbeddingSize];
        hidden[0] = 1.0f;
        
        float[] expectedRow = _weights.OutputWeights[0];
        float[] result = _linearhead.MultiplyHiddenOnWeights(hidden);

        Assert.That(result, Is.EqualTo(expectedRow));
    }

    [Test]
    public void Backward_ValidInputs_UpdatesOutputWeightsCorrectly()
    {
        float[] hidden =
        {
            0.4f, -0.7f, 0.2f, 1.1f, -0.3f, 0.6f, 0.9f, -0.5f, 0.8f, -0.2f, 0.1f, 0.3f, -0.6f, 0.7f, -0.4f, 0.5f,
            0.2f, -0.9f, 1.0f, -0.1f, 0.6f, 0.4f, -0.8f, 0.3f, 0.7f, -0.2f, 0.5f, -0.6f, 0.9f, 0.1f, -0.4f, 0.8f
        };

        float[] dLogits = { 0.1f, 0.05f, 0.2f, -0.6f, 0.25f, 0.12f, -0.08f, 0.33f, -0.15f, 0.18f };

        float[][] oldWeights = new float[_weights.OutputWeights.Length][];
        for (int i = 0; i < _weights.OutputWeights.Length; i++)
        {
            oldWeights[i] = new float[_weights.OutputWeights[i].Length];
            Array.Copy(_weights.OutputWeights[i], oldWeights[i], _weights.OutputWeights[i].Length);
        }

        float lr = 0.01f;

        _linearhead.Backward(hidden, dLogits, lr);

        for (int i = 0; i < _weights.OutputWeights.Length; i++)
        {
            for (int j = 0; j < _weights.OutputWeights[i].Length; j++)
            {
                float absDelta = Math.Abs(hidden[i] * dLogits[j] * lr);
                float delta = Math.Abs(oldWeights[i][j] - _weights.OutputWeights[i][j]);

                Assert.That(delta, Is.EqualTo(absDelta).Within(0.000001f));
            }
        }
    }

    [Test]
    public void Backward_ValidInputs_UpdatesOutputBiasCorrectly()
    {
        float[] hidden =
        {
            0.4f, -0.7f, 0.2f, 1.1f, -0.3f, 0.6f, 0.9f, -0.5f, 0.8f, -0.2f, 0.1f, 0.3f, -0.6f, 0.7f, -0.4f, 0.5f,
            0.2f, -0.9f, 1.0f, -0.1f, 0.6f, 0.4f, -0.8f, 0.3f, 0.7f, -0.2f, 0.5f, -0.6f, 0.9f, 0.1f, -0.4f, 0.8f
        };

        float[] dLogits = { 0.1f, 0.05f, 0.2f, -0.6f, 0.25f, 0.12f, -0.08f, 0.33f, -0.15f, 0.18f };

        float[] oldBias = new float[_weights.OutputBias.Length];
        Array.Copy(_weights.OutputBias, oldBias, _weights.OutputBias.Length);

        float lr = 0.01f;

        _linearhead.Backward(hidden, dLogits, lr);

        for (int i = 0; i < _weights.OutputBias.Length; i++)
        {
            float absDelta = Math.Abs(dLogits[i] * lr);
            float delta = Math.Abs(oldBias[i] - _weights.OutputBias[i]);

            Assert.That(delta, Is.EqualTo(absDelta).Within(0.000001f));
        }
    }

    [Test]
    public void Backward_ValidInputs_ReturnsCorrectDHidden()
    {
        float[] hidden =
        {
            0.4f, -0.7f, 0.2f, 1.1f, -0.3f, 0.6f, 0.9f, -0.5f, 0.8f, -0.2f, 0.1f, 0.3f, -0.6f, 0.7f, -0.4f, 0.5f,
            0.2f, -0.9f, 1.0f, -0.1f, 0.6f, 0.4f, -0.8f, 0.3f, 0.7f, -0.2f, 0.5f, -0.6f, 0.9f, 0.1f, -0.4f, 0.8f
        };

        float[] dLogits = { 0.1f, 0.05f, 0.2f, -0.6f, 0.25f, 0.12f, -0.08f, 0.33f, -0.15f, 0.18f };
        float lr = 0.01f;

        float[] dHidden = _linearhead.Backward(hidden, dLogits, lr);

        for (int i = 0; i < _weights.OutputWeights.Length; i++)
        {
            float component = 0;
            for (int j = 0; j < _weights.OutputWeights[i].Length; j++)
            {
                component += _weights.OutputWeights[i][j] * dLogits[j];
            }

            Assert.That(dHidden[i], Is.EqualTo(component).Within(0.000001f));
        }
    }
}