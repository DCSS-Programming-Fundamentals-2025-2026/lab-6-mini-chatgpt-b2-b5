using System.Text.Json.Serialization;

namespace Lib.Models.TinyNN.State;

public class TinyNNWeights
{
    public float[][] Embeddings { get; private set; } 
    public float[][] OutputWeights { get; private set; } 
    public float[] OutputBias { get; private set; } 


    [JsonConstructor]
    public TinyNNWeights(float[][] embeddings, float[][] outputWeights, float[] outputBias)
    {
        Embeddings = embeddings;
        OutputWeights = outputWeights;
        OutputBias = outputBias;
    }
    
    public TinyNNWeights (int vocabSize, int embeddingSize)
    {
        Embeddings = InitializeEmbeddings(vocabSize, embeddingSize);
        OutputWeights = InitializeOutputWeights(vocabSize, embeddingSize);
        OutputBias = new float[vocabSize];
    }

    private float[][] InitializeEmbeddings (int vocabSize, int embeddingSize)
    {
        Random random = new Random();

        float[][] array = new float[vocabSize][];
        for (int i = 0; i < vocabSize; i++)
        {
            array[i] = new float[embeddingSize];

            for (int j = 0; j < embeddingSize; j++)
            {
                array[i][j] = (float)(random.NextDouble() * 0.2 - 0.1);
            }
        }

        return array;
    }

    private float[][] InitializeOutputWeights(int vocabSize, int embeddingSize)
    {
        Random random = new Random();

        float[][] array = new float[embeddingSize][];
        for (int i = 0; i < embeddingSize; i++)
        {
            array[i] = new float[vocabSize];

            for (int j = 0; j < vocabSize; j++)
            {
                array[i][j] = (float)(random.NextDouble() * 0.2 - 0.1);
            }
        }

        return array;
    }
}