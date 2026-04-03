namespace Lib.Models.TinyNN.Configuration;

public class TinyNNConfig 
{
    public int VocabSize { get; set; } 
    public int EmbeddingSize { get; } 
    public int ContextSize { get; } 

    public TinyNNConfig(int vocabSize, int embeddingSize = 32, int contextSize = 8192)
    {
        VocabSize = vocabSize;
        EmbeddingSize = embeddingSize;
        ContextSize = contextSize;
    }
}