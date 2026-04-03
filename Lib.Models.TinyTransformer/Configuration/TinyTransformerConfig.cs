namespace Lib.Models.TinyTransformer.Configuration
{
    public class TinyTransformerConfig
    {
        public int VocabSize { get; set; }
        public int EmbeddingSize { get; set; } = 16;
        public int HeadCount { get; set; } = 1;
        public int ContextSize { get; set; } = 8;

        public TinyTransformerConfig()
        {
        }

        public TinyTransformerConfig(int vocabSize)
        {
            VocabSize = vocabSize;
        }

        public TinyTransformerConfig(int vocabSize, int embeddingSize, int headCount, int contextSize)
        {
            VocabSize = vocabSize;
            EmbeddingSize = embeddingSize;
            HeadCount = headCount;
            ContextSize = contextSize;
        }
    }
}
